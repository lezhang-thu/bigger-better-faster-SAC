# coding=utf-8

import sys
import collections
import random
import copy
import functools
import itertools
import time
import scipy
import math

from absl import logging
from flax.core.frozen_dict import FrozenDict
import gin
import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax

from bbf import spr_networks
from bbf.replay_memory import subsequence_replay_buffer, circular_replay_buffer

NATURE_DQN_OBSERVATION_SHAPE = (84, 84)  # Size of downscaled Atari 2600 frame.
NATURE_DQN_DTYPE = np.uint8  # DType of Atari 2600 observations.
NATURE_DQN_STACK_SIZE = 4  # Number of frames in the state stack.


def project_distribution(supports, weights, target_support):
    """Projects a batch of (support, weights) onto target_support.

  Based on equation (7) in (Bellemare et al., 2017):
    https://arxiv.org/abs/1707.06887
  In the rest of the comments we will refer to this equation simply as Eq7.

  Args:
    supports: Jax array of shape (num_dims) defining supports for
      the distribution.
    weights: Jax array of shape (num_dims) defining weights on the
      original support points. Although for the CategoricalDQN agent these
      weights are probabilities, it is not required that they are.
    target_support: Jax array of shape (num_dims) defining support of the
      projected distribution. The values must be monotonically increasing. Vmin
      and Vmax will be inferred from the first and last elements of this Jax
      array, respectively. The values in this Jax array must be equally spaced.

  Returns:
    A Jax array of shape (num_dims) with the projection of a batch
    of (support, weights) onto target_support.

  Raises:
    ValueError: If target_support has no dimensions, or if shapes of supports,
      weights, and target_support are incompatible.
  """
    v_min, v_max = target_support[0], target_support[-1]
    # `N` in Eq7.
    num_dims = target_support.shape[0]
    # delta_z = `\Delta z` in Eq7.
    delta_z = (v_max - v_min) / (num_dims - 1)
    # clipped_support = `[\hat{T}_{z_j}]^{V_max}_{V_min}` in Eq7.
    clipped_support = jnp.clip(supports, v_min, v_max)
    # numerator = `|clipped_support - z_i|` in Eq7.
    numerator = jnp.abs(clipped_support - target_support[:, None])
    quotient = 1 - (numerator / delta_z)
    # clipped_quotient = `[1 - numerator / (\Delta z)]_0^1` in Eq7.
    clipped_quotient = jnp.clip(quotient, 0, 1)
    # inner_prod = `\sum_{j=0}^{N-1} clipped_quotient * p_j(x', \pi(x'))` in Eq7.
    inner_prod = clipped_quotient * weights
    return jnp.squeeze(jnp.sum(inner_prod, -1))


def softmax_cross_entropy_loss_with_logits(labels: jnp.array,
                                           logits: jnp.array) -> jnp.ndarray:
    """Implementation of the softmax cross entropy loss."""
    return -jnp.sum(labels * flax.linen.log_softmax(logits))


def prefetch_to_device(iterator, size):
    queue = collections.deque()

    def enqueue(n):  # Enqueues *up to* `n` elements from the iterator.
        for data in itertools.islice(iterator, n):
            #queue.append(jax.device_put(data, device=jax.local_devices()[0]))
            queue.append(data)

    enqueue(size)  # Fill up the buffer.
    while queue:
        yield queue.popleft()
        enqueue(1)


def copy_within_frozen_tree(old, new, prefix):
    new_entry = old[prefix].copy(add_or_replace=new)
    return old.copy(add_or_replace={prefix: new_entry})


def copy_params(source, target, keys=("encoder", "transition_model")):
    """Copies a set of keys from one set of params to another.

  Args:
    source: Set of parameters to take keys from.
    target: Set of parameters to overwrite keys in.
    keys: Set of keys to copy.

  Returns:
    A parameter dictionary of the same shape as target.
  """
    if (isinstance(source, dict) or
            isinstance(source, collections.OrderedDict) or
            isinstance(source, FrozenDict)):
        fresh_dict = {}
        for k, v in source.items():
            if k in keys:
                fresh_dict[k] = v
            else:
                fresh_dict[k] = copy_params(source[k], target[k], keys)
        return fresh_dict
    else:
        return target


@functools.partial(jax.jit, static_argnames=("keys", "strip_params_layer"))
def interpolate_weights(
    old_params,
    new_params,
    keys,
    old_weight=0.5,
    new_weight=0.5,
    strip_params_layer=True,
):
    """Interpolates between two parameter dictionaries.

  Args:
    old_params: The first parameter dictionary.
    new_params: The second parameter dictionary, of same shape and structure.
    keys: Which keys in the parameter dictionaries to interpolate. If None,
      interpolates everything.
    old_weight: The weight to place on the old dictionary.
    new_weight: The weight to place on the new dictionary.
    strip_params_layer: Whether to strip an outer "params" layer, as is often
      present in e.g., Flax.

  Returns:
    A parameter dictionary of the same shape as the inputs.
  """
    if strip_params_layer:
        old_params = old_params["params"]
        new_params = new_params["params"]

    def combination(old_param, new_param):
        return old_param * old_weight + new_param * new_weight

    combined_params = {}
    if keys is None:
        keys = old_params.keys()
    for k in keys:
        combined_params[k] = jax.tree_util.tree_map(combination, old_params[k],
                                                    new_params[k])
    for k, v in old_params.items():
        if k not in keys:
            combined_params[k] = v

    if strip_params_layer:
        combined_params = {"params": combined_params}
    return FrozenDict(combined_params)


@functools.partial(
    jax.jit,
    static_argnames=(
        "do_rollout",
        "state_shape",
        "keys_to_copy",
        "shrink_perturb_keys",
        "reset_target",
        "network_def",
        "optimizer",
    ),
)
def jit_reset(
    online_params,
    target_network_params,
    optimizer_state,
    network_def,
    optimizer,
    rng,
    state_shape,
    do_rollout,
    support,
    reset_target,
    shrink_perturb_keys,
    shrink_factor,
    perturb_factor,
    keys_to_copy,
):
    """A jittable function to reset network parameters.

  Args:
    online_params: Parameter dictionary for the online network.
    target_network_params: Parameter dictionary for the target network.
    optimizer_state: Optax optimizer state.
    network_def: Network definition.
    optimizer: Optax optimizer.
    rng: JAX PRNG key.
    state_shape: Shape of the network inputs.
    do_rollout: Whether to do a dynamics model rollout (e.g., if SPR is being
      used).
    support: Support of the categorical distribution if using distributional RL.
    reset_target: Whether to also reset the target network.
    shrink_perturb_keys: Parameter keys to apply shrink-and-perturb to.
    shrink_factor: Factor to rescale current weights by (1 keeps , 0 deletes).
    perturb_factor: Factor to scale random noise by in [0, 1].
    keys_to_copy: Keys to copy over without resetting.

  Returns:
  """
    online_rng, target_rng = jax.random.split(rng, 2)
    state = jnp.zeros(state_shape, dtype=jnp.float32)
    # Create some dummy actions of arbitrary length to initialize the transition
    # model, if the network has one.
    actions = jnp.zeros((5,))
    random_params = flax.core.frozen_dict.FrozenDict(
        network_def.init(
            online_rng,
            method=network_def.init_fn,
            x=state,
            actions=actions,
            do_rollout=do_rollout,
            support=support,
        ))
    target_random_params = flax.core.frozen_dict.FrozenDict(
        network_def.init(
            target_rng,
            method=network_def.init_fn,
            x=state,
            actions=actions,
            do_rollout=do_rollout,
            support=support,
        ))
    if shrink_perturb_keys:
        online_params = interpolate_weights(
            online_params,
            random_params,
            shrink_perturb_keys,
            old_weight=shrink_factor,
            new_weight=perturb_factor,
        )
    online_params = FrozenDict(
        copy_params(online_params, random_params, keys=keys_to_copy))

    updated_optim_state = []
    optim_state = optimizer.init(online_params)
    for i in range(len(optim_state)):
        optim_to_copy = copy_params(
            dict(optimizer_state[i]._asdict()),
            dict(optim_state[i]._asdict()),
            keys=keys_to_copy,
        )
        optim_to_copy = FrozenDict(optim_to_copy)
        updated_optim_state.append(optim_state[i]._replace(**optim_to_copy))
    optimizer_state = tuple(updated_optim_state)

    if reset_target:
        if shrink_perturb_keys:
            target_network_params = interpolate_weights(
                target_network_params,
                target_random_params,
                shrink_perturb_keys,
                old_weight=shrink_factor,
                new_weight=perturb_factor,
            )
        target_network_params = copy_params(target_network_params,
                                            target_random_params,
                                            keys=keys_to_copy)
        target_network_params = FrozenDict(target_network_params)

    return online_params, target_network_params, optimizer_state, random_params


def exponential_decay_scheduler(decay_period,
                                warmup_steps,
                                initial_value,
                                final_value,
                                reverse=False):
    """Instantiate a logarithmic schedule for a parameter.

  By default the extreme point to or from which values decay logarithmically
  is 0, while changes near 1 are fast. In cases where this may not
  be correct (e.g., lambda) pass reversed=True to get proper
  exponential scaling.

  Args:
      decay_period: float, the period over which the value is decayed.
      warmup_steps: int, the number of steps taken before decay starts.
      initial_value: float, the starting value for the parameter.
      final_value: float, the final value for the parameter.
      reverse: bool, whether to treat 1 as the asmpytote instead of 0.

  Returns:
      A decay function mapping step to parameter value.
  """
    if reverse:
        initial_value = 1 - initial_value
        final_value = 1 - final_value

    start = np.log(initial_value)
    end = np.log(final_value)

    if decay_period == 0:
        return lambda x: initial_value if x < warmup_steps else final_value

    def scheduler(step):
        steps_left = decay_period + warmup_steps - step
        bonus_frac = steps_left / decay_period
        bonus = np.clip(bonus_frac, 0.0, 1.0)
        new_value = bonus * (start - end) + end

        new_value = np.exp(new_value)
        if reverse:
            new_value = 1 - new_value
        return new_value

    return scheduler


@functools.partial(jax.jit, static_argnames=["network_def", "eval_mode"])
def select_action_from_r2(
    network_def,
    params,
    stoch,
    deter,
    rng,
    eval_mode,
):
    def logits_w_samples(stoch, deter, action_sample_key):
        return network_def.apply(
            params,
            stoch,
            deter,
            rngs={"action_sample": action_sample_key},
            method=network_def.get_policy_from_r2,
        )

    rng, key = jax.random.split(rng)
    key = jax.random.split(key, stoch.shape[0])
    logits, samples = jax.vmap(logits_w_samples,
                               in_axes=(0, 0, 0),
                               axis_name="batch")(stoch, deter, key)
    new_actions = jnp.where(eval_mode, jnp.argmax(logits, axis=-1), samples)
    return rng, new_actions, jax.nn.softmax(logits)


@functools.partial(jax.jit, static_argnames=("network_def", "dtype"))
def r2_world_model_observe(network_def, params, raw_state, prev_action, stoch,
                           deter, is_first, rng, dtype):
    state = spr_networks.process_inputs(raw_state,
                                        rng=rng,
                                        data_augmentation=False,
                                        dtype=dtype)
    stoch, deter, _ = network_def.apply(
        params,
        state,
        prev_action,
        stoch,
        deter,
        is_first,
        rng,
        eval_mode=True,
        method=network_def.r2_world_model_observe,
    )
    return stoch, deter


train_static_argnames = [
    'network_def',
    'optimizer',
    'data_augmentation',
    'dtype',
    'batch_size',
    'use_target_backups',
    'match_online_target_rngs',
    'target_eval_mode',
]


R2LaPropState = collections.namedtuple(
    "R2LaPropState",
    ["count", "exp_avg", "exp_avg_sq", "exp_avg_lr_1", "exp_avg_lr_2"],
)


def r2_laprop_init(params):
    return R2LaPropState(
        count=jnp.array(0, dtype=jnp.int32),
        exp_avg=jax.tree_util.tree_map(jnp.zeros_like, params),
        exp_avg_sq=jax.tree_util.tree_map(jnp.zeros_like, params),
        exp_avg_lr_1=jnp.array(0.0, dtype=jnp.float32),
        exp_avg_lr_2=jnp.array(0.0, dtype=jnp.float32),
    )


def r2_laprop_update(grads, state, params, learning_rate, beta1, beta2, eps,
                     warmup, agc, pmin):
    """JAX port of the LaProp + warmup + AGC update used by r2dreamer."""
    step = state.count + 1
    warmup = jnp.asarray(warmup, dtype=jnp.float32)
    base_lr = jnp.asarray(learning_rate, dtype=jnp.float32)
    lr_scale = jnp.where(warmup > 0, jnp.minimum(1.0, step / warmup), 1.0)
    lr = base_lr * lr_scale

    def agc_clip(grad, param):
        param_norm = jnp.linalg.norm(param)
        grad_norm = jnp.linalg.norm(grad)
        upper = agc * jnp.maximum(param_norm, pmin)
        scale = 1.0 / jnp.maximum(grad_norm / (upper + 1e-16), 1.0)
        return grad * scale

    grads = jax.tree_util.tree_map(agc_clip, grads, params)
    exp_avg_sq = jax.tree_util.tree_map(
        lambda sq, grad: beta2 * sq + (1.0 - beta2) * grad * grad,
        state.exp_avg_sq,
        grads,
    )
    exp_avg_lr_1 = state.exp_avg_lr_1 * beta1 + (1.0 - beta1) * lr
    exp_avg_lr_2 = state.exp_avg_lr_2 * beta2 + (1.0 - beta2)
    bias_correction1 = jnp.where(lr != 0.0, exp_avg_lr_1 / lr, 1.0)
    step_size = 1.0 / bias_correction1

    def update_avg(avg, sq, grad):
        denom = jnp.sqrt(sq / exp_avg_lr_2) + eps
        grad_step = grad / denom
        return beta1 * avg + (1.0 - beta1) * lr * grad_step

    exp_avg = jax.tree_util.tree_map(update_avg, state.exp_avg, exp_avg_sq,
                                     grads)
    updates = jax.tree_util.tree_map(lambda avg: -step_size * avg, exp_avg)
    new_state = R2LaPropState(
        count=step,
        exp_avg=exp_avg,
        exp_avg_sq=exp_avg_sq,
        exp_avg_lr_1=exp_avg_lr_1,
        exp_avg_lr_2=exp_avg_lr_2,
    )
    return updates, new_state


def zero_param_subtrees_except(grads, keys_to_keep):
    keys_to_keep = frozenset(keys_to_keep)
    params = {
        key: value if key in keys_to_keep else jax.tree_util.tree_map(
            jnp.zeros_like, value)
        for key, value in grads["params"].items()
    }
    return flax.core.freeze(replace_mapping(grads, {"params": params}))


def replace_mapping(mapping, replacements):
    if isinstance(mapping, FrozenDict):
        return mapping.copy(add_or_replace=replacements)
    updated = dict(mapping)
    updated.update(replacements)
    return updated


def select_mapping(mapping, keys):
    return flax.core.freeze({key: mapping[key] for key in keys if key in mapping})


def train(
    network_def,  # 0, static
    online_params,  # 1
    target_params,  # 2
    optimizer,  # 3, static
    optimizer_state,  # 4
    r2wm_optimizer_state,
    support,
    discount,
    rng,
    data_augmentation,  # static
    dtype,  # static
    batch_size,  # static
    use_target_backups,  # static
    target_update_tau,
    target_update_every,
    step,
    match_online_target_rngs,  # static
    target_eval_mode,  # static
    #ent_targ,
    x_ent_coef,
    r2wm_raw_states,
    r2wm_actions,
    r2wm_rewards,
    r2wm_terminals,
    r2wm_is_first,
    r2wm_initial_stoch,
    r2wm_initial_deter,
    r2wm_learning_rate,
    r2wm_beta1,
    r2wm_beta2,
    r2wm_eps,
    r2wm_warmup,
    r2wm_agc,
    r2wm_pmin,
):
    online_params = flax.core.freeze(online_params)
    target_params = flax.core.freeze(target_params)

    @functools.partial(
        jax.jit,
        donate_argnums=(0,),
    )
    def train_one_batch(state, inputs):
        """Runs a training step."""
        # Unpack inputs from scan
        (
            online_params,
            target_params,
            optimizer_state,
            r2wm_optimizer_state,
            rng,
            step,
        ) = state
        (
            r2wm_raw_states,
            r2wm_actions,
            r2wm_rewards,
            r2wm_terminals,
            r2wm_is_first,
            r2wm_initial_stoch,
            r2wm_initial_deter,
        ) = inputs

        flat_transition_count = r2wm_raw_states.shape[0] * r2wm_raw_states.shape[1]
        rng, rng2 = jax.random.split(rng, num=2)

        def policy_target_r2(stoch, deter, action_sample_key):
            return network_def.apply(
                target_params,
                stoch,
                deter,
                rngs={"action_sample": action_sample_key},
                method=network_def.get_policy_from_r2,
            )

        def q_target_r2(stoch, deter):
            return network_def.apply(
                target_params,
                stoch,
                deter,
                support,
                eval_mode=target_eval_mode,
                method=network_def.q_from_r2_features,
            )

        def loss_fn(
            params,
            key,
            target_rng,
            r2wm_key,
            r2wm_aug_key,
        ):

            def r2_q_results(stoch, deter):
                return network_def.apply(
                    params,
                    stoch,
                    deter,
                    support,
                    eval_mode=False,
                    method=network_def.q_from_r2_features,
                )

            def r2_policy(stoch, deter, action_sample_key):
                return network_def.apply(
                    params,
                    stoch,
                    deter,
                    rngs={"action_sample": action_sample_key},
                    method=network_def.get_policy_from_r2,
                )

            def policy_loss(q_values, logits, x_key):
                samples = jax.random.categorical(x_key, logits)

                log_prob = jax.nn.log_softmax(logits)
                prob = jax.nn.softmax(logits)
                q_values = q_values[samples] - (q_values * prob).sum()
                ent_coef = network_def.apply(params,
                                             method=network_def.entropy_scale)
                x_ent = -(prob * log_prob).sum()
                #if True:
                if False:
                    return -(jax.lax.stop_gradient(q_values) * log_prob[samples]
                            ) + ent_coef * (-x_ent + ent_targ), x_ent
                else:
                    return -(jax.lax.stop_gradient(q_values) *
                             log_prob[samples]) + x_ent_coef * (-x_ent), x_ent

            r2wm_states = spr_networks.process_inputs(
                r2wm_raw_states,
                rng=r2wm_aug_key,
                data_augmentation=data_augmentation,
                dtype=dtype,
            )
            r2wm_loss, r2wm_metrics, post_stoch, post_deter = (
                network_def.apply(
                    params,
                    r2wm_states,
                    r2wm_actions,
                    r2wm_rewards,
                    r2wm_terminals,
                    r2wm_is_first,
                    r2wm_initial_stoch,
                    r2wm_initial_deter,
                    r2wm_key,
                    eval_mode=True,
                    method=network_def.r2_world_model_loss_from_states,
                ))
            r2wm_metrics["R2WMUpdate"] = jnp.array(1.0, dtype=jnp.float32)

            current_stoch = jnp.concatenate(
                [r2wm_initial_stoch[:, None], post_stoch[:, :-1]], axis=1)
            current_deter = jnp.concatenate(
                [r2wm_initial_deter[:, None], post_deter[:, :-1]], axis=1)
            current_stoch = jax.lax.stop_gradient(current_stoch)
            current_deter = jax.lax.stop_gradient(current_deter)
            next_stoch = jax.lax.stop_gradient(post_stoch)
            next_deter = jax.lax.stop_gradient(post_deter)

            flat_current_stoch = current_stoch.reshape(
                flat_transition_count, *current_stoch.shape[2:])
            flat_current_deter = current_deter.reshape(
                flat_transition_count, *current_deter.shape[2:])
            flat_next_stoch = next_stoch.reshape(flat_transition_count,
                                                 *next_stoch.shape[2:])
            flat_next_deter = next_deter.reshape(flat_transition_count,
                                                 *next_deter.shape[2:])
            flat_actions = jnp.argmax(r2wm_actions, axis=-1).reshape(-1)
            flat_rewards = r2wm_rewards[..., 0].reshape(-1)
            flat_terminals = r2wm_terminals[..., 0].reshape(-1)
            flat_discount = jnp.full_like(flat_rewards, discount)

            target = jax.vmap(target_output_r2,
                              in_axes=(None, None, 0, 0, 0, 0, None, 0, 0),
                              axis_name="batch")(
                                  policy_target_r2,
                                  q_target_r2,
                                  flat_next_stoch,
                                  flat_next_deter,
                                  flat_rewards,
                                  flat_terminals,
                                  support,
                                  flat_discount,
                                  target_rng,
                              )

            r2_x = jax.vmap(r2_q_results,
                            in_axes=(0, 0),
                            axis_name="batch")(flat_current_stoch,
                                               flat_current_deter)
            logits, _ = jax.vmap(r2_policy,
                                 in_axes=(0, 0, 0),
                                 axis_name="batch")(flat_current_stoch,
                                                    flat_current_deter, key)
            q_values_for_policy = r2_x.q_values
            q_logits = jnp.squeeze(r2_x.logits)
            chosen_action_logits = q_logits[jnp.arange(q_logits.shape[0]),
                                            flat_actions]
            dqn_loss = jax.vmap(softmax_cross_entropy_loss_with_logits)(
                target, chosen_action_logits)
            td_error = dqn_loss + jnp.nan_to_num(
                target * jnp.log(target)).sum(-1)

            mean_loss = jnp.mean(dqn_loss)
            policy_aux = jax.vmap(policy_loss, in_axes=0, axis_name="batch")(
                q_values_for_policy, logits, key)
            aux_losses = {
                "TotalLoss": jnp.mean(mean_loss),
                "DQNLoss": jnp.mean(dqn_loss),
                "TD Error": jnp.mean(td_error),
                "SPRLoss": jnp.array(0.0, dtype=jnp.float32),
                "ent": jnp.mean(policy_aux[1]),
            }
            total_loss = mean_loss + jnp.mean(policy_aux[0])
            total_loss = total_loss + r2wm_loss
            aux_losses.update(r2wm_metrics)

            return total_loss, (aux_losses, (post_stoch, post_deter))

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        rng2, r2wm_key, r2wm_aug_key, policy_key, target_key = jax.random.split(
            rng2, 5)
        split_policy_key = jax.random.split(policy_key,
                                            flat_transition_count + 1)
        rng2 = split_policy_key[0]
        key = split_policy_key[1:]
        if match_online_target_rngs:
            target_rng = key
        else:
            target_rng = jax.random.split(target_key, flat_transition_count)
        (_, (aux_losses, r2wm_posts)), grad = grad_fn(
            online_params,
            key,
            target_rng,
            r2wm_key,
            r2wm_aug_key,
        )

        bbf_optimizer_keys = (
            "r2_feature_projection",
            "r2_head",
            "r2_policy_projection",
            "r2_policy",
        )
        bbf_grad = zero_param_subtrees_except(grad, bbf_optimizer_keys)
        updates, new_optimizer_state = optimizer.update(
            bbf_grad, optimizer_state, params=online_params)
        new_online_params = optax.apply_updates(online_params, updates)

        r2_laprop_keys = ("encoder", "representation_projection",
                          "r2_world_model")
        r2_laprop_grads = select_mapping(grad["params"], r2_laprop_keys)
        r2_laprop_params = select_mapping(online_params["params"],
                                          r2_laprop_keys)
        r2wm_updates, r2wm_optimizer_state = r2_laprop_update(
            r2_laprop_grads,
            r2wm_optimizer_state,
            r2_laprop_params,
            r2wm_learning_rate,
            r2wm_beta1,
            r2wm_beta2,
            r2wm_eps,
            r2wm_warmup,
            r2wm_agc,
            r2wm_pmin,
        )
        new_r2_laprop_params = optax.apply_updates(r2_laprop_params,
                                                   r2wm_updates)
        new_online_params = replace_mapping(
            new_online_params, {
                "params": replace_mapping(new_online_params["params"],
                                          dict(new_r2_laprop_params))
            })
        new_online_params = flax.core.freeze(new_online_params)

        optimizer_state = new_optimizer_state
        online_params = new_online_params

        target_update_step = functools.partial(
            interpolate_weights,
            keys=None,
            old_weight=1 - target_update_tau,
            new_weight=target_update_tau,
        )
        target_params = jax.lax.cond(
            step % target_update_every == 0,
            target_update_step,
            lambda old, new: old,
            target_params,
            online_params,
        )

        return (
            (
                online_params,
                target_params,
                optimizer_state,
                r2wm_optimizer_state,
                rng2,
                step + 1,
            ),
            (aux_losses, r2wm_posts),
        )

    init_state = (
        online_params,
        target_params,
        optimizer_state,
        r2wm_optimizer_state,
        rng,
        step,
    )
    assert r2wm_raw_states.shape[0] % batch_size == 0
    num_batches = r2wm_raw_states.shape[0] // batch_size

    inputs = (
        r2wm_raw_states.reshape(num_batches, -1, *r2wm_raw_states.shape[1:]),
        r2wm_actions.reshape(num_batches, -1, *r2wm_actions.shape[1:]),
        r2wm_rewards.reshape(num_batches, -1, *r2wm_rewards.shape[1:]),
        r2wm_terminals.reshape(num_batches, -1, *r2wm_terminals.shape[1:]),
        r2wm_is_first.reshape(num_batches, -1, *r2wm_is_first.shape[1:]),
        r2wm_initial_stoch.reshape(num_batches, -1,
                                   *r2wm_initial_stoch.shape[1:]),
        r2wm_initial_deter.reshape(num_batches, -1,
                                   *r2wm_initial_deter.shape[1:]),
    )

    (
        (
            online_params,
            target_params,
            optimizer_state,
            r2wm_optimizer_state,
            rng,
            step,
        ),
        (aux_losses, r2wm_posts),
    ) = jax.lax.scan(train_one_batch, init_state, inputs)

    return (
        online_params,
        target_params,
        optimizer_state,
        r2wm_optimizer_state,
        {k: jnp.reshape(v, (-1,)) for k, v in aux_losses.items()},
        r2wm_posts[0],
        r2wm_posts[1],
    )


def target_output_r2(
    policy_info,
    target_network,
    next_stoch,
    next_deter,
    rewards,
    terminals,
    support,
    cumulative_gamma,
    rng,
):
    gamma_with_terminal = (cumulative_gamma *
                           (1.0 - terminals.astype(jnp.float32)))
    target_dist = target_network(next_stoch, next_deter)
    _, next_qt_argmax = policy_info(next_stoch, next_deter, rng)

    probabilities = jnp.squeeze(target_dist.probabilities)
    next_probabilities = probabilities[next_qt_argmax]
    target_support = rewards + gamma_with_terminal * support
    target = project_distribution(target_support, next_probabilities, support)

    return jax.lax.stop_gradient(target)


@gin.configurable
def create_scaling_optimizer(
    learning_rate=6.25e-5,
    beta1=0.9,
    beta2=0.999,
    eps=1.5e-4,
    centered=False,
    weight_decay=0.0,
):
    logging.info(
        ("Creating AdamW optimizer with settings lr=%f, beta1=%f, "
         "beta2=%f, eps=%f, wd=%f"),
        learning_rate,
        beta1,
        beta2,
        eps,
        weight_decay,
    )
    mask = lambda p: jax.tree_util.tree_map(lambda x: x.ndim != 1, p)
    return optax.adamw(
        learning_rate,
        b1=beta1,
        b2=beta2,
        eps=eps,
        weight_decay=weight_decay,
        mask=mask,
    )


@gin.configurable
class JaxDQNAgent(object):

    def __init__(
        self,
        num_actions,
        observation_shape=NATURE_DQN_OBSERVATION_SHAPE,
        observation_dtype=NATURE_DQN_DTYPE,
        stack_size=NATURE_DQN_STACK_SIZE,
        network=None,
        gamma=0.99,
        update_horizon=1,
        min_replay_history=20000,
        update_period=4,
        target_update_period=8000,
        epsilon_train=0.01,
        epsilon_eval=0.001,
        epsilon_decay_period=250000,
        eval_mode=False,
        optimizer='adam',
        allow_partial_reload=False,
        seed=None,
        loss_type='mse',
        preprocess_fn=None,
    ):
        assert isinstance(observation_shape, tuple)
        seed = int(time.time() * 1e6) if seed is None else seed
        logging.info('Creating %s agent with the following parameters:',
                     self.__class__.__name__)
        logging.info('\t gamma: %f', gamma)
        logging.info('\t update_horizon: %f', update_horizon)
        logging.info('\t min_replay_history: %d', min_replay_history)
        logging.info('\t update_period: %d', update_period)
        logging.info('\t target_update_period: %d', target_update_period)
        logging.info('\t optimizer: %s', optimizer)
        logging.info('\t seed: %d', seed)
        logging.info('\t loss_type: %s', loss_type)
        logging.info('\t preprocess_fn: %s', preprocess_fn)
        logging.info('\t allow_partial_reload: %s', allow_partial_reload)

        self.num_actions = num_actions
        self.observation_shape = tuple(observation_shape)
        self.observation_dtype = observation_dtype
        self.stack_size = stack_size
        if preprocess_fn is None:
            self.network_def = network(num_actions=num_actions)
            self.preprocess_fn = lambda x: x
        else:
            self.network_def = network(num_actions=num_actions,
                                       inputs_preprocessed=True)
            self.preprocess_fn = preprocess_fn
        self.gamma = gamma
        self.update_horizon = update_horizon
        self.cumulative_gamma = math.pow(gamma, update_horizon)
        self.min_replay_history = min_replay_history
        self.target_update_period = target_update_period
        self.update_period = update_period
        self.eval_mode = eval_mode
        self.training_steps = 0
        self.allow_partial_reload = allow_partial_reload
        self._loss_type = loss_type

        self._rng = jax.random.PRNGKey(seed)
        state_shape = self.observation_shape + (stack_size,)
        self.state = np.zeros(state_shape)
        self._replay = self._build_replay_buffer()
        self._optimizer_name = optimizer
        self._build_networks_and_optimizer()

        # Variables to be initialized by the agent once it interacts with the
        # environment.
        self._observation = None
        self._last_observation = None


@gin.configurable
class BBFAgent(JaxDQNAgent):
    """A compact implementation of the full Rainbow agent."""

    def __init__(
        self,
        num_actions,
        double_dqn=True,
        distributional=True,
        data_augmentation=False,
        num_updates_per_train_step=1,
        network=spr_networks.RainbowDQNNetwork,
        num_atoms=51,
        vmax=10.0,
        vmin=None,
        jumps=0,
        spr_weight=0,
        batch_size=32,
        replay_ratio=64,
        batches_to_group=1,
        update_horizon=10,
        max_update_horizon=None,
        min_gamma=None,
        reset_every=-1,
        no_resets_after=-1,
        reset_offset=1,
        learning_rate=0.0001,
        encoder_learning_rate=0.0001,
        reset_target=True,
        reset_head=True,
        reset_projection=True,
        reset_encoder=False,
        reset_interval_scaling=None,
        shrink_perturb_keys="",
        perturb_factor=0.2,  # original was 0.1
        shrink_factor=0.8,  # original was 0.4
        target_update_tau=1.0,
        max_target_update_tau=None,
        cycle_steps=0,
        target_update_period=1,
        target_action_selection=False,
        use_target_network=True,
        match_online_target_rngs=True,
        target_eval_mode=False,
        offline_update_frac=0,
        half_precision=False,
        seed=None,
        log_every=None,
        explore_end_steps=None,
        r2_world_model_batch_length=64,
        r2_world_model_replay_capacity=500000,
        r2_world_model_learning_rate=4e-5,
        r2_world_model_beta1=0.9,
        r2_world_model_beta2=0.999,
        r2_world_model_eps=1e-20,
        r2_world_model_warmup=1000,
        r2_world_model_agc=0.3,
        r2_world_model_pmin=1e-3,
        r2_world_model_stoch=32,
        r2_world_model_deter=6144,
        r2_world_model_hidden=768,
        r2_world_model_discrete=48,
        r2_world_model_units=768,
        r2_world_model_blocks=8,
    ):
        logging.info(
            "Creating %s agent with the following parameters:",
            self.__class__.__name__,
        )
        logging.info("\t double_dqn: %s", double_dqn)
        logging.info("\t distributional: %s", distributional)
        logging.info("\t data_augmentation: %s", data_augmentation)
        logging.info("\t num_updates_per_train_step: %d",
                     num_updates_per_train_step)
        # We need casting because passing arguments can convert ints to floats
        vmax = float(vmax)
        self._num_atoms = int(num_atoms)
        vmin = float(vmin) if vmin else -vmax
        self._support = jnp.linspace(vmin, vmax, self._num_atoms)
        self._double_dqn = bool(double_dqn)
        self._distributional = bool(distributional)
        self._data_augmentation = bool(data_augmentation)
        self._replay_ratio = int(replay_ratio)
        self._batch_size = int(batch_size)
        self._batches_to_group = int(batches_to_group)
        self.update_horizon = int(update_horizon)
        self._jumps = int(jumps)
        self.spr_weight = spr_weight

        self.reset_every = int(reset_every)
        self.reset_target = reset_target
        self.reset_head = reset_head
        self.reset_projection = reset_projection
        self.reset_encoder = reset_encoder
        self.offline_update_frac = float(offline_update_frac)
        self.no_resets_after = int(no_resets_after)
        self.cumulative_resets = 0
        self.reset_interval_scaling = reset_interval_scaling
        self.reset_offset = int(reset_offset)
        self.next_reset = self.reset_every + self.reset_offset

        self.learning_rate = learning_rate
        self.encoder_learning_rate = encoder_learning_rate

        self.shrink_perturb_keys = [
            s for s in shrink_perturb_keys.lower().split(",") if s
        ]
        self.shrink_perturb_keys = tuple(self.shrink_perturb_keys)
        self.shrink_factor = shrink_factor
        self.perturb_factor = perturb_factor

        self.target_action_selection = target_action_selection
        self.use_target_network = use_target_network
        self.match_online_target_rngs = match_online_target_rngs
        self.target_eval_mode = target_eval_mode

        # debug - start
        print('*' * 20)
        print(' self.target_eval_mode: {}'.format(self.target_eval_mode))
        print(' self.target_action_selection: {}'.format(
            self.target_action_selection))
        print(" num_actions: {}".format(num_actions))
        print(" self.reset_target: {}".format(self.reset_target))
        # debug - end

        self.grad_steps = 0
        self.cycle_grad_steps = 0
        self.target_update_period = int(target_update_period)
        self.target_update_tau = target_update_tau

        if max_update_horizon is None:
            self.max_update_horizon = self.update_horizon
            self.update_horizon_scheduler = lambda x: self.update_horizon
        else:
            self.max_update_horizon = int(max_update_horizon)
            n_schedule = exponential_decay_scheduler(
                cycle_steps, 0, 1,
                self.update_horizon / self.max_update_horizon)
            self.update_horizon_scheduler = lambda x: int(  # pylint: disable=g-long-lambda
                np.round(n_schedule(x) * self.max_update_horizon))

        self.max_target_update_tau = target_update_tau
        self.target_update_tau_scheduler = lambda x: self.target_update_tau

        logging.info("\t Found following local devices: %s",
                     str(jax.local_devices()))

        self.dtype = jnp.float32
        self.dtype_str = "float32"
        self.r2_world_model_batch_length = int(r2_world_model_batch_length)
        self.r2_world_model_replay_capacity = int(
            r2_world_model_replay_capacity)
        self.r2_world_model_learning_rate = float(
            r2_world_model_learning_rate)
        self.r2_world_model_beta1 = float(r2_world_model_beta1)
        self.r2_world_model_beta2 = float(r2_world_model_beta2)
        self.r2_world_model_eps = float(r2_world_model_eps)
        self.r2_world_model_warmup = int(r2_world_model_warmup)
        self.r2_world_model_agc = float(r2_world_model_agc)
        self.r2_world_model_pmin = float(r2_world_model_pmin)
        self.r2_world_model_stoch = int(r2_world_model_stoch)
        self.r2_world_model_deter = int(r2_world_model_deter)
        self.r2_world_model_hidden = int(r2_world_model_hidden)
        self.r2_world_model_discrete = int(r2_world_model_discrete)
        self.r2_world_model_units = int(r2_world_model_units)
        self.r2_world_model_blocks = int(r2_world_model_blocks)

        logging.info("\t Running with dtype %s", str(self.dtype))

        super().__init__(
            num_actions=num_actions,
            network=functools.partial(
                network,
                num_atoms=self._num_atoms,
                noisy=False,
                distributional=self._distributional,
                dtype=self.dtype,
                r2_world_model_stoch=self.r2_world_model_stoch,
                r2_world_model_deter=self.r2_world_model_deter,
                r2_world_model_hidden=self.r2_world_model_hidden,
                r2_world_model_discrete=self.r2_world_model_discrete,
                r2_world_model_units=self.r2_world_model_units,
                r2_world_model_blocks=self.r2_world_model_blocks,
            ),
            target_update_period=self.target_update_period,
            update_horizon=self.max_update_horizon,
            seed=seed,
        )

        self._build_r2_world_model_replay()
        self.set_replay_settings()

        if min_gamma is None or cycle_steps <= 1:
            self.min_gamma = self.gamma
            self.gamma_scheduler = lambda x: self.gamma
        else:
            self.min_gamma = min_gamma
            self.gamma_scheduler = exponential_decay_scheduler(cycle_steps,
                                                               0,
                                                               self.min_gamma,
                                                               self.gamma,
                                                               reverse=True)

        self.cumulative_gamma = (np.ones(
            (self.max_update_horizon,)) * self.gamma).cumprod()

        self.train_fn = jax.jit(train,
                                static_argnames=train_static_argnames,
                                device=jax.local_devices()[0])

        # debug - start
        self.greedy_action = False
        # debug - end
        self.stats_ent = 0
        self.explore_end_steps = explore_end_steps
        print('explore_end_steps: {}'.format(explore_end_steps))
        sys.stdout.flush()

    def _build_networks_and_optimizer(self):
        self._rng, rng = jax.random.split(self._rng)
        self.state_shape = self.state.shape

        # Create some dummy actions of arbitrary length to initialize the transition
        # model, if the network has one.
        actions = jnp.zeros((5,))
        self.online_params = flax.core.frozen_dict.FrozenDict(
            self.network_def.init(
                rng,
                method=self.network_def.init_fn,
                x=self.state.astype(self.dtype),
                actions=actions,
                do_rollout=self.spr_weight > 0,
                support=self._support,
            ))

        q_optim = create_scaling_optimizer(learning_rate=self.learning_rate,)
        policy_optim = create_scaling_optimizer(learning_rate=1e-4,)

        q_keys = {"r2_feature_projection", "r2_head"}
        q_mask = FrozenDict({
            "params": {k: k in q_keys for k in self.online_params["params"]}
        })

        policy_key = {"r2_policy_projection", "r2_policy"}
        policy_mask = FrozenDict({
            "params": {
                k: k in policy_key for k in self.online_params["params"]
            }
        })

        self.optimizer = optax.chain(
            optax.masked(q_optim, q_mask),
            optax.masked(policy_optim, policy_mask),
        )

        self.optimizer_state = self.optimizer.init(self.online_params)
        self.target_network_params = copy.deepcopy(self.online_params)
        self.random_params = copy.deepcopy(self.online_params)
        self.r2wm_optimizer_state = r2_laprop_init(
            select_mapping(self.online_params["params"],
                           ("encoder", "representation_projection",
                            "r2_world_model")))

        #print(' so far so good')
        #exit(0)

    def _build_replay_buffer(self):
        extra_storage_types = [
            circular_replay_buffer.ReplayElement('r2_index', (2,), np.int64)
        ]
        prioritized_buffer = subsequence_replay_buffer.PrioritizedJaxSubsequenceParallelEnvReplayBuffer(
            observation_shape=self.observation_shape,
            stack_size=self.stack_size,
            update_horizon=self.max_update_horizon,
            gamma=self.gamma,
            subseq_len=self._jumps + 1,
            batch_size=self._batch_size,
            observation_dtype=self.observation_dtype,
            extra_storage_types=extra_storage_types,
        )

        self.n_envs = prioritized_buffer._n_envs  # pylint: disable=protected-access
        self.start = time.time()
        return prioritized_buffer

    def _build_r2_world_model_replay(self):
        self._r2_pending_transition = None
        self._r2_stoch = np.zeros(
            (self.n_envs, self.r2_world_model_stoch,
             self.r2_world_model_discrete),
            dtype=np.float32)
        self._r2_deter = np.zeros((self.n_envs, self.r2_world_model_deter),
                                  dtype=np.float32)
        self._r2_prev_action = np.zeros((self.n_envs, self.num_actions),
                                        dtype=np.float32)
        self._r2_is_first = np.ones((self.n_envs, 1), dtype=np.float32)
        self._r2_last_added_index = np.full((self.n_envs, 2),
                                            -1,
                                            dtype=np.int64)
        from types import SimpleNamespace
        from buffer import Buffer

        config = SimpleNamespace(
            device="cpu",
            storage_device="cpu",
            batch_size=self._batch_size * self._batches_to_group,
            batch_length=self.r2_world_model_batch_length,
            max_size=self.r2_world_model_replay_capacity,
            num_actions=self.num_actions,
            stoch=self.r2_world_model_stoch,
            discrete=self.r2_world_model_discrete,
            deter=self.r2_world_model_deter,
        )
        self._r2_replay = Buffer(config)

    def set_replay_settings(self):
        logging.info(
            "\t Operating with %s environments, batch size %s and replay ratio %s",
            self.n_envs, self._batch_size, self._replay_ratio)
        self._num_updates_per_train_step = max(
            1, self._replay_ratio * self.n_envs // self._batch_size)
        self.update_period = max(
            1, self._batch_size // self._replay_ratio * self.n_envs)
        logging.info(
            "\t Calculated %s updates per update phase",
            self._num_updates_per_train_step,
        )
        logging.info(
            "\t Calculated update frequency of %s step%s",
            self.update_period,
            "s" if self.update_period > 1 else "",
        )
        logging.info(
            "\t Setting min_replay_history to %s from %s",
            self.min_replay_history / self.n_envs,
            self.min_replay_history,
        )
        self.min_replay_history = self.min_replay_history / self.n_envs
        self._batches_to_group = min(self._batches_to_group,
                                     self._num_updates_per_train_step)
        assert self._num_updates_per_train_step % self._batches_to_group == 0
        self._num_updates_per_train_step = int(
            max(1, self._num_updates_per_train_step / self._batches_to_group))

        # debug - start
        print(
            " self._num_updates_per_train_step: {}\n self._batches_to_group: {}"
            .format(self._num_updates_per_train_step, self._batches_to_group))
        #exit(0)
        # debug - end

        logging.info(
            "\t Running %s groups of %s batch%s per %s env step%s",
            self._num_updates_per_train_step,
            self._batches_to_group,
            "es" if self._batches_to_group > 1 else "",
            self.update_period,
            "s" if self.update_period > 1 else "",
        )

    def _replay_sampler_generator(self):
        types = self._replay.get_transition_elements()
        while True:
            self._rng, rng = jax.random.split(self._rng)

            samples = self._replay.sample_transition_batch(
                rng,
                batch_size=self._batch_size * self._batches_to_group,
                update_horizon=self.update_horizon_scheduler(
                    self.cycle_grad_steps),
                gamma=self.gamma_scheduler(self.cycle_grad_steps),
            )
            replay_elements = collections.OrderedDict()
            for element, element_type in zip(samples, types):
                replay_elements[element_type.name] = element
            yield replay_elements

    def sample_eval_batch(self, batch_size, subseq_len=1):
        self._rng, rng = jax.random.split(self._rng)
        samples = self._replay.sample_transition_batch(rng,
                                                       batch_size=batch_size,
                                                       subseq_len=subseq_len)
        types = self._replay.get_transition_elements()
        replay_elements = collections.OrderedDict()
        for element, element_type in zip(samples, types):
            replay_elements[element_type.name] = element
        # Add code for data augmentation.

        return replay_elements

    def initialize_prefetcher(self):
        self.prefetcher = prefetch_to_device(self._replay_sampler_generator(),
                                             2)

    def _sample_from_replay_buffer(self):
        self.replay_elements = next(self.prefetcher)

    def _torch_to_numpy(self, value):
        if hasattr(value, "detach"):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def _sample_r2_world_model_batch(self):
        data, index, initial = self._r2_replay.sample()
        return {
            "state": self._torch_to_numpy(data["state"]),
            "action": self._torch_to_numpy(data["action"]),
            "reward": self._torch_to_numpy(data["reward"]),
            "is_terminal": self._torch_to_numpy(data["is_terminal"]).astype(
                np.float32),
            "is_first": self._torch_to_numpy(data["is_first"]).astype(
                np.float32),
            "initial_stoch": self._torch_to_numpy(initial[0]),
            "initial_deter": self._torch_to_numpy(initial[1]),
            "index": index,
        }

    def _r2_replay_ready(self):
        min_count = max(int(self.min_replay_history),
                        self.r2_world_model_batch_length + 1)
        return self._r2_replay.count() > min_count

    def _update_r2_world_model_buffer(self, index, stoch, deter):
        stoch = np.asarray(jax.device_get(stoch), dtype=np.float32)
        deter = np.asarray(jax.device_get(deter), dtype=np.float32)
        stoch = stoch.reshape(-1, *stoch.shape[2:])
        deter = deter.reshape(-1, *deter.shape[2:])
        self._r2_replay.update(index, stoch, deter)

    def _observe_r2_world_model_state(self):
        self._rng, rng = jax.random.split(self._rng)
        stoch, deter = r2_world_model_observe(
            self.network_def,
            self.online_params,
            self.state,
            self._r2_prev_action,
            self._r2_stoch,
            self._r2_deter,
            self._r2_is_first,
            rng,
            self.dtype,
        )
        self._r2_stoch = np.asarray(jax.device_get(stoch), dtype=np.float32)
        self._r2_deter = np.asarray(jax.device_get(deter), dtype=np.float32)
        self._r2_is_first.fill(0.0)

    def _one_hot_actions(self, action):
        action = np.asarray(action, dtype=np.int32)
        return np.eye(self.num_actions, dtype=np.float32)[action]

    def _flush_r2_pending_transition(self, action):
        if self.eval_mode or self._r2_pending_transition is None:
            return
        action_onehot = self._one_hot_actions(action)
        r2_index = self._r2_replay.add_atari_transition(
            state=self._r2_pending_transition["state"],
            action=action_onehot,
            reward=self._r2_pending_transition["reward"],
            is_terminal=self._r2_pending_transition["is_terminal"],
            is_first=self._r2_pending_transition["is_first"],
            stoch=self._r2_stoch,
            deter=self._r2_deter,
        )
        self._r2_last_added_index = np.asarray(
            self._torch_to_numpy(r2_index), dtype=np.int64)
        self._r2_pending_transition = None

    def _set_r2_pending_transition(self, reward, terminal, is_first):
        if self.eval_mode:
            return
        self._r2_pending_transition = {
            "state": np.asarray(self.state, dtype=self.observation_dtype).copy(),
            "reward": np.asarray(reward, dtype=np.float32).reshape(-1, 1),
            "is_terminal": np.asarray(terminal, dtype=np.float32).reshape(-1, 1),
            "is_first": np.asarray(is_first, dtype=np.float32).reshape(-1, 1),
        }

    def reset_weights(self):
        self.cumulative_resets += 1
        interval = self.reset_every

        self.next_reset = int(interval) + self.training_steps
        if self.next_reset > self.no_resets_after + self.reset_offset:
            logging.info(
                "\t Not resetting at step %s, as need at least"
                " %s before %s to recover.", self.training_steps, interval,
                self.no_resets_after)
            return
        else:
            logging.info("\t Resetting weights at step %s.",
                         self.training_steps)

        self._rng, reset_rng = jax.random.split(self._rng, 2)

        #keys_to_copy = ("encoder", "transition_model")
        keys_to_copy = ("encoder", "transition_model",
                        "representation_projection", "_log_alpha",
                        "r2_world_model")
        (
            self.online_params,
            self.target_network_params,
            self.optimizer_state,
            self.random_params,
        ) = jit_reset(
            self.online_params,
            self.target_network_params,
            self.optimizer_state,
            self.network_def,
            self.optimizer,
            reset_rng,
            self.state_shape,
            self.spr_weight > 0,
            self._support,
            self.reset_target,
            self.shrink_perturb_keys,
            self.shrink_factor,
            self.perturb_factor,
            keys_to_copy,
        )

        self.cycle_grad_steps = 0

    def _training_step_update(self, step_index, offline=False):
        """Gradient update during every training step."""
        self.start = time.time()

        self._rng, train_rng = jax.random.split(self._rng)
        r2wm_batch = self._sample_r2_world_model_batch()
        (
            new_online_params,
            new_target_params,
            new_optimizer_state,
            new_r2wm_optimizer_state,
            aux_losses,
            r2wm_post_stoch,
            r2wm_post_deter,
        ) = self.train_fn(
            self.network_def,
            self.online_params,
            self.target_network_params,
            self.optimizer,
            self.optimizer_state,
            self.r2wm_optimizer_state,
            self._support,
            self.gamma_scheduler(self.cycle_grad_steps),
            train_rng,
            self._data_augmentation,
            self.dtype,
            self._batch_size,
            self.use_target_network,
            self.target_update_tau_scheduler(self.cycle_grad_steps),
            self.target_update_period,
            self.grad_steps,
            self.match_online_target_rngs,
            self.target_eval_mode,
            #self.ent_targ,
            self.x_ent_coef,
            r2wm_batch["state"],
            r2wm_batch["action"],
            r2wm_batch["reward"],
            r2wm_batch["is_terminal"],
            r2wm_batch["is_first"],
            r2wm_batch["initial_stoch"],
            r2wm_batch["initial_deter"],
            self.r2_world_model_learning_rate,
            self.r2_world_model_beta1,
            self.r2_world_model_beta2,
            self.r2_world_model_eps,
            self.r2_world_model_warmup,
            self.r2_world_model_agc,
            self.r2_world_model_pmin,
        )
        self._update_r2_world_model_buffer(r2wm_batch["index"],
                                           r2wm_post_stoch,
                                           r2wm_post_deter)
        self.grad_steps += self._batches_to_group
        self.cycle_grad_steps += self._batches_to_group

        self.target_network_params = new_target_params
        self.online_params = new_online_params
        self.optimizer_state = new_optimizer_state
        self.r2wm_optimizer_state = new_r2wm_optimizer_state

    def _store_transition(
        self,
        last_observation,
        action,
        reward,
        is_terminal,
        *args,
        episode_end=False,
    ):
        priority = np.full((last_observation.shape[0]),
                           self._replay.sum_tree.max_recorded_priority)

        if not self.eval_mode:
            args = args + (self._r2_last_added_index.copy(),)
            self._replay.add(
                last_observation,
                action,
                reward,
                is_terminal,
                *args,
                priority=priority,
                episode_end=episode_end,
            )

    def _train_step(self):
        # linearly decay target entropy - start
        def linearly_decaying_epsilon(decay_period, step, warmup_steps,
                                      epsilon):
            # Begin at 1. until warmup_steps steps have been taken; then
            # Linearly decay epsilon from 1. to epsilon in decay_period steps; and then
            # Use epsilon from there on.
            steps_left = decay_period + warmup_steps - step
            if False:
                bonus = (1.0 - epsilon) * steps_left / decay_period
                bonus = jnp.clip(bonus, 0., 1. - epsilon)
            # Begin at 0.5 until warmup_steps steps have been taken; then
            # Linearly decay epsilon from 0.5 to epsilon in decay_period steps; and then
            elif False:
                bonus = (0.5 - epsilon) * steps_left / decay_period
                bonus = jnp.clip(bonus, 0., 0.5 - epsilon)
            else:
                bonus = (1e-2 - epsilon) * steps_left / decay_period
                bonus = jnp.clip(bonus, 0., 1e-2 - epsilon)
            return epsilon + bonus

        ##frac = linearly_decaying_epsilon(1e5, self.training_steps, 0, 0.01)
        #frac = linearly_decaying_epsilon(self.explore_end_steps,
        #                                 self.training_steps, 0, 1e-3)
        #x = np.full((self.num_actions,),
        #            fill_value=frac / self.num_actions,
        #            dtype=np.float32)
        #x[0] += 1 - frac
        #self.ent_targ = jnp.asarray(scipy.stats.entropy(x))
        ##if random.uniform(0, 1) < 1e-3:
        #if False:
        #    logging.info("step: {}, frac: {}, ent_targ: {}".format(
        #        self.training_steps, frac, self.ent_targ))
        ##exit(0)
        # linearly decay target entropy - end

        self.x_ent_coef = linearly_decaying_epsilon(int(80e3),
                                                    self.training_steps, 0, .0)
        if random.uniform(0, 1) < 1e-3:
            logging.info("step: {}, x_ent_coef: {}".format(
                self.training_steps, self.x_ent_coef))

        if self._r2_replay_ready():
            if self.training_steps % self.update_period == 0:
                for i in range(self._num_updates_per_train_step):
                    self._training_step_update(i, offline=False)
        if self.reset_every > 0 and self.training_steps > self.next_reset:
            self.reset_weights()
        # debug - start
        #if random.uniform(0, 1) < 1e-3:
        if False:
            ent_coef = self.network_def.apply(
                self.online_params, method=self.network_def.entropy_scale)
            logging.info("ent_coef: {}".format(ent_coef))
            #logging.info("self.ent_targ: {}".format(self.ent_targ))
        # debug - end
        self.training_steps += 1

        # Cool down gpu
        #time.sleep(0.1)

    def _reset_state(self, n_envs):
        """Resets the agent state by filling it with zeros."""
        self.state = np.zeros(n_envs, *self.state_shape)

    def _record_observation(self, observation):
        """Records an observation and update state.

    Extracts a frame from the observation vector and overwrites the oldest
    frame in the state buffer.

    Args:
      observation: numpy array, an observation from the environment.
    """
        # Set current observation. We do the reshaping to handle environments
        # without frame stacking.
        observation = observation.squeeze(-1)
        if len(observation.shape) == len(self.observation_shape):
            self._observation = np.reshape(observation, self.observation_shape)
        else:
            self._observation = np.reshape(
                observation, (observation.shape[0], *self.observation_shape))
        # Swap out the oldest frame with the current frame.
        self.state = np.roll(self.state, -1, axis=-1)
        self.state[Ellipsis, -1] = self._observation

    def reset_all(self, new_obs):
        """Resets the agent state by filling it with zeros."""
        n_envs = new_obs.shape[0]
        self.state = np.zeros((n_envs, *self.state_shape))
        self._record_observation(new_obs)
        self._r2_stoch = np.zeros(
            (n_envs, self.r2_world_model_stoch,
             self.r2_world_model_discrete),
            dtype=np.float32)
        self._r2_deter = np.zeros((n_envs, self.r2_world_model_deter),
                                  dtype=np.float32)
        self._r2_prev_action = np.zeros((n_envs, self.num_actions),
                                        dtype=np.float32)
        self._r2_is_first = np.ones((n_envs, 1), dtype=np.float32)
        self._r2_last_added_index = np.full((n_envs, 2),
                                            -1,
                                            dtype=np.int64)
        self._set_r2_pending_transition(
            np.zeros((n_envs,), dtype=np.float32),
            np.zeros((n_envs,), dtype=np.float32),
            np.ones((n_envs,), dtype=np.float32),
        )

    def reset_one(self, env_id):
        self.state[env_id].fill(0)
        self._r2_stoch[env_id].fill(0.0)
        self._r2_deter[env_id].fill(0.0)
        self._r2_prev_action[env_id].fill(0.0)
        self._r2_is_first[env_id] = 1.0

    def delete_one(self, env_id):
        self.state = np.concatenate(
            [self.state[:env_id], self.state[env_id + 1:]], 0)
        self._r2_stoch = np.concatenate(
            [self._r2_stoch[:env_id], self._r2_stoch[env_id + 1:]], 0)
        self._r2_deter = np.concatenate(
            [self._r2_deter[:env_id], self._r2_deter[env_id + 1:]], 0)
        self._r2_prev_action = np.concatenate([
            self._r2_prev_action[:env_id],
            self._r2_prev_action[env_id + 1:]
        ], 0)
        self._r2_is_first = np.concatenate([
            self._r2_is_first[:env_id],
            self._r2_is_first[env_id + 1:]
        ], 0)
        self._r2_last_added_index = np.concatenate([
            self._r2_last_added_index[:env_id],
            self._r2_last_added_index[env_id + 1:]
        ], 0)

    def cache_train_state(self):
        self.training_state = (
            copy.deepcopy(self.state),
            copy.deepcopy(self._last_observation),
            copy.deepcopy(self._observation),
            copy.deepcopy(self._r2_stoch),
            copy.deepcopy(self._r2_deter),
            copy.deepcopy(self._r2_prev_action),
            copy.deepcopy(self._r2_is_first),
            copy.deepcopy(self._r2_pending_transition),
            copy.deepcopy(self._r2_last_added_index),
        )

    def restore_train_state(self):
        (self.state, self._last_observation, self._observation,
         self._r2_stoch, self._r2_deter, self._r2_prev_action,
         self._r2_is_first, self._r2_pending_transition,
         self._r2_last_added_index) = (
             self.training_state)

    def log_transition(self, observation, action, reward, terminal,
                       episode_end, raw_reward=None):
        self._last_observation = self._observation
        self._record_observation(observation)

        if not self.eval_mode:
            self._store_transition(
                self._last_observation,
                action,
                reward,
                terminal,
                episode_end=episode_end,
            )
            if raw_reward is None:
                raw_reward = reward
            is_first = np.logical_or(terminal, episode_end).astype(np.float32)
            self._set_r2_pending_transition(raw_reward, terminal, is_first)

    def select_action(
        self,
        select_params,
        eval_mode,
    ):
        if not eval_mode and self.training_steps < self.min_replay_history:
            self._rng, key = jax.random.split(self._rng)
            return jax.random.randint(
                key,
                (self._r2_stoch.shape[0],),
                0,
                self.num_actions,
            )
        self._rng, action, probs = select_action_from_r2(
            self.network_def,
            select_params,
            self._r2_stoch,
            self._r2_deter,
            self._rng,
            False,
            #eval_mode,
        )
        #print(probs.shape)
        #if not self.eval_mode:
        if not self.eval_mode:
            self.stats_ent = 0.99 * self.stats_ent + 0.01 * scipy.stats.entropy(
                probs[0])
            if random.uniform(0, 1) < 1e-3:
                logging.info('ema entropy: {}'.format(self.stats_ent))
        #exit(0)
        return action

    def step(self):
        """Records the most recent transition, returns the agent's next action, and trains if appropriate.
    """
        if not self.eval_mode:
            self._train_step()
        self._observe_r2_world_model_state()
        select_params = self.target_network_params
        #select_params = self.online_params
        ## time inference - start
        #self.eval_mode = True
        #import time
        #start_time = time.time()
        #for _ in range(int(100 * 32)):
        action = self.select_action(
            select_params,
            self.eval_mode,
        )
        self.action = np.asarray(action)
        if not self.eval_mode:
            self._flush_r2_pending_transition(self.action)
        self._r2_prev_action = self._one_hot_actions(self.action)
        #time_delta = time.time() - start_time
        #print('time_delta: {}'.format(time_delta))
        #exit(0)
        return self.action
