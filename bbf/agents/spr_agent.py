# coding=utf-8

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


def sigmoid_binary_cross_entropy(logits, labels):
    """Numerically stable sigmoid cross entropy."""
    return jnp.maximum(logits, 0) - logits * labels + jnp.log1p(
        jnp.exp(-jnp.abs(logits)))


def masked_mean(values, mask, eps=1e-6):
    mask = mask.astype(jnp.float32)
    return jnp.sum(values * mask) / (jnp.sum(mask) + eps)


def weighted_barlow_twins_loss(predictions,
                               targets,
                               mask,
                               lambd=5e-4,
                               eps=1e-5):
    """Barlow Twins loss over valid predicted/target latent pairs."""
    predictions = predictions.reshape((-1, predictions.shape[-1]))
    targets = targets.reshape((-1, targets.shape[-1]))
    weights = mask.reshape((-1, 1)).astype(jnp.float32)
    count = jnp.sum(weights) + eps

    pred_mean = jnp.sum(predictions * weights, axis=0, keepdims=True) / count
    targ_mean = jnp.sum(targets * weights, axis=0, keepdims=True) / count
    pred_centered = predictions - pred_mean
    targ_centered = targets - targ_mean
    pred_std = jnp.sqrt(
        jnp.sum(jnp.square(pred_centered) * weights, axis=0, keepdims=True) /
        count + eps)
    targ_std = jnp.sqrt(
        jnp.sum(jnp.square(targ_centered) * weights, axis=0, keepdims=True) /
        count + eps)
    pred_norm = pred_centered / pred_std
    targ_norm = targ_centered / targ_std

    corr = (pred_norm * weights).T @ targ_norm / count
    diag = jnp.diag(corr)
    eye = jnp.eye(corr.shape[0], dtype=corr.dtype)
    invariance = jnp.sum(jnp.square(diag - 1.0))
    redundancy = jnp.sum(jnp.square(corr * (1.0 - eye)))
    return invariance + lambd * redundancy


def lambda_return(rewards, continues, values, discount, lambd):
    """Computes lambda returns for action-aligned imagined transitions.

    Rewards and continuations at time t describe the transition caused by the
    action at t. Values contain both the pre-action values and one final
    post-action bootstrap. This explicit contract prevents a one-step shift.

    Args:
      rewards: Array with shape [B, H].
      continues: Array with shape [B, H].
      values: Array with shape [B, H + 1].
      discount: Scalar discount.
      lambd: Scalar lambda-return mixing coefficient.

    Returns:
      Array with shape [B, H].
    """
    if rewards.shape != continues.shape:
        raise ValueError("rewards and continues must have identical shapes")
    if (values.shape[:-1] != rewards.shape[:-1] or
            values.shape[-1] != rewards.shape[-1] + 1):
        raise ValueError("values must contain one more time step than rewards")

    next_values = values[:, 1:]
    inputs = rewards + continues * discount * (1.0 - lambd) * next_values
    discounts = continues * discount * lambd

    def scan_fn(carry, elems):
        inp, disc = elems
        ret = inp + disc * carry
        return ret, ret

    _, returns = jax.lax.scan(scan_fn,
                              values[:, -1],
                              (jnp.swapaxes(inputs, 0, 1),
                               jnp.swapaxes(discounts, 0, 1)),
                              reverse=True)
    return jnp.swapaxes(returns, 0, 1)


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
    input_was_frozen = isinstance(old_params, FrozenDict)
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
    return (FrozenDict(combined_params)
            if input_was_frozen else combined_params)


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


@functools.partial(
    jax.jit,
    static_argnames=("network_def", "eval_mode"),
)
def select_rssm_action(
    network_def,
    params,
    state,
    prev_rssm_state,
    prev_action,
    is_first,
    rng,
    eval_mode,
):
    """Updates the online posterior and samples from the shared actor."""
    rng, posterior_key, action_key = jax.random.split(rng, 3)
    state = spr_networks.process_inputs(
        state,
        rng=posterior_key,
        data_augmentation=False,
        dtype=jnp.float32,
    )
    post, _ = network_def.apply(
        params,
        state,
        prev_rssm_state,
        prev_action,
        is_first,
        eval_mode,
        rngs={"sample": posterior_key},
        method=network_def.rssm_observe_step,
    )
    feature = network_def.apply(
        params,
        post,
        method=network_def.rssm_feature,
    )
    logits = network_def.apply(
        params,
        feature,
        eval_mode,
        method=network_def.actor_from_rssm_feature,
    )
    if eval_mode:
        actions = jnp.argmax(logits, axis=-1)
    else:
        actions = jax.random.categorical(action_key, logits)
    return rng, actions, jax.nn.softmax(logits, axis=-1), post


train_static_argnames = [
    'network_def',
    'optimizer',
    'double_dqn',
    'distributional',
    'spr_weight',
    'data_augmentation',
    'dtype',
    'batch_size',
    'use_target_backups',
    'match_online_target_rngs',
    'target_eval_mode',
    'world_model_weight',
    'imag_horizon',
    'rssm_burnin',
    'target_action_selection',
]


def train(
    network_def,
    online_params,
    target_params,
    optimizer,
    optimizer_state,
    raw_states,
    actions,
    raw_next_states,
    td_returns,
    one_step_rewards,
    terminals,
    same_traj_mask,
    loss_weights,
    support,
    cumulative_gamma,
    double_dqn,
    distributional,
    rng,
    spr_weight,
    data_augmentation,
    dtype,
    batch_size,
    use_target_backups,
    target_update_tau,
    target_update_every,
    step,
    match_online_target_rngs,
    target_eval_mode,
    x_ent_coef,
    world_model_weight,
    reward_weight,
    continue_weight,
    barlow_weight,
    barlow_lambd,
    rssm_dyn_weight,
    rssm_rep_weight,
    rssm_burnin,
    imag_horizon,
    imag_actor_weight,
    imag_value_weight,
    imag_discount,
    imag_lambda,
    imag_entropy_weight,
    imag_warmup_steps,
    target_action_selection,
):
    """Trains a shared RSSM actor with real C51 and imagined value critics.

    Replay rows follow (s_t, a_t, r_{t+1}, done_{t+1}). C51 is updated only
    from replay data. The separate scalar value is updated only from imagined
    lambda returns, and imagined transitions never enter prioritized replay.
    """

    del double_dqn, distributional, match_online_target_rngs

    @functools.partial(jax.jit, donate_argnums=(0,))
    def train_one_batch(state, inputs):
        (online_params, target_params, optimizer_state, rng, step) = state
        (
            raw_states,
            actions,
            raw_next_states,
            td_returns,
            one_step_rewards,
            terminals,
            same_traj_mask,
            loss_weights,
            cumulative_gamma,
        ) = inputs

        same_mask = same_traj_mask.astype(jnp.float32)
        same_mask = same_mask.at[:, 0].set(1.0)
        batch_indices = jnp.arange(raw_states.shape[0])[:, None]
        last_valid = (
            jnp.sum(same_mask > 0.5, axis=1).astype(jnp.int32) - 1)
        last_valid = jnp.maximum(last_valid, 0)
        # Ground C51 and the real actor at both a zero-context posterior and
        # the longest valid recurrent posterior in each sampled sequence.
        control_indices = jnp.stack(
            [jnp.zeros_like(last_valid), last_valid], axis=1)
        td_rewards = td_returns[batch_indices, control_indices]
        td_terminals = terminals[batch_indices, control_indices]
        td_cumulative_gamma = cumulative_gamma[
            batch_indices, control_indices]
        control_next_states = raw_next_states[
            batch_indices, control_indices]

        (
            rng,
            state_aug_key,
            next_aug_key,
            target_action_key,
            loss_key,
        ) = jax.random.split(rng, 5)
        states = spr_networks.process_inputs(
            raw_states,
            rng=state_aug_key,
            data_augmentation=data_augmentation,
            dtype=dtype,
        )
        next_states = spr_networks.process_inputs(
            control_next_states,
            rng=next_aug_key,
            data_augmentation=data_augmentation,
            dtype=dtype,
        )
        current_state = states[:, 0]

        q_target_params = (
            target_params if use_target_backups else online_params)
        target_post, _ = network_def.apply(
            q_target_params,
            next_states,
            True,
            method=network_def.rssm_from_observation,
        )
        target_feature = network_def.apply(
            q_target_params,
            target_post,
            method=network_def.rssm_feature,
        )
        target_feature = jax.lax.stop_gradient(target_feature)
        target_q = network_def.apply(
            q_target_params,
            target_feature,
            support,
            target_eval_mode,
            method=network_def.q_from_rssm_feature,
        )

        actor_target_params = (
            target_params if target_action_selection else online_params)
        target_policy_logits = network_def.apply(
            actor_target_params,
            target_feature,
            target_eval_mode,
            method=network_def.actor_from_rssm_feature,
        )
        if target_eval_mode:
            next_actions = jnp.argmax(target_policy_logits, axis=-1)
        else:
            next_actions = jax.random.categorical(
                target_action_key, target_policy_logits)
        next_probabilities = jnp.take_along_axis(
            target_q.probabilities,
            next_actions[..., None, None],
            axis=2,
        )[:, :, 0]
        gamma_with_terminal = (
            td_cumulative_gamma *
            (1.0 - td_terminals.astype(jnp.float32)))
        target_supports = (
            td_rewards[..., None] +
            gamma_with_terminal[..., None] * support[None, None])
        flat_target_supports = target_supports.reshape(
            (-1, support.shape[0]))
        flat_next_probabilities = next_probabilities.reshape(
            (-1, support.shape[0]))
        target = jax.vmap(
            project_distribution,
            in_axes=(0, 0, None),
        )(flat_target_supports, flat_next_probabilities, support)
        target = target.reshape(target_supports.shape)
        target = jax.lax.stop_gradient(target)

        if spr_weight > 0.0:
            future_states = states[:, 1:]

            def encode_project_target(observation):
                return network_def.apply(
                    target_params,
                    observation,
                    True,
                    method=network_def.encode_project,
                )

            spr_targets = jax.vmap(
                jax.vmap(encode_project_target, in_axes=0),
                in_axes=0,
            )(future_states)
            spr_targets = jax.lax.stop_gradient(spr_targets)
        else:
            spr_targets = None

        def loss_fn(params, key):
            (
                rssm_key,
                real_actor_key,
                imag_action_key,
                imag_model_key,
            ) = jax.random.split(key, 4)

            embeds = network_def.apply(
                params,
                states,
                False,
                method=network_def.encode_rssm,
            )
            action_onehots = jax.nn.one_hot(
                actions,
                network_def.num_actions,
                dtype=embeds.dtype,
            )
            prev_actions = jnp.concatenate(
                [jnp.zeros_like(action_onehots[:, :1]),
                 action_onehots[:, :-1]],
                axis=1,
            )
            is_first = jnp.zeros_like(same_mask, dtype=jnp.bool_)
            is_first = is_first.at[:, 0].set(True)
            post, prior = network_def.apply(
                params,
                embeds,
                prev_actions,
                is_first,
                False,
                rngs={"sample": rssm_key},
                method=network_def.rssm_observe,
            )
            post_features = network_def.apply(
                params,
                post,
                method=network_def.rssm_feature,
            )
            prior_features = network_def.apply(
                params,
                prior,
                method=network_def.rssm_feature,
            )

            control_feature = jax.lax.stop_gradient(
                post_features[batch_indices, control_indices])
            control_actions = actions[batch_indices, control_indices]
            q_output = network_def.apply(
                params,
                control_feature,
                support,
                False,
                method=network_def.q_from_rssm_feature,
            )
            chosen_logits = jnp.take_along_axis(
                q_output.logits,
                control_actions[..., None, None],
                axis=2,
            )[:, :, 0]
            dqn_loss_per_position = -jnp.sum(
                target * jax.nn.log_softmax(chosen_logits, axis=-1),
                axis=-1,
            )
            dqn_loss = jnp.mean(dqn_loss_per_position, axis=1)
            target_entropy_term = jnp.sum(
                target * jnp.log(jnp.maximum(target, 1e-8)),
                axis=-1,
            )
            td_error = jnp.mean(
                dqn_loss_per_position + target_entropy_term, axis=1)

            if spr_weight > 0.0:
                def spr_result(observation, rollout_actions):
                    return network_def.apply(
                        params,
                        observation,
                        rollout_actions,
                        False,
                        method=network_def.spr_from_observation,
                    )

                spr_predictions = jax.vmap(
                    spr_result,
                    in_axes=(0, 0),
                )(current_state, actions[:, :-1])
                branch_shape = (
                    *spr_predictions.shape[:-1],
                    2,
                    network_def.hidden_dim,
                )
                spr_predictions = spr_predictions.reshape(branch_shape)
                normalized_targets = spr_targets.reshape(branch_shape)
                spr_predictions = spr_predictions / jnp.maximum(
                    jnp.linalg.norm(
                        spr_predictions, axis=-1, keepdims=True),
                    1e-8,
                )
                normalized_targets = normalized_targets / jnp.maximum(
                    jnp.linalg.norm(
                        normalized_targets, axis=-1, keepdims=True),
                    1e-8,
                )
                spr_per_step = 0.5 * jnp.sum(
                    jnp.square(spr_predictions - normalized_targets),
                    axis=(-1, -2),
                )
                spr_valid = same_mask[:, 1:]
                spr_loss = (
                    jnp.sum(spr_per_step * spr_valid, axis=1) /
                    jnp.maximum(jnp.sum(spr_valid, axis=1), 1.0))
            else:
                spr_loss = jnp.zeros_like(dqn_loss)

            real_policy_logits = network_def.apply(
                params,
                control_feature,
                False,
                method=network_def.actor_from_rssm_feature,
            )
            real_log_probs = jax.nn.log_softmax(
                real_policy_logits, axis=-1)
            real_probs = jax.nn.softmax(real_policy_logits, axis=-1)
            sampled_actions = jax.random.categorical(
                real_actor_key, real_policy_logits)
            sampled_q = jnp.take_along_axis(
                q_output.q_values,
                sampled_actions[..., None],
                axis=-1,
            )[:, :, 0]
            q_baseline = jnp.sum(
                real_probs * q_output.q_values, axis=-1)
            real_advantage = jax.lax.stop_gradient(
                sampled_q - q_baseline)
            selected_log_prob = jnp.take_along_axis(
                real_log_probs,
                sampled_actions[..., None],
                axis=-1,
            )[:, :, 0]
            real_entropy = -jnp.sum(
                real_probs * real_log_probs, axis=-1)
            real_actor_per_position = (
                -real_advantage * selected_log_prob -
                x_ent_coef * real_entropy)
            real_actor_per_sample = jnp.mean(
                real_actor_per_position, axis=1)

            real_loss = jnp.mean(
                loss_weights * (dqn_loss + spr_weight * spr_loss))
            real_actor_loss = jnp.mean(
                loss_weights * real_actor_per_sample)

            _, dyn_kl, rep_kl = network_def.apply(
                params,
                post,
                prior,
                method=network_def.rssm_kl_loss,
            )
            dyn_loss = masked_mean(dyn_kl, same_mask)
            rep_loss = masked_mean(rep_kl, same_mask)

            barlow_predictions = network_def.apply(
                params,
                post_features,
                False,
                method=network_def.rssm_barlow_prediction,
            )
            barlow_loss = weighted_barlow_twins_loss(
                barlow_predictions,
                jax.lax.stop_gradient(embeds),
                same_mask,
                lambd=barlow_lambd,
            )

            if states.shape[1] > 1:
                transition_mask = same_mask[:, :-1]
                # A full-game boundary may replace the terminal observation
                # with the next episode's reset frame. Use the action-aligned
                # prior for terminal transitions and the posterior otherwise.
                transition_features = jnp.where(
                    (same_mask[:, 1:] > 0.5)[..., None],
                    post_features[:, 1:],
                    prior_features[:, 1:],
                )
                reward_predictions = network_def.apply(
                    params,
                    transition_features,
                    method=network_def.reward_from_feature,
                )
                continue_logits = network_def.apply(
                    params,
                    transition_features,
                    method=network_def.continue_from_feature,
                )
                reward_loss = masked_mean(
                    jnp.square(
                        reward_predictions - one_step_rewards[:, :-1]),
                    transition_mask,
                )
                continue_loss = masked_mean(
                    sigmoid_binary_cross_entropy(
                        continue_logits, same_mask[:, 1:]),
                    transition_mask,
                )
            else:
                reward_loss = jnp.asarray(0.0, dtype=embeds.dtype)
                continue_loss = jnp.asarray(0.0, dtype=embeds.dtype)

            model_loss = (
                rssm_dyn_weight * dyn_loss +
                rssm_rep_weight * rep_loss +
                reward_weight * reward_loss +
                continue_weight * continue_loss +
                barlow_weight * barlow_loss)

            imag_actor_loss = jnp.asarray(0.0, dtype=embeds.dtype)
            imag_value_loss = jnp.asarray(0.0, dtype=embeds.dtype)
            imag_return_scale = jnp.asarray(1.0, dtype=embeds.dtype)
            imag_weight_mean = jnp.asarray(0.0, dtype=embeds.dtype)
            imagination_gate = jnp.asarray(
                step >= imag_warmup_steps, dtype=embeds.dtype)

            if imag_horizon > 0:
                flat_start = jax.tree_util.tree_map(
                    lambda value: value.reshape(
                        (-1,) + value.shape[2:]),
                    post,
                )
                flat_start = jax.tree_util.tree_map(
                    jax.lax.stop_gradient, flat_start)
                num_starts = flat_start.deter.shape[0]
                action_keys = jax.random.split(
                    imag_action_key, num_starts)
                model_keys = jax.random.split(
                    imag_model_key, num_starts)

                def imagine_one(start_state, action_key, model_key):
                    return network_def.apply(
                        params,
                        start_state,
                        imag_horizon,
                        False,
                        rngs={
                            "action_sample": action_key,
                            "sample": model_key,
                        },
                        method=network_def.imagine_from_rssm,
                    )

                imagined = jax.vmap(imagine_one)(
                    flat_start, action_keys, model_keys)
                imag_features = jax.lax.stop_gradient(
                    imagined["features"])
                slow_values = network_def.apply(
                    target_params,
                    imag_features,
                    method=network_def.value_from_feature,
                )
                slow_values = jax.lax.stop_gradient(slow_values)
                imagined_rewards = jax.lax.stop_gradient(
                    imagined["rewards"])
                imagined_continues = jax.lax.stop_gradient(
                    imagined["continues"])
                imag_returns = lambda_return(
                    imagined_rewards,
                    imagined_continues,
                    slow_values,
                    imag_discount,
                    imag_lambda,
                )
                imag_returns = jax.lax.stop_gradient(imag_returns)

                start_mask = same_mask
                if rssm_burnin > 0:
                    burnin = min(rssm_burnin, start_mask.shape[1])
                    start_mask = start_mask.at[:, :burnin].set(0.0)
                start_mask = start_mask.reshape((-1, 1))
                prefix = jnp.concatenate(
                    [
                        jnp.ones_like(imagined_continues[:, :1]),
                        (imagined_continues[:, :-1] * imag_discount),
                    ],
                    axis=1,
                )
                imag_weights = (
                    jnp.cumprod(prefix, axis=1) * start_mask)
                imag_weights = jax.lax.stop_gradient(imag_weights)
                weight_sum = jnp.sum(imag_weights) + 1e-6
                return_mean = (
                    jnp.sum(imag_weights * imag_returns) / weight_sum)
                return_variance = (
                    jnp.sum(
                        imag_weights *
                        jnp.square(imag_returns - return_mean)) /
                    weight_sum)
                imag_return_scale = jax.lax.stop_gradient(
                    jnp.maximum(1.0, jnp.sqrt(return_variance + 1e-8)))

                advantage = jax.lax.stop_gradient(
                    (imag_returns - slow_values[:, :-1]) /
                    imag_return_scale)
                actor_objective = (
                    imagined["log_probs"] * advantage +
                    imag_entropy_weight * imagined["entropies"])
                imag_actor_loss = (
                    -jnp.sum(imag_weights * actor_objective) /
                    weight_sum)

                online_values = imagined["values"][:, :-1]
                slow_consistency = slow_values[:, :-1]
                value_error = 0.5 * (
                    jnp.square(online_values - imag_returns) +
                    jnp.square(online_values - slow_consistency))
                imag_value_loss = (
                    jnp.sum(imag_weights * value_error) /
                    weight_sum)
                imag_weight_mean = jnp.mean(imag_weights)
                imag_actor_loss *= imagination_gate
                imag_value_loss *= imagination_gate

            total_loss = (
                real_loss +
                real_actor_loss +
                world_model_weight * model_loss +
                imag_actor_weight * imag_actor_loss +
                imag_value_weight * imag_value_loss)
            aux_losses = {
                "TotalLoss": total_loss,
                "DQNLoss": dqn_loss,
                "TD Error": jnp.mean(td_error),
                "SPRLoss": jnp.mean(spr_loss),
                "RealActorLoss": real_actor_loss,
                "ControlDepth": jnp.mean(last_valid.astype(jnp.float32)),
                "WorldModelLoss": model_loss,
                "RSSMDynamicsLoss": dyn_loss,
                "RSSMRepresentationLoss": rep_loss,
                "RewardLoss": reward_loss,
                "ContinueLoss": continue_loss,
                "BarlowLoss": barlow_loss,
                "ImagActorLoss": imag_actor_loss,
                "ImagValueLoss": imag_value_loss,
                "ImagReturnScale": imag_return_scale,
                "ImagWeight": imag_weight_mean,
                "ImagEnabled": imagination_gate,
                "ent": jnp.mean(real_entropy),
            }
            return total_loss, aux_losses

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (_, aux_losses), grad = grad_fn(
            online_params, loss_key)
        updates, new_optimizer_state = optimizer.update(
            grad, optimizer_state, params=online_params)
        new_online_params = optax.apply_updates(
            online_params, updates)

        target_update_step = functools.partial(
            interpolate_weights,
            keys=None,
            old_weight=1.0 - target_update_tau,
            new_weight=target_update_tau,
        )
        new_target_params = jax.lax.cond(
            step % target_update_every == 0,
            target_update_step,
            lambda old, new: old,
            target_params,
            new_online_params,
        )

        new_state = (
            new_online_params,
            new_target_params,
            new_optimizer_state,
            rng,
            step + 1,
        )
        return new_state, aux_losses

    init_state = (
        online_params,
        target_params,
        optimizer_state,
        rng,
        step,
    )
    if raw_states.shape[0] % batch_size:
        raise ValueError("Grouped replay batch must divide by batch_size")
    num_batches = raw_states.shape[0] // batch_size

    def group(value):
        return value.reshape(
            num_batches, batch_size, *value.shape[1:])

    inputs = (
        group(raw_states),
        group(actions),
        group(raw_next_states),
        group(td_returns),
        group(one_step_rewards),
        group(terminals),
        group(same_traj_mask),
        group(loss_weights),
        group(cumulative_gamma),
    )
    (
        (
            online_params,
            target_params,
            optimizer_state,
            _,
            _,
        ),
        aux_losses,
    ) = jax.lax.scan(train_one_batch, init_state, inputs)

    return (
        online_params,
        target_params,
        optimizer_state,
        {key: jnp.reshape(value, (-1,))
         for key, value in aux_losses.items()},
    )


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
        world_model_weight=0.0,
        reward_weight=1.0,
        continue_weight=1.0,
        barlow_weight=0.05,
        barlow_lambd=5e-4,
        rssm_dyn_weight=1.0,
        rssm_rep_weight=0.1,
        rssm_burnin=1,
        imag_horizon=0,
        imag_actor_weight=0.0,
        imag_value_weight=0.0,
        imag_discount=None,
        imag_lambda=0.95,
        imag_entropy_weight=3e-4,
        imag_warmup_steps=10_000,
        half_precision=False,
        seed=None,
        log_every=None,
        explore_end_steps=None,
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
        self.log_every = None if log_every is None else int(log_every)
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

        self.target_action_selection = bool(target_action_selection)
        self.use_target_network = bool(use_target_network)
        self.match_online_target_rngs = bool(match_online_target_rngs)
        self.target_eval_mode = bool(target_eval_mode)
        self.world_model_weight = float(world_model_weight)
        self.reward_weight = float(reward_weight)
        self.continue_weight = float(continue_weight)
        self.barlow_weight = float(barlow_weight)
        self.barlow_lambd = float(barlow_lambd)
        self.rssm_dyn_weight = float(rssm_dyn_weight)
        self.rssm_rep_weight = float(rssm_rep_weight)
        self.rssm_burnin = int(rssm_burnin)
        self.imag_horizon = int(imag_horizon)
        self.imag_actor_weight = float(imag_actor_weight)
        self.imag_value_weight = float(imag_value_weight)
        self.imag_discount = None if imag_discount is None else float(
            imag_discount)
        self.imag_lambda = float(imag_lambda)
        self.imag_entropy_weight = float(imag_entropy_weight)
        self.imag_warmup_steps = int(imag_warmup_steps)
        if self.rssm_burnin < 0 or self.imag_warmup_steps < 0:
            raise ValueError(
                "RSSM burn-in and imagination warm-up must be nonnegative")
        logging.info(
            "\t RSSM burn-in: %d, imagination horizon: %d, warm-up: %d",
            self.rssm_burnin,
            self.imag_horizon,
            self.imag_warmup_steps,
        )

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
            # pylint: disable=g-long-lambda
            self.update_horizon_scheduler = lambda x: int(
                np.round(n_schedule(x) * self.max_update_horizon))

        self.max_target_update_tau = target_update_tau
        self.target_update_tau_scheduler = lambda x: self.target_update_tau

        logging.info("\t Found following local devices: %s",
                     str(jax.local_devices()))

        self.dtype = jnp.float32
        self.dtype_str = "float32"

        logging.info("\t Running with dtype %s", str(self.dtype))

        super().__init__(
            num_actions=num_actions,
            network=functools.partial(
                network,
                num_atoms=self._num_atoms,
                noisy=False,
                distributional=self._distributional,
                dtype=self.dtype,
            ),
            target_update_period=self.target_update_period,
            update_horizon=self.max_update_horizon,
            seed=seed,
        )
        if self.imag_discount is None:
            self.imag_discount = self.gamma

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

        self.greedy_action = False
        self.stats_ent = 0
        self.explore_end_steps = explore_end_steps
        logging.info("\t exploration schedule end: %s", explore_end_steps)

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
                do_rollout=(self.spr_weight > 0 or
                            self.world_model_weight > 0 or
                            self.imag_horizon > 0),
                support=self._support,
            ))

        optimizer = create_scaling_optimizer(learning_rate=self.learning_rate,)
        encoder_optimizer = create_scaling_optimizer(
            learning_rate=self.encoder_learning_rate,)
        policy_optim = create_scaling_optimizer(learning_rate=1e-4,)

        encoder_keys = {
            "encoder", "transition_model", "rssm_embed_projection",
            "rssm", "rssm_projector"
        }
        encoder_mask = FrozenDict({
            "params": {
                k: k in encoder_keys for k in self.online_params["params"]
            }
        })

        head_keys = {
            "projection", "q_projection", "head", "predictor",
            "reward_head", "continue_head", "value_head"
        }
        head_mask = FrozenDict({
            "params": {k: k in head_keys for k in self.online_params["params"]}
        })

        policy_keys = {
            "policy_projection", "predict_policy", "actor_projection", "policy"
        }
        policy_mask = FrozenDict({
            "params": {
                key: key in policy_keys
                for key in self.online_params["params"]
            }
        })

        alpha_optim = optax.sgd(learning_rate=-1e-3)
        alpha_key = {"_log_alpha"}
        alpha_mask = FrozenDict({
            "params": {k: k in alpha_key for k in self.online_params["params"]}
        })
        self.optimizer = optax.chain(
            optax.masked(encoder_optimizer, encoder_mask),
            optax.masked(optimizer, head_mask),
            optax.masked(policy_optim, policy_mask),
            optax.masked(alpha_optim, alpha_mask),
        )

        self.optimizer_state = self.optimizer.init(self.online_params)
        self.target_network_params = copy.deepcopy(self.online_params)
        self.random_params = copy.deepcopy(self.online_params)

    def _build_replay_buffer(self):
        replay_cls = (subsequence_replay_buffer.
                      PrioritizedJaxSubsequenceParallelEnvReplayBuffer)
        prioritized_buffer = replay_cls(
            observation_shape=self.observation_shape,
            stack_size=self.stack_size,
            update_horizon=self.max_update_horizon,
            gamma=self.gamma,
            subseq_len=self._jumps + 1,
            batch_size=self._batch_size,
            observation_dtype=self.observation_dtype,
        )

        self.n_envs = prioritized_buffer._n_envs  # pylint: disable=protected-access
        self.start = time.time()
        return prioritized_buffer

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
        types = self._replay.get_transition_elements(
            batch_size=batch_size, subseq_len=subseq_len)
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

        keys_to_copy = ("encoder", "transition_model",
                        "rssm_embed_projection",
                        "rssm", "rssm_projector", "reward_head",
                        "continue_head", "value_head", "_log_alpha")
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
            (self.spr_weight > 0 or self.world_model_weight > 0 or
             self.imag_horizon > 0),
            self._support,
            self.reset_target,
            self.shrink_perturb_keys,
            self.shrink_factor,
            self.perturb_factor,
            keys_to_copy,
        )

        if hasattr(self, "rssm_state"):
            n_envs = self.state.shape[0]
            self.rssm_state = self._initial_rssm_state(n_envs)
            self.rssm_prev_action = np.zeros(
                (n_envs, self.num_actions), dtype=np.float32)
            self.rssm_is_first = np.ones((n_envs,), dtype=np.bool_)

        self.cycle_grad_steps = 0

    def _training_step_update(self, step_index, offline=False):
        """Gradient update during every training step."""
        self.start = time.time()

        if not hasattr(self, "replay_elements"):
            self._sample_from_replay_buffer()

        # The original prioritized experience replay uses a linear exponent
        # schedule 0.4 -> 1.0. Comparing the schedule to a fixed exponent of
        # 0.5 on 5 games (Asterix, Pong, Q*Bert, Seaquest, Space Invaders)
        # suggested a fixed exponent actually performs better, except on Pong.
        probs = self.replay_elements["sampling_probabilities"]
        # Weight the loss by the inverse priorities.
        loss_weights = 1.0 / np.sqrt(probs + 1e-10)
        loss_weights /= np.max(loss_weights)
        indices = self.replay_elements["indices"]

        if False:
            # debug - start
            print(' self.replay_elements.keys():\n {}'.format(
                self.replay_elements.keys()))
            print(' self._jumps + 1: {}'.format(self._jumps + 1))
            if False:
                print(
                    ' self.update_horizon_scheduler(self.cycle_grad_steps): {}'.
                    format(self.update_horizon_scheduler(
                        self.cycle_grad_steps)))
            for k, v in self.replay_elements.items():
                print(' {}: {}'.format(k, v.shape))

            exit(0)
            # debug - end

        self._rng, train_rng = jax.random.split(self._rng)
        (
            new_online_params,
            new_target_params,
            new_optimizer_state,
            aux_losses,
        ) = self.train_fn(
            self.network_def,
            self.online_params,
            self.target_network_params,
            self.optimizer,
            self.optimizer_state,
            self.replay_elements["state"],
            self.replay_elements["action"],
            self.replay_elements["next_state"],
            self.replay_elements["return"],
            self.replay_elements["reward"],
            self.replay_elements["terminal"],
            self.replay_elements["same_trajectory"],
            loss_weights,
            self._support,
            self.replay_elements["discount"],
            self._double_dqn,
            self._distributional,
            train_rng,
            self.spr_weight,
            self._data_augmentation,
            self.dtype,
            self._batch_size,
            self.use_target_network,
            self.target_update_tau_scheduler(self.cycle_grad_steps),
            self.target_update_period,
            self.grad_steps,
            self.match_online_target_rngs,
            self.target_eval_mode,
            self.x_ent_coef,
            self.world_model_weight,
            self.reward_weight,
            self.continue_weight,
            self.barlow_weight,
            self.barlow_lambd,
            self.rssm_dyn_weight,
            self.rssm_rep_weight,
            self.rssm_burnin,
            self.imag_horizon,
            self.imag_actor_weight,
            self.imag_value_weight,
            self.imag_discount,
            self.imag_lambda,
            self.imag_entropy_weight,
            self.imag_warmup_steps,
            self.target_action_selection,
        )
        self.grad_steps += self._batches_to_group
        self.cycle_grad_steps += self._batches_to_group

        # Sample asynchronously while we wait for training
        self._sample_from_replay_buffer()
        # Rainbow and prioritized replay are parametrized by an exponent
        # alpha, but in both cases it is set to 0.5 - for simplicity's sake we
        # leave it as is here, using the more direct sqrt(). Taking the square
        # root "makes sense", as we are dealing with a squared loss.  Add a
        # small nonzero value to the loss to avoid 0 priority items. While
        # technically this may be okay, setting all items to 0 priority will
        # cause troubles, and also result in 1.0 / 0.0 = NaN correction terms.
        indices = np.reshape(np.asarray(indices), (-1,))
        dqn_loss = np.reshape(np.asarray(aux_losses["DQNLoss"]), (-1))

        # debug - start
        #if random.uniform(0, 1) < 1e-3:
        if False:
            logging.info("ent: {}".format(aux_losses["ent"]))
        # debug - end

        priorities = np.sqrt(dqn_loss + 1e-10)
        self._replay.set_priority(indices, priorities)
        if (self.log_every and
                self.grad_steps % self.log_every < self._batches_to_group):
            metrics = {
                key: float(np.mean(np.asarray(value)))
                for key, value in aux_losses.items()
            }
            logging.info("train metrics: %s", metrics)

        self.target_network_params = new_target_params
        self.online_params = new_online_params
        self.optimizer_state = new_optimizer_state

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

        if self._replay.add_count == self.min_replay_history:
            self.initialize_prefetcher()

        if self._replay.add_count > self.min_replay_history:
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

    def _initial_rssm_state(self, n_envs):
        state = self.network_def.apply(
            self.online_params,
            n_envs,
            method=self.network_def.rssm_initial,
        )
        return jax.tree_util.tree_map(np.asarray, state)

    def reset_all(self, new_obs):
        """Resets frame stacks and recurrent latent state for all envs."""
        n_envs = new_obs.shape[0]
        self.state = np.zeros((n_envs, *self.state_shape))
        self.rssm_state = self._initial_rssm_state(n_envs)
        self.rssm_prev_action = np.zeros(
            (n_envs, self.num_actions), dtype=np.float32)
        self.rssm_is_first = np.ones((n_envs,), dtype=np.bool_)
        self._record_observation(new_obs)

    def reset_one(self, env_id):
        self.state[env_id].fill(0)
        fresh = self._initial_rssm_state(1)

        def replace_one(current, initial):
            current = np.array(current, copy=True)
            current[env_id] = initial[0]
            return current

        self.rssm_state = jax.tree_util.tree_map(
            replace_one, self.rssm_state, fresh)
        self.rssm_prev_action[env_id].fill(0)
        self.rssm_is_first[env_id] = True

    def delete_one(self, env_id):
        self.state = np.concatenate(
            [self.state[:env_id], self.state[env_id + 1:]], axis=0)
        self.rssm_state = jax.tree_util.tree_map(
            lambda value: np.concatenate(
                [value[:env_id], value[env_id + 1:]], axis=0),
            self.rssm_state,
        )
        self.rssm_prev_action = np.concatenate(
            [self.rssm_prev_action[:env_id],
             self.rssm_prev_action[env_id + 1:]],
            axis=0,
        )
        self.rssm_is_first = np.concatenate(
            [self.rssm_is_first[:env_id],
             self.rssm_is_first[env_id + 1:]],
            axis=0,
        )

    def cache_train_state(self):
        self.training_state = (
            copy.deepcopy(self.state),
            copy.deepcopy(self._last_observation),
            copy.deepcopy(self._observation),
            copy.deepcopy(self.rssm_state),
            copy.deepcopy(self.rssm_prev_action),
            copy.deepcopy(self.rssm_is_first),
        )

    def restore_train_state(self):
        (
            self.state,
            self._last_observation,
            self._observation,
            self.rssm_state,
            self.rssm_prev_action,
            self.rssm_is_first,
        ) = self.training_state

    def log_transition(self, observation, action, reward, terminal,
                       episode_end):
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

    def select_action(
        self,
        state,
        select_params,
        eval_mode,
    ):
        self._rng, policy_action, probs, posterior = select_rssm_action(
            self.network_def,
            select_params,
            state,
            self.rssm_state,
            self.rssm_prev_action,
            self.rssm_is_first,
            self._rng,
            eval_mode,
        )
        self.rssm_state = jax.tree_util.tree_map(
            np.asarray, posterior)

        in_random_warmup = (
            not eval_mode and
            self.training_steps < self.min_replay_history)
        if in_random_warmup:
            self._rng, random_key = jax.random.split(self._rng)
            action = jax.random.randint(
                random_key,
                (state.shape[0],),
                0,
                self.num_actions,
            )
        else:
            action = policy_action

        action_array = np.asarray(action)
        self.rssm_prev_action = np.array(
            jax.nn.one_hot(
                action_array,
                self.num_actions,
                dtype=jnp.float32,
            ),
            copy=True)
        self.rssm_is_first = np.zeros(
            (state.shape[0],), dtype=np.bool_)

        if not eval_mode and not in_random_warmup:
            self.stats_ent = (
                0.99 * self.stats_ent +
                0.01 * scipy.stats.entropy(np.asarray(probs[0])))
            if random.uniform(0, 1) < 1e-3:
                logging.info("ema entropy: %s", self.stats_ent)
        return action

    def step(self):
        """Trains if needed, advances the posterior, and chooses an action."""
        if not self.eval_mode:
            self._train_step()
        action = self.select_action(
            self.state,
            self.online_params,
            self.eval_mode,
        )
        self.action = np.asarray(action)
        return self.action
