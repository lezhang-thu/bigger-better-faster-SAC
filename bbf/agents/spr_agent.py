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
    """Compute lambda returns for imagined trajectories.

  Args:
    rewards: [B, H + 1] predicted rewards.
    continues: [B, H + 1] predicted continuation probabilities.
    values: [B, H + 1] predicted values.
  Returns:
    [B, H] lambda returns.
  """
    next_values = values[:, 1:]
    inputs = rewards[:, :-1] + continues[:, :-1] * discount * (
        1.0 - lambd) * next_values
    discounts = continues[:, :-1] * discount * lambd

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


@functools.partial(jax.jit, static_argnames=[
    "network_def",
    "eval_mode",
])
def select_action(
    network_def,
    params,
    state,
    rng,
    num_actions,
    eval_mode,
):
    rng, key = jax.random.split(rng)
    state = spr_networks.process_inputs(state,
                                        rng=key,
                                        data_augmentation=False,
                                        dtype=jnp.float32)

    #epsilon = jnp.where(eval_mode, 1e-3, 0)

    def logits_w_samples(state, action_sample_key):
        return network_def.apply(
            params,
            state,
            rngs={"action_sample": action_sample_key},
            method=network_def.get_policy,
        )

    rng, key = jax.random.split(key)
    key = jax.random.split(key, state.shape[0])
    logits, samples = jax.vmap(logits_w_samples, in_axes=0,
                               axis_name="batch")(state, key)
    new_actions = jnp.where(eval_mode, jnp.argmax(logits, axis=-1), samples)
    return rng, new_actions, jax.nn.softmax(logits)

    #best_actions = jnp.argmax(logits, axis=-1)

    #rng, key0, key1 = jax.random.split(rng, num=3)
    #p = jax.random.uniform(key0, shape=(state.shape[0],))
    #new_actions = jnp.where(
    #    p < epsilon,
    #    jax.random.randint(
    #        key1,
    #        (state.shape[0],),
    #        0,
    #        num_actions,
    #    ),
    #    best_actions,
    #)
    ##return rng, new_actions, jax.nn.softmax(logits)
    #return rng, samples, jax.nn.softmax(logits)


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
    'r2wm_enabled',
    'r2_imag_horizon',
    'r2_imag_start_count',
    'r2_imag_return_norm',
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


def r2_lambda_return(last, term, reward, boot, disc, lamb):
    """Port of r2dreamer Dreamer._lambda_return (its `value` arg is unused).

    All inputs (B, T); returns (B, T-1). ret[t] targets step t, uses
    reward[t+1], cuts the lambda recursion at `last` boundaries and stops
    bootstrapping through `term`, bootstrapping per-step from `boot`.
    """
    live = (1.0 - term)[:, 1:] * disc
    cont = (1.0 - last)[:, 1:] * lamb
    interm = reward[:, 1:] + (1.0 - cont) * live * boot[:, 1:]

    def scan_fn(carry, elems):
        interm_t, live_t, cont_t = elems
        ret = interm_t + live_t * cont_t * carry
        return ret, ret

    _, rets = jax.lax.scan(
        scan_fn,
        boot[:, -1],
        (
            jnp.moveaxis(interm, 1, 0),
            jnp.moveaxis(live, 1, 0),
            jnp.moveaxis(cont, 1, 0),
        ),
        reverse=True,
    )
    return jnp.moveaxis(rets, 0, 1)


R2WM_METRIC_KEYS = (
    "R2WMLoss",
    "R2WMDynLoss",
    "R2WMRepLoss",
    "R2WMBarlowLoss",
    "R2WMRewardLoss",
    "R2WMContLoss",
    "R2WMBridgeLoss",
    "R2WMBridgeCos",
    "R2WMRewardMAE",
    "R2WMRewardMAENonzero",
    "R2WMContAcc",
    "R2WMContAccTerminal",
    "R2WMDynEntropy",
    "R2WMRepEntropy",
    "R2WMUpdate",
    "R2ValueLoss",
    "R2ValueMean",
    "R2BootMean",
    "R2ValueBootMAE",
    "R2PolicyBridgeKL",
    "R2ImagActorLoss",
    "R2ImagValueLoss",
    "R2ImagReturn",
    "R2ImagEntropy",
    "R2ImagCont",
    "R2ImagValue",
    "R2ReturnScale",
)


def zero_param_subtree(grads, key):
    if key not in grads["params"]:
        return grads
    zero_subtree = jax.tree_util.tree_map(jnp.zeros_like, grads["params"][key])
    return flax.core.freeze(replace_mapping(grads, {
        "params": replace_mapping(grads["params"], {key: zero_subtree})
    }))


def replace_mapping(mapping, replacements):
    if isinstance(mapping, FrozenDict):
        return mapping.copy(add_or_replace=replacements)
    updated = dict(mapping)
    updated.update(replacements)
    return updated


def train(
    network_def,  # 0, static
    online_params,  # 1
    target_params,  # 2
    optimizer,  # 3, static
    optimizer_state,  # 4
    r2wm_optimizer_state,
    raw_states,  # 5
    actions,  # 6
    raw_next_states,  # 7
    rewards,  # 8
    terminals,  # 9
    same_traj_mask,  # 10
    loss_weights,  # 11
    support,  # 12
    cumulative_gamma,  # 13
    double_dqn,  # 14, static
    distributional,  # 15, static
    rng,  # 16
    spr_weight,  # 17, static (gates rollouts)
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
    world_model_weight,
    reward_weight,
    continue_weight,
    barlow_weight,
    barlow_lambd,
    imag_horizon,
    imag_actor_weight,
    imag_value_weight,
    imag_discount,
    imag_lambda,
    imag_entropy_weight,
    r2wm_enabled,  # static
    r2wm_raw_states,
    r2wm_actions,
    r2wm_rewards,
    r2wm_terminals,
    r2wm_is_first,
    r2wm_initial_stoch,
    r2wm_initial_deter,
    r2wm_update_mask,
    r2wm_learning_rate,
    r2wm_beta1,
    r2wm_beta2,
    r2wm_eps,
    r2wm_warmup,
    r2wm_agc,
    r2wm_pmin,
    r2_value_weight,
    r2_value_lambda,
    r2_value_discount,
    r2_imag_horizon,  # static
    r2_imag_start_count,  # static
    r2_imag_return_norm,  # static
    r2_imag_actor_weight,
    r2_imag_value_weight,
    r2_imag_entropy_weight,
    r2_imag_lambda,
    r2_imag_discount,
    r2_imag_unimix,
    r2_imag_scale,
    return_ema_vals,
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
            return_ema_vals,
            rng,
            step,
        ) = state
        (
            raw_states,
            actions,
            raw_next_states,
            rewards,
            terminals,
            same_traj_mask,
            loss_weights,
            cumulative_gamma,
            r2wm_raw_states,
            r2wm_actions,
            r2wm_rewards,
            r2wm_terminals,
            r2wm_is_first,
            r2wm_initial_stoch,
            r2wm_initial_deter,
            r2wm_do_update,
        ) = inputs
        transition_rewards = rewards[:, :-1]
        model_continue_targets = same_traj_mask[:, 1:].astype(jnp.float32)
        model_transition_mask = jnp.concatenate(
            [
                jnp.ones_like(model_continue_targets[:, :1]),
                model_continue_targets[:, :-1],
            ],
            axis=1,
        )
        same_traj_mask = same_traj_mask[:, 1:]
        td_rewards = rewards[:, 0]
        td_terminals = terminals[:, 0]
        td_cumulative_gamma = cumulative_gamma[:, 0]

        rng, rng1, rng2 = jax.random.split(rng, num=3)
        states = spr_networks.process_inputs(
            raw_states,
            rng=rng1,
            data_augmentation=data_augmentation,
            dtype=dtype)
        next_states = spr_networks.process_inputs(
            raw_next_states[:, 0],
            rng=rng2,
            data_augmentation=data_augmentation,
            dtype=dtype,
        )
        current_state = states[:, 0]

        # Split the current rng to update the rng after this call
        rng, rng1, rng2 = jax.random.split(rng, num=3)

        batch_rngs = jax.random.split(rng, num=states.shape[0])
        if match_online_target_rngs:
            target_rng = batch_rngs
        else:
            target_rng = jax.random.split(rng1, num=states.shape[0])
        use_spr = spr_weight > 0 or world_model_weight > 0 or imag_horizon > 0

        def policy_online(state, action_sample_key):
            return network_def.apply(
                online_params,
                state,
                rngs={"action_sample": action_sample_key},
                method=network_def.get_policy,
            )

        def q_target(state):
            return network_def.apply(
                target_params,
                state,
                support=support,
                eval_mode=target_eval_mode,
            )

        def encode_project(state):
            return network_def.apply(
                target_params,
                state,
                eval_mode=True,
                method=network_def.encode_project,
            )

        def loss_fn(
            params,
            target,
            spr_targets,
            loss_multipliers,
            key,
            r2wm_key,
            r2wm_aug_key,
        ):

            def all_results(state, actions, do_rollout):
                return network_def.apply(params,
                                         state,
                                         support,
                                         actions,
                                         do_rollout,
                                         method=network_def.init_fn)

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
                    #return -(jax.lax.stop_gradient(q_values) *
                    #         log_prob[samples]), x_ent

            x, logits = jax.vmap(all_results,
                                 in_axes=(0, 0, None),
                                 axis_name="batch")(current_state,
                                                    actions[:, :-1], use_spr)
            predicted_features = x.latent
            spr_predictions = predicted_features
            q_logits = jnp.squeeze(x.logits)
            chosen_action_logits = q_logits[jnp.arange(q_logits.shape[0]),
                                            actions[:, 0]]
            dqn_loss = jax.vmap(softmax_cross_entropy_loss_with_logits)(
                target, chosen_action_logits)
            td_error = dqn_loss + jnp.nan_to_num(
                target * jnp.log(target)).sum(-1)

            spr_predictions = spr_predictions.transpose(1, 0, 2)

            spr_predictions = spr_predictions / jnp.linalg.norm(
                spr_predictions, 2, -1, keepdims=True)

            spr_targets = spr_targets / jnp.linalg.norm(
                spr_targets, 2, -1, keepdims=True)
            spr_loss = jnp.power(spr_predictions - spr_targets, 2).sum(-1)
            #logging.info("spr_loss.shape: {}".format(spr_loss.shape))
            spr_loss = (spr_loss * same_traj_mask.transpose(1, 0)).mean(0) * .5
            #logging.info("spr_loss.shape: {}".format(spr_loss.shape))
            #exit(0)
            loss = dqn_loss + spr_weight * spr_loss
            loss = loss_multipliers * loss

            mean_loss = jnp.mean(loss)

            def reward_from_feature(feature):
                return network_def.apply(
                    params,
                    feature,
                    method=network_def.reward_from_feature,
                )

            def continue_from_feature(feature):
                return network_def.apply(
                    params,
                    feature,
                    method=network_def.continue_from_feature,
                )

            predicted_rewards = jax.vmap(jax.vmap(reward_from_feature,
                                                  in_axes=0),
                                         in_axes=0)(predicted_features)
            continue_logits = jax.vmap(jax.vmap(continue_from_feature,
                                                in_axes=0),
                                       in_axes=0)(predicted_features)
            reward_loss = masked_mean(
                jnp.square(predicted_rewards - transition_rewards),
                model_transition_mask,
            )
            continue_loss = masked_mean(
                sigmoid_binary_cross_entropy(continue_logits,
                                             model_continue_targets),
                model_transition_mask,
            )
            barlow_loss = weighted_barlow_twins_loss(
                spr_predictions,
                spr_targets,
                same_traj_mask.transpose(1, 0),
                lambd=barlow_lambd,
            )
            model_loss = (
                reward_weight * reward_loss +
                continue_weight * continue_loss +
                barlow_weight * barlow_loss)

            imag_actor_loss = jnp.asarray(0.0, dtype=mean_loss.dtype)
            imag_value_loss = jnp.asarray(0.0, dtype=mean_loss.dtype)
            if imag_horizon > 0:

                def imagine_one(state, imagine_key):
                    return network_def.apply(
                        params,
                        state,
                        imag_horizon,
                        rngs={"action_sample": imagine_key},
                        method=network_def.imagine_from_observation,
                    )

                imagined = jax.vmap(imagine_one,
                                    in_axes=(0, 0),
                                    axis_name="batch")(current_state, key)
                imag_returns = lambda_return(
                    jax.lax.stop_gradient(imagined['rewards']),
                    jax.lax.stop_gradient(imagined['continues']),
                    jax.lax.stop_gradient(imagined['values']),
                    imag_discount,
                    imag_lambda,
                )
                imag_values = imagined['values'][:, :-1]
                imag_advantage = jax.lax.stop_gradient(imag_returns -
                                                       imag_values)
                imag_weights = jax.lax.stop_gradient(
                    jnp.cumprod(imagined['continues'][:, :-1] *
                                imag_discount,
                                axis=1))
                imag_actor_loss = -jnp.mean(
                    imag_weights *
                    (imagined['log_probs'][:, :-1] * imag_advantage +
                     imag_entropy_weight * imagined['entropies'][:, :-1]))
                imag_value_loss = jnp.mean(
                    imag_weights *
                    jnp.square(imag_values -
                               jax.lax.stop_gradient(imag_returns)))

            x = jax.vmap(policy_loss, in_axes=0, axis_name="batch")(x.q_values,
                                                                    logits, key)
            policy_aux_loss = jnp.mean(loss_multipliers * x[0])
            total_loss = (mean_loss + policy_aux_loss +
                          world_model_weight * model_loss +
                          imag_actor_weight * imag_actor_loss +
                          imag_value_weight * imag_value_loss)
            aux_losses = {
                "TotalLoss": jnp.mean(total_loss),
                "DQNLoss": jnp.mean(dqn_loss),
                "TD Error": jnp.mean(td_error),
                "SPRLoss": jnp.mean(spr_loss),
                "WorldModelLoss": jnp.mean(model_loss),
                "RewardLoss": jnp.mean(reward_loss),
                "ContinueLoss": jnp.mean(continue_loss),
                "BarlowLoss": jnp.mean(barlow_loss),
                "ImagActorLoss": jnp.mean(imag_actor_loss),
                "ImagValueLoss": jnp.mean(imag_value_loss),
                "ent": jnp.mean(x[1]),
            }
            zero_post_stoch = jnp.zeros(
                r2wm_raw_states.shape[:2] + r2wm_initial_stoch.shape[1:],
                dtype=jnp.float32)
            zero_post_deter = jnp.zeros(
                r2wm_raw_states.shape[:2] + r2wm_initial_deter.shape[1:],
                dtype=jnp.float32)

            def zero_r2wm_metrics():
                return {
                    k: jnp.array(0.0, dtype=jnp.float32)
                    for k in R2WM_METRIC_KEYS
                }

            if r2wm_enabled:

                def run_r2wm_loss(_):
                    r2wm_states = spr_networks.process_inputs(
                        r2wm_raw_states,
                        rng=r2wm_aug_key,
                        data_augmentation=data_augmentation,
                        dtype=dtype,
                    )
                    (r2wm_loss, r2wm_metrics, post_stoch, post_deter,
                     extras) = network_def.apply(
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
                     )
                    r2wm_metrics["R2WMUpdate"] = jnp.array(1.0,
                                                           dtype=jnp.float32)
                    seq_b, seq_t = r2wm_raw_states.shape[:2]

                    # === Stage 1: replay value learning on RSSM features,
                    # bootstrapped from the BBF critic, plus bridge/policy
                    # agreement diagnostics. ===
                    embed_sg = jax.lax.stop_gradient(extras["embed"])
                    flat_embed = embed_sg.reshape(seq_b * seq_t, -1)

                    def q_from_repr(representation):
                        return network_def.apply(
                            params,
                            representation,
                            support,
                            method=network_def.q_from_representation,
                        )

                    q_values = jax.vmap(q_from_repr)(flat_embed)
                    repr_policy_logits = network_def.apply(
                        params,
                        flat_embed,
                        False,
                        method=network_def.policy_logits_from_feature,
                    )
                    pi_probs = jax.nn.softmax(repr_policy_logits)
                    boot = jax.lax.stop_gradient(
                        jnp.sum(pi_probs * q_values,
                                axis=-1)).reshape(seq_b, seq_t)

                    reward_seq = jnp.squeeze(r2wm_rewards,
                                             axis=-1).astype(jnp.float32)
                    term_seq = jnp.squeeze(r2wm_terminals,
                                           axis=-1).astype(jnp.float32)
                    first_seq = jnp.squeeze(r2wm_is_first,
                                            axis=-1).astype(jnp.float32)
                    last_seq = jnp.concatenate(
                        [first_seq[:, 1:],
                         jnp.zeros_like(first_seq[:, :1])],
                        axis=1)
                    replay_ret = r2_lambda_return(last_seq, term_seq,
                                                  reward_seq, boot,
                                                  r2_value_discount,
                                                  r2_value_lambda)
                    value_logits = extras["value_logits"]
                    slow_value_logits = network_def.apply(
                        target_params,
                        jax.lax.stop_gradient(extras["feat"]),
                        method=network_def.r2_value_logits_from_feat,
                    )
                    slow_value = jax.lax.stop_gradient(
                        spr_networks._r2_twohot_mode(slow_value_logits, 255))
                    value_nll_target = spr_networks._r2_twohot_neg_log_prob(
                        value_logits[:, :-1],
                        jax.lax.stop_gradient(replay_ret)[..., None], 255)
                    value_nll_slow = spr_networks._r2_twohot_neg_log_prob(
                        value_logits[:, :-1], slow_value[:, :-1][..., None],
                        255)
                    value_weight_mask = (1.0 - last_seq)[:, :-1]
                    value_replay_loss = jnp.mean(
                        value_weight_mask * (value_nll_target + value_nll_slow))
                    value_mode = jax.lax.stop_gradient(
                        spr_networks._r2_twohot_mode(value_logits, 255))

                    bridge_sg = jax.lax.stop_gradient(
                        extras["bridge_pred"]).reshape(seq_b * seq_t, -1)
                    bridge_policy_logits = network_def.apply(
                        params,
                        bridge_sg,
                        False,
                        method=network_def.policy_logits_from_feature,
                    )
                    policy_kl = jnp.mean(
                        jnp.sum(
                            pi_probs *
                            (jax.nn.log_softmax(repr_policy_logits) -
                             jax.nn.log_softmax(bridge_policy_logits)),
                            axis=-1))

                    r2wm_metrics["R2ValueLoss"] = value_replay_loss
                    r2wm_metrics["R2ValueMean"] = jnp.mean(value_mode)
                    r2wm_metrics["R2BootMean"] = jnp.mean(boot)
                    r2wm_metrics["R2ValueBootMAE"] = jnp.mean(
                        jnp.abs(value_mode - boot))
                    r2wm_metrics["R2PolicyBridgeKL"] = jax.lax.stop_gradient(
                        policy_kl)

                    r2wm_total = r2wm_loss + r2_value_weight * value_replay_loss
                    new_ema_vals = return_ema_vals

                    # === Stage 2/3: imagination on the RSSM prior under the
                    # shared policy (via the bridge), r2dreamer-style. ===
                    imag_actor_loss = jnp.array(0.0, dtype=jnp.float32)
                    imag_value_loss = jnp.array(0.0, dtype=jnp.float32)
                    imag_ret_mean = jnp.array(0.0, dtype=jnp.float32)
                    imag_entropy_mean = jnp.array(0.0, dtype=jnp.float32)
                    imag_cont_mean = jnp.array(0.0, dtype=jnp.float32)
                    imag_value_mean = jnp.array(0.0, dtype=jnp.float32)
                    ret_scale = jnp.array(1.0, dtype=jnp.float32)
                    if r2_imag_horizon > 0:
                        sel_key, roll_key = jax.random.split(r2wm_key)
                        flat_stoch = jax.lax.stop_gradient(
                            post_stoch.reshape(seq_b * seq_t,
                                               *post_stoch.shape[2:]))
                        flat_deter = jax.lax.stop_gradient(
                            post_deter.reshape(seq_b * seq_t,
                                               *post_deter.shape[2:]))
                        total_starts = seq_b * seq_t
                        if 0 < r2_imag_start_count < total_starts:
                            sel = jax.random.choice(sel_key,
                                                    total_starts,
                                                    (r2_imag_start_count,),
                                                    replace=False)
                            flat_stoch = flat_stoch[sel]
                            flat_deter = flat_deter[sel]
                        imag_feat, imag_action = network_def.apply(
                            params,
                            flat_stoch,
                            flat_deter,
                            r2_imag_horizon,
                            r2_imag_unimix,
                            roll_key,
                            method=network_def.r2_imagine,
                        )
                        bridge_imag = jax.lax.stop_gradient(
                            network_def.apply(
                                params,
                                imag_feat,
                                method=network_def.r2_bridge_from_feat,
                            ))
                        imag_policy_logits = network_def.apply(
                            params,
                            bridge_imag,
                            False,
                            method=network_def.policy_logits_from_feature,
                        )
                        imag_probs = jax.nn.softmax(
                            imag_policy_logits.astype(jnp.float32))
                        imag_probs = (
                            imag_probs * (1.0 - r2_imag_unimix) +
                            r2_imag_unimix / imag_probs.shape[-1])
                        imag_logp_all = jnp.log(imag_probs)
                        logpi = jnp.sum(imag_logp_all * imag_action, axis=-1)
                        imag_entropy = -jnp.sum(imag_probs * imag_logp_all,
                                                axis=-1)
                        imag_reward, imag_cont, imag_value_logits = (
                            network_def.apply(
                                params,
                                imag_feat,
                                method=network_def.r2_imag_heads,
                            ))
                        imag_value = jax.lax.stop_gradient(
                            spr_networks._r2_twohot_mode(
                                imag_value_logits, 255))
                        imag_slow_value = jax.lax.stop_gradient(
                            spr_networks._r2_twohot_mode(
                                network_def.apply(
                                    target_params,
                                    imag_feat,
                                    method=network_def.r2_value_logits_from_feat,
                                ), 255))
                        imag_ret = r2_lambda_return(
                            jnp.zeros_like(imag_cont), 1.0 - imag_cont,
                            imag_reward, imag_value, r2_imag_discount,
                            r2_imag_lambda)
                        ret_quantiles = jnp.quantile(
                            jax.lax.stop_gradient(imag_ret).reshape(-1),
                            jnp.array([0.05, 0.95], dtype=jnp.float32))
                        new_ema_vals = (0.01 * ret_quantiles +
                                        0.99 * return_ema_vals)
                        if r2_imag_return_norm:
                            ret_scale = jnp.maximum(
                                new_ema_vals[1] - new_ema_vals[0], 1.0)
                        adv = jax.lax.stop_gradient(
                            (imag_ret - imag_value[:, :-1]) / ret_scale)
                        imag_weight = jax.lax.stop_gradient(
                            jnp.cumprod(imag_cont * r2_imag_discount, axis=1))
                        imag_actor_loss = jnp.mean(
                            imag_weight[:, :-1] *
                            -(logpi[:, :-1] * adv +
                              r2_imag_entropy_weight * imag_entropy[:, :-1]))
                        imag_nll_target = spr_networks._r2_twohot_neg_log_prob(
                            imag_value_logits[:, :-1],
                            jax.lax.stop_gradient(imag_ret)[..., None], 255)
                        imag_nll_slow = spr_networks._r2_twohot_neg_log_prob(
                            imag_value_logits[:, :-1],
                            imag_slow_value[:, :-1][..., None], 255)
                        imag_value_loss = jnp.mean(
                            imag_weight[:, :-1] *
                            (imag_nll_target + imag_nll_slow))
                        r2wm_total = r2wm_total + r2_imag_scale * (
                            r2_imag_actor_weight * imag_actor_loss +
                            r2_imag_value_weight * imag_value_loss)
                        imag_ret_mean = jnp.mean(imag_ret)
                        imag_entropy_mean = jnp.mean(imag_entropy)
                        imag_cont_mean = jnp.mean(imag_cont)
                        imag_value_mean = jnp.mean(imag_value)

                    r2wm_metrics["R2ImagActorLoss"] = imag_actor_loss
                    r2wm_metrics["R2ImagValueLoss"] = imag_value_loss
                    r2wm_metrics["R2ImagReturn"] = imag_ret_mean
                    r2wm_metrics["R2ImagEntropy"] = imag_entropy_mean
                    r2wm_metrics["R2ImagCont"] = imag_cont_mean
                    r2wm_metrics["R2ImagValue"] = imag_value_mean
                    r2wm_metrics["R2ReturnScale"] = ret_scale
                    r2wm_metrics = {
                        k: r2wm_metrics[k] for k in R2WM_METRIC_KEYS
                    }
                    return (r2wm_total, r2wm_metrics, post_stoch, post_deter,
                            new_ema_vals)

                def skip_r2wm_loss(_):
                    return (jnp.array(0.0, dtype=jnp.float32),
                            zero_r2wm_metrics(), zero_post_stoch,
                            zero_post_deter, return_ema_vals)

                (r2wm_loss, r2wm_metrics, post_stoch, post_deter,
                 new_return_ema_vals) = jax.lax.cond(
                     r2wm_do_update,
                     run_r2wm_loss,
                     skip_r2wm_loss,
                     operand=None,
                 )
                total_loss = total_loss + r2wm_loss
                aux_losses.update(r2wm_metrics)
            else:
                post_stoch = zero_post_stoch
                post_deter = zero_post_deter
                new_return_ema_vals = return_ema_vals

            return total_loss, (aux_losses, (post_stoch, post_deter),
                                new_return_ema_vals)

        # Use the weighted mean loss for gradient computation.
        target = jax.vmap(target_output,
                          in_axes=(None, None, 0, 0, 0, None, 0, 0),
                          axis_name="batch")(
                              policy_online,
                              q_target,
                              next_states,
                              td_rewards,
                              td_terminals,
                              support,
                              td_cumulative_gamma,
                              target_rng,
                          )

        future_states = states[:, 1:]
        spr_targets = jax.vmap(jax.vmap(encode_project,
                                        in_axes=0,
                                        axis_name="time"),
                               in_axes=0,
                               axis_name="batch")(future_states)
        spr_targets = spr_targets.transpose(1, 0, 2)

        # Get the unweighted loss without taking its mean for updating priorities.
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        rng2, r2wm_key, r2wm_aug_key, policy_key = jax.random.split(rng2, 4)
        x = jax.random.split(policy_key, current_state.shape[0] + 1)
        rng2 = x[0]
        key = x[1:]
        (_, (aux_losses, r2wm_posts, new_return_ema_vals)), grad = grad_fn(
            online_params,
            target,
            spr_targets,
            loss_weights,
            key,
            r2wm_key,
            r2wm_aug_key,
        )

        bbf_grad = zero_param_subtree(grad, "r2_world_model")
        updates, new_optimizer_state = optimizer.update(
            bbf_grad, optimizer_state, params=online_params)
        new_online_params = optax.apply_updates(online_params, updates)

        if r2wm_enabled:

            def update_r2wm(args):
                new_online_params, r2wm_optimizer_state = args
                r2wm_updates, r2wm_optimizer_state = r2_laprop_update(
                    grad["params"]["r2_world_model"],
                    r2wm_optimizer_state,
                    online_params["params"]["r2_world_model"],
                    r2wm_learning_rate,
                    r2wm_beta1,
                    r2wm_beta2,
                    r2wm_eps,
                    r2wm_warmup,
                    r2wm_agc,
                    r2wm_pmin,
                )
                new_r2wm_params = optax.apply_updates(
                    online_params["params"]["r2_world_model"], r2wm_updates)
                new_online_params = replace_mapping(
                    new_online_params, {
                        "params": replace_mapping(
                            new_online_params["params"],
                            {"r2_world_model": new_r2wm_params})
                    })
                return flax.core.freeze(new_online_params), r2wm_optimizer_state

            def skip_r2wm_update(args):
                return args

            new_online_params, r2wm_optimizer_state = jax.lax.cond(
                r2wm_do_update,
                update_r2wm,
                skip_r2wm_update,
                operand=(new_online_params, r2wm_optimizer_state),
            )

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
                new_return_ema_vals,
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
        return_ema_vals,
        rng,
        step,
    )
    assert raw_states.shape[0] % batch_size == 0
    num_batches = raw_states.shape[0] // batch_size

    # debug - start
    #print(" num_batches: {}\n batch_size: {}".format(num_batches, batch_size))
    #print(" raw_policy_states: {}".format(raw_policy_states))
    #exit(0)
    # debug - end

    inputs = (
        raw_states.reshape(num_batches, batch_size, *raw_states.shape[1:]),
        actions.reshape(num_batches, batch_size, *actions.shape[1:]),
        raw_next_states.reshape(num_batches, batch_size,
                                *raw_next_states.shape[1:]),
        rewards.reshape(num_batches, batch_size, *rewards.shape[1:]),
        terminals.reshape(num_batches, batch_size, *terminals.shape[1:]),
        same_traj_mask.reshape(num_batches, batch_size,
                               *same_traj_mask.shape[1:]),
        loss_weights.reshape(num_batches, batch_size, *loss_weights.shape[1:]),
        cumulative_gamma.reshape(num_batches, batch_size,
                                 *cumulative_gamma.shape[1:]),
        r2wm_raw_states.reshape(num_batches, -1, *r2wm_raw_states.shape[1:]),
        r2wm_actions.reshape(num_batches, -1, *r2wm_actions.shape[1:]),
        r2wm_rewards.reshape(num_batches, -1, *r2wm_rewards.shape[1:]),
        r2wm_terminals.reshape(num_batches, -1, *r2wm_terminals.shape[1:]),
        r2wm_is_first.reshape(num_batches, -1, *r2wm_is_first.shape[1:]),
        r2wm_initial_stoch.reshape(num_batches, -1,
                                   *r2wm_initial_stoch.shape[1:]),
        r2wm_initial_deter.reshape(num_batches, -1,
                                   *r2wm_initial_deter.shape[1:]),
        r2wm_update_mask.reshape(num_batches),
    )

    (
        (
            online_params,
            target_params,
            optimizer_state,
            r2wm_optimizer_state,
            return_ema_vals,
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
        return_ema_vals,
        {k: jnp.reshape(v, (-1,)) for k, v in aux_losses.items()},
        r2wm_posts[0],
        r2wm_posts[1],
    )


def target_output(
    policy_info,
    target_network,
    next_states,
    rewards,
    terminals,
    support,
    cumulative_gamma,
    rng,
):
    gamma_with_terminal = (cumulative_gamma *
                           (1.0 - terminals.astype(jnp.float32)))
    target_dist = target_network(next_states)
    _, next_qt_argmax = policy_info(next_states, rng)

    # Compute the target Q-value distribution
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
        world_model_weight=0.0,
        reward_weight=1.0,
        continue_weight=1.0,
        barlow_weight=0.05,
        barlow_lambd=5e-4,
        imag_horizon=0,
        imag_actor_weight=0.0,
        imag_value_weight=0.0,
        imag_discount=None,
        imag_lambda=0.95,
        imag_entropy_weight=3e-4,
        half_precision=False,
        seed=None,
        log_every=None,
        explore_end_steps=None,
        r2_world_model_enabled=False,
        r2_world_model_batch_size=16,
        r2_world_model_batch_length=64,
        r2_world_model_replay_capacity=500000,
        r2_world_model_learning_rate=4e-5,
        r2_world_model_beta1=0.9,
        r2_world_model_beta2=0.999,
        r2_world_model_eps=1e-20,
        r2_world_model_warmup=1000,
        r2_world_model_agc=0.3,
        r2_world_model_pmin=1e-3,
        r2_world_model_update_period=1,
        r2_world_model_stoch=32,
        r2_world_model_deter=6144,
        r2_world_model_hidden=768,
        r2_world_model_discrete=48,
        r2_world_model_units=768,
        r2_world_model_blocks=8,
        r2_world_model_clip_reward=True,
        r2_bridge_weight=1.0,
        r2_value_through_wm=False,
        r2_value_weight=0.3,
        r2_value_lambda=0.95,
        r2_imag_horizon=0,
        r2_imag_start_count=256,
        r2_imag_actor_weight=0.0,
        r2_imag_value_weight=0.0,
        r2_imag_entropy_weight=3e-4,
        r2_imag_lambda=0.95,
        r2_imag_discount=None,
        r2_imag_unimix=0.01,
        r2_imag_return_norm=False,
        r2_imag_reset_freeze_steps=2000,
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
        self.world_model_weight = float(world_model_weight)
        self.reward_weight = float(reward_weight)
        self.continue_weight = float(continue_weight)
        self.barlow_weight = float(barlow_weight)
        self.barlow_lambd = float(barlow_lambd)
        self.imag_horizon = int(imag_horizon)
        self.imag_actor_weight = float(imag_actor_weight)
        self.imag_value_weight = float(imag_value_weight)
        self.imag_discount = None if imag_discount is None else float(
            imag_discount)
        self.imag_lambda = float(imag_lambda)
        self.imag_entropy_weight = float(imag_entropy_weight)

        # debug - start
        print('*' * 20)
        print(' self.target_eval_mode: {}'.format(self.target_eval_mode))
        print(' self.target_action_selection: {}'.format(
            self.target_action_selection))
        print(" num_actions: {}".format(num_actions))
        print(" self.reset_target: {}".format(self.reset_target))
        print(" self.world_model_weight: {}".format(self.world_model_weight))
        print(" self.imag_horizon: {}".format(self.imag_horizon))
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
        self.r2_world_model_enabled = bool(r2_world_model_enabled)
        self.r2_world_model_batch_size = int(r2_world_model_batch_size)
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
        self.r2_world_model_update_period = max(
            1, int(r2_world_model_update_period))
        self.r2_world_model_stoch = int(r2_world_model_stoch)
        self.r2_world_model_deter = int(r2_world_model_deter)
        self.r2_world_model_hidden = int(r2_world_model_hidden)
        self.r2_world_model_discrete = int(r2_world_model_discrete)
        self.r2_world_model_units = int(r2_world_model_units)
        self.r2_world_model_blocks = int(r2_world_model_blocks)
        self.r2_world_model_clip_reward = bool(r2_world_model_clip_reward)
        self.r2_bridge_weight = float(r2_bridge_weight)
        self.r2_value_through_wm = bool(r2_value_through_wm)
        self.r2_value_weight = float(r2_value_weight)
        self.r2_value_lambda = float(r2_value_lambda)
        self.r2_imag_horizon = int(r2_imag_horizon)
        self.r2_imag_start_count = int(r2_imag_start_count)
        self.r2_imag_actor_weight = float(r2_imag_actor_weight)
        self.r2_imag_value_weight = float(r2_imag_value_weight)
        self.r2_imag_entropy_weight = float(r2_imag_entropy_weight)
        self.r2_imag_lambda = float(r2_imag_lambda)
        self.r2_imag_discount = None if r2_imag_discount is None else float(
            r2_imag_discount)
        self.r2_imag_unimix = float(r2_imag_unimix)
        self.r2_imag_return_norm = bool(r2_imag_return_norm)
        self.r2_imag_reset_freeze_steps = int(r2_imag_reset_freeze_steps)
        self._r2_imag_unfreeze_step = 0
        self._r2_return_ema_vals = np.zeros((2,), dtype=np.float32)

        logging.info("\t Running with dtype %s", str(self.dtype))

        super().__init__(
            num_actions=num_actions,
            network=functools.partial(
                network,
                num_atoms=self._num_atoms,
                noisy=False,
                distributional=self._distributional,
                dtype=self.dtype,
                r2_world_model_enabled=self.r2_world_model_enabled,
                r2_world_model_stoch=self.r2_world_model_stoch,
                r2_world_model_deter=self.r2_world_model_deter,
                r2_world_model_hidden=self.r2_world_model_hidden,
                r2_world_model_discrete=self.r2_world_model_discrete,
                r2_world_model_units=self.r2_world_model_units,
                r2_world_model_blocks=self.r2_world_model_blocks,
                r2_world_model_bridge_weight=self.r2_bridge_weight,
                r2_world_model_value_through_wm=self.r2_value_through_wm,
            ),
            target_update_period=self.target_update_period,
            update_horizon=self.max_update_horizon,
            seed=seed,
        )
        if self.imag_discount is None:
            self.imag_discount = self.gamma
        if self.r2_imag_discount is None:
            self.r2_imag_discount = self.gamma

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
                do_rollout=(self.spr_weight > 0 or
                            self.world_model_weight > 0 or
                            self.imag_horizon > 0),
                support=self._support,
            ))

        optimizer = create_scaling_optimizer(learning_rate=self.learning_rate,)
        encoder_optimizer = create_scaling_optimizer(
            learning_rate=self.encoder_learning_rate,)
        policy_optim = create_scaling_optimizer(learning_rate=1e-4,)

        encoder_keys = {"encoder", "transition_model"}
        encoder_mask = FrozenDict({
            "params": {
                k: k in encoder_keys for k in self.online_params["params"]
            }
        })

        head_keys = {
            "representation_projection", "projection", "head", "predictor",
            "reward_head", "continue_head", "value_head"
        }
        head_mask = FrozenDict({
            "params": {k: k in head_keys for k in self.online_params["params"]}
        })

        policy_key = {"policy_projection", "policy"}
        policy_mask = FrozenDict({
            "params": {
                k: k in policy_key for k in self.online_params["params"]
            }
        })

        alpha_optim = optax.sgd(learning_rate=-1e-3)
        alpha_key = {"_log_alpha"}
        alpha_mask = FrozenDict({
            "params": {k: k in alpha_key for k in self.online_params["params"]}
        })
        #print(" alpha_mask:\n{}".format(alpha_mask))
        #print(" policy_mask:\n{}".format(policy_mask))
        #exit(0)

        # debug - start
        if False:
            print(' self.head_mask: {}'.format(self.head_mask))
            print(
                ' jax.tree_util.tree_map(lambda x: x.shape, self.online_params["params"]["projection"]: {}'
                .format(
                    jax.tree_util.tree_map(
                        lambda x: x.shape,
                        self.online_params["params"]["projection"])))
            print(
                ' jax.tree_util.tree_map(lambda x: x.shape, self.online_params["params"]["predictor"]: {}'
                .format(
                    jax.tree_util.tree_map(
                        lambda x: x.shape,
                        self.online_params["params"]["predictor"])))
            print(
                ' jax.tree_util.tree_map(lambda x: x.shape, self.online_params["params"]["head"]: {}'
                .format(
                    jax.tree_util.tree_map(
                        lambda x: x.shape,
                        self.online_params["params"]["head"])))
            #exit(0)
        # debug - end

        self.optimizer = optax.chain(
            optax.masked(encoder_optimizer, encoder_mask),
            optax.masked(optimizer, head_mask),
            optax.masked(policy_optim, policy_mask),
            optax.masked(alpha_optim, alpha_mask),
        )

        self.optimizer_state = self.optimizer.init(self.online_params)
        self.target_network_params = copy.deepcopy(self.online_params)
        self.random_params = copy.deepcopy(self.online_params)
        if self.r2_world_model_enabled:
            self.r2wm_optimizer_state = r2_laprop_init(
                self.online_params["params"]["r2_world_model"])
        else:
            self.r2wm_optimizer_state = r2_laprop_init({})

        #print(' so far so good')
        #exit(0)

    def _build_replay_buffer(self):
        prioritized_buffer = subsequence_replay_buffer.PrioritizedJaxSubsequenceParallelEnvReplayBuffer(
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

    def _build_r2_world_model_replay(self):
        self._r2_replay = None
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
        if not self.r2_world_model_enabled:
            return

        from types import SimpleNamespace
        from buffer import Buffer

        config = SimpleNamespace(
            device="cpu",
            storage_device="cpu",
            batch_size=self.r2_world_model_batch_size *
            self._batches_to_group,
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

    def _empty_r2_world_model_batch(self):
        total_batch = self.r2_world_model_batch_size * self._batches_to_group
        shape = (total_batch, self.r2_world_model_batch_length)
        return {
            "state": np.zeros(shape + self.state_shape, dtype=self.observation_dtype),
            "action": np.zeros(shape + (self.num_actions,), dtype=np.float32),
            "reward": np.zeros(shape + (1,), dtype=np.float32),
            "is_terminal": np.zeros(shape + (1,), dtype=np.float32),
            "is_first": np.zeros(shape + (1,), dtype=np.float32),
            "initial_stoch": np.zeros(
                (total_batch, self.r2_world_model_stoch,
                 self.r2_world_model_discrete),
                dtype=np.float32),
            "initial_deter": np.zeros(
                (total_batch, self.r2_world_model_deter), dtype=np.float32),
            "index": None,
        }

    def _sample_r2_world_model_batch(self):
        if (not self.r2_world_model_enabled or self._r2_replay is None or
                self._r2_replay.count() <=
                self.r2_world_model_batch_length + self.stack_size + 1):
            return self._empty_r2_world_model_batch()

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

    def _r2_world_model_update_mask(self):
        if not self.r2_world_model_enabled:
            return np.zeros((self._batches_to_group,), dtype=np.bool_)
        update_ids = self.grad_steps + np.arange(self._batches_to_group) + 1
        return (update_ids % self.r2_world_model_update_period) == 0

    def _update_r2_world_model_buffer(self, index, stoch, deter, update_mask):
        if not self.r2_world_model_enabled or index is None:
            return
        update_mask = np.asarray(update_mask, dtype=np.bool_)
        if not np.any(update_mask):
            return
        stoch = np.asarray(jax.device_get(stoch), dtype=np.float32)
        deter = np.asarray(jax.device_get(deter), dtype=np.float32)
        per_update_batch = stoch.shape[1]
        flat_mask = np.repeat(update_mask, per_update_batch)
        selected_rows = np.nonzero(flat_mask)[0].tolist()
        index = [ind[selected_rows] for ind in index]
        stoch = stoch[update_mask]
        deter = deter[update_mask]
        stoch = stoch.reshape(-1, *stoch.shape[2:])
        deter = deter.reshape(-1, *deter.shape[2:])
        self._r2_replay.update(index, stoch, deter)

    def _r2_imag_scale(self):
        """Imagination loss multiplier; zero for a window after each reset.

        After shrink-and-perturb the embedding shifts and the world model
        needs some steps to re-adapt before its rollouts are trustworthy.
        """
        if self.training_steps < self._r2_imag_unfreeze_step:
            return 0.0
        return 1.0

    def _log_r2_metrics(self, aux_losses):
        if not self.r2_world_model_enabled:
            return
        if self.grad_steps % 200 >= self._batches_to_group:
            return
        anchor_parts = []
        for key in ("TotalLoss", "DQNLoss", "SPRLoss", "ent"):
            if key in aux_losses:
                anchor_parts.append("{}={:.4f}".format(
                    key, float(np.asarray(aux_losses[key]).mean())))
        logging.info("BBF step %s: %s", self.training_steps,
                     ", ".join(anchor_parts))
        update_mask = np.asarray(aux_losses.get("R2WMUpdate"))
        n_updates = float(update_mask.sum())
        if n_updates == 0:
            return
        parts = []
        for key in R2WM_METRIC_KEYS:
            if key == "R2WMUpdate":
                continue
            value = float(
                (np.asarray(aux_losses[key]) * update_mask).sum() / n_updates)
            parts.append("{}={:.4f}".format(key, value))
        logging.info("R2WM step %s: %s", self.training_steps,
                     ", ".join(parts))

    def _observe_r2_world_model_state(self):
        if not self.r2_world_model_enabled:
            return
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

    def add_r2_episode_boundary_transition(self, env_id, observation, reward,
                                           terminal):
        if not self.r2_world_model_enabled or self.eval_mode:
            return
        obs = np.asarray(observation).squeeze(-1)
        terminal_state = np.asarray(
            self.state[env_id], dtype=self.observation_dtype).copy()
        terminal_state = np.roll(terminal_state, -1, axis=-1)
        terminal_state[Ellipsis, -1] = np.reshape(obs, self.observation_shape)

        self._rng, rng = jax.random.split(self._rng)
        stoch, deter = r2_world_model_observe(
            self.network_def,
            self.online_params,
            terminal_state[None],
            self._r2_prev_action[env_id:env_id + 1],
            self._r2_stoch[env_id:env_id + 1],
            self._r2_deter[env_id:env_id + 1],
            self._r2_is_first[env_id:env_id + 1],
            rng,
            self.dtype,
        )
        zero_action = np.zeros((1, self.num_actions), dtype=np.float32)
        boundary_reward = np.asarray([reward], dtype=np.float32)
        if self.r2_world_model_clip_reward:
            boundary_reward = np.clip(boundary_reward, -1.0, 1.0)
        self._r2_replay.add_atari_transition(
            state=terminal_state[None],
            action=zero_action,
            reward=boundary_reward.reshape(1, 1),
            is_terminal=np.asarray([terminal],
                                   dtype=np.float32).reshape(1, 1),
            is_first=np.zeros((1, 1), dtype=np.float32),
            stoch=np.asarray(jax.device_get(stoch), dtype=np.float32),
            deter=np.asarray(jax.device_get(deter), dtype=np.float32),
        )

    def _flush_r2_pending_transition(self, action):
        if (not self.r2_world_model_enabled or self.eval_mode or
                self._r2_pending_transition is None):
            return
        action_onehot = self._one_hot_actions(action)
        self._r2_replay.add_atari_transition(
            state=self._r2_pending_transition["state"],
            action=action_onehot,
            reward=self._r2_pending_transition["reward"],
            is_terminal=self._r2_pending_transition["is_terminal"],
            is_first=self._r2_pending_transition["is_first"],
            stoch=self._r2_stoch,
            deter=self._r2_deter,
        )
        self._r2_pending_transition = None

    def _set_r2_pending_transition(self, reward, terminal, is_first):
        if not self.r2_world_model_enabled or self.eval_mode:
            return
        reward = np.asarray(reward, dtype=np.float32)
        if self.r2_world_model_clip_reward:
            reward = np.clip(reward, -1.0, 1.0)
        self._r2_pending_transition = {
            "state": np.asarray(self.state, dtype=self.observation_dtype).copy(),
            "reward": reward.reshape(-1, 1),
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
                        "representation_projection", "reward_head",
                        "continue_head", "value_head", "_log_alpha",
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
            (self.spr_weight > 0 or self.world_model_weight > 0 or
             self.imag_horizon > 0),
            self._support,
            self.reset_target,
            self.shrink_perturb_keys,
            self.shrink_factor,
            self.perturb_factor,
            keys_to_copy,
        )

        self._r2_imag_unfreeze_step = (self.training_steps +
                                       self.r2_imag_reset_freeze_steps)
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
        r2wm_update_mask = self._r2_world_model_update_mask()
        if np.any(r2wm_update_mask):
            r2wm_batch = self._sample_r2_world_model_batch()
        else:
            r2wm_batch = self._empty_r2_world_model_batch()
        (
            new_online_params,
            new_target_params,
            new_optimizer_state,
            new_r2wm_optimizer_state,
            new_return_ema_vals,
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
            self.replay_elements["state"],
            self.replay_elements["action"],
            self.replay_elements["next_state"],
            self.replay_elements["return"],
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
            #self.ent_targ,
            self.x_ent_coef,
            self.world_model_weight,
            self.reward_weight,
            self.continue_weight,
            self.barlow_weight,
            self.barlow_lambd,
            self.imag_horizon,
            self.imag_actor_weight,
            self.imag_value_weight,
            self.imag_discount,
            self.imag_lambda,
            self.imag_entropy_weight,
            self.r2_world_model_enabled,
            r2wm_batch["state"],
            r2wm_batch["action"],
            r2wm_batch["reward"],
            r2wm_batch["is_terminal"],
            r2wm_batch["is_first"],
            r2wm_batch["initial_stoch"],
            r2wm_batch["initial_deter"],
            r2wm_update_mask,
            self.r2_world_model_learning_rate,
            self.r2_world_model_beta1,
            self.r2_world_model_beta2,
            self.r2_world_model_eps,
            self.r2_world_model_warmup,
            self.r2_world_model_agc,
            self.r2_world_model_pmin,
            self.r2_value_weight,
            self.r2_value_lambda,
            self.gamma,
            self.r2_imag_horizon,
            self.r2_imag_start_count,
            self.r2_imag_return_norm,
            self.r2_imag_actor_weight,
            self.r2_imag_value_weight,
            self.r2_imag_entropy_weight,
            self.r2_imag_lambda,
            self.r2_imag_discount,
            self.r2_imag_unimix,
            self._r2_imag_scale(),
            jnp.asarray(self._r2_return_ema_vals, dtype=jnp.float32),
        )
        self._r2_return_ema_vals = np.asarray(
            jax.device_get(new_return_ema_vals), dtype=np.float32)
        self._update_r2_world_model_buffer(r2wm_batch["index"],
                                           r2wm_post_stoch,
                                           r2wm_post_deter,
                                           r2wm_update_mask)
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
        self._log_r2_metrics(aux_losses)

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

    def reset_all(self, new_obs):
        """Resets the agent state by filling it with zeros."""
        n_envs = new_obs.shape[0]
        self.state = np.zeros((n_envs, *self.state_shape))
        self._record_observation(new_obs)
        if self.r2_world_model_enabled:
            self._r2_stoch = np.zeros(
                (n_envs, self.r2_world_model_stoch,
                 self.r2_world_model_discrete),
                dtype=np.float32)
            self._r2_deter = np.zeros((n_envs, self.r2_world_model_deter),
                                      dtype=np.float32)
            self._r2_prev_action = np.zeros((n_envs, self.num_actions),
                                            dtype=np.float32)
            self._r2_is_first = np.ones((n_envs, 1), dtype=np.float32)
            self._set_r2_pending_transition(
                np.zeros((n_envs,), dtype=np.float32),
                np.zeros((n_envs,), dtype=np.float32),
                np.ones((n_envs,), dtype=np.float32),
            )

    def reset_one(self, env_id):
        self.state[env_id].fill(0)
        if self.r2_world_model_enabled:
            self._r2_stoch[env_id].fill(0.0)
            self._r2_deter[env_id].fill(0.0)
            self._r2_prev_action[env_id].fill(0.0)
            self._r2_is_first[env_id] = 1.0

    def delete_one(self, env_id):
        self.state = np.concatenate(
            [self.state[:env_id], self.state[env_id + 1:]], 0)
        if self.r2_world_model_enabled:
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
        )

    def restore_train_state(self):
        (self.state, self._last_observation, self._observation,
         self._r2_stoch, self._r2_deter, self._r2_prev_action,
         self._r2_is_first, self._r2_pending_transition) = (
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
            r2_reward = np.asarray(raw_reward, dtype=np.float32).copy()
            r2_terminal = np.asarray(terminal, dtype=np.float32).copy()
            if np.any(is_first):
                r2_reward = np.where(is_first, 0.0, r2_reward)
                r2_terminal = np.where(is_first, 0.0, r2_terminal)
            self._set_r2_pending_transition(r2_reward, r2_terminal, is_first)

    def select_action(
        self,
        state,
        select_params,
        eval_mode,
    ):
        if not eval_mode and self.training_steps < self.min_replay_history:
            self._rng, key = jax.random.split(self._rng)
            return jax.random.randint(
                key,
                (state.shape[0],),
                0,
                self.num_actions,
            )
        self._rng, action, probs = select_action(
            self.network_def,
            select_params,
            state,
            self._rng,
            self.num_actions,
            False,
            #eval_mode,
            #eval_mode=self.greedy_action,
            #eval_mode=True,
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
        state = self.state
        select_params = self.target_network_params
        #select_params = self.online_params
        ## time inference - start
        #self.eval_mode = True
        #import time
        #start_time = time.time()
        #for _ in range(int(100 * 32)):
        action = self.select_action(
            state,
            select_params,
            self.eval_mode,
        )
        self.action = np.asarray(action)
        if not self.eval_mode and self.r2_world_model_enabled:
            self._flush_r2_pending_transition(self.action)
            self._r2_prev_action = self._one_hot_actions(self.action)
        #time_delta = time.time() - start_time
        #print('time_delta: {}'.format(time_delta))
        #exit(0)
        return self.action
