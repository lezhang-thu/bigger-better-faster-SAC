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

# Parameters kept through shrink-and-perturb resets. Retained trunk and
# grounding modules have separate optimizer transforms from freshly reset
# Q/SPR and policy heads, so their moments and Adam counts can also survive.
RESET_PARAMETER_KEYS_TO_COPY = (
    "encoder",
    "transition_model",
    "reward_head",
    "continue_head",
    "_log_alpha",
)
RESET_OPTIMIZER_KEYS_TO_COPY = (
    "encoder",
    "transition_model",
    "reward_head",
    "continue_head",
)


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


def categorical_retrace(policy_probabilities, target_probabilities,
                        trace_actions, behavior_probabilities, rewards,
                        terminals, support, gamma, lamb):
    """Builds a finite-horizon Distributional Retrace target for C51.

    The sampled raw trajectory is ``S0,A0,R1,D1,...,RH,DH,SH``. Policy and
    target probabilities are evaluated at the endpoint states ``S1..SH``;
    ``trace_actions`` and ``behavior_probabilities`` contain only the
    intermediate behavior choices ``A1..A{H-1}``.  The final endpoint uses an
    ordinary target-policy mixture and therefore needs no behavior action.

    Following Reactor, the result is a signed categorical measure for an
    individual off-policy trajectory. Its mass is one, but individual bins may
    be negative; the linear cross-entropy gradient is still the unbiased
    Distributional Retrace update.

    Args:
      policy_probabilities: ``[H, A]`` current-policy probabilities.
      target_probabilities: ``[H, A, Z]`` target-critic distributions.
      trace_actions: ``[H - 1]`` replay actions at ``S1..S{H-1}``.
      behavior_probabilities: ``[H - 1]`` logged probabilities of those actions.
      rewards: ``[H]`` raw rewards ``R1..RH``.
      terminals: ``[H]`` raw terminal flags ``D1..DH``.
      support: ``[Z]`` C51 support.
      gamma: Fixed scalar discount.
      lamb: Retrace lambda in ``[0, 1]``.

    Returns:
      A stopped-gradient signed categorical target with shape ``[Z]``.
    """
    horizon = rewards.shape[0]
    if horizon < 1:
        raise ValueError("Retrace requires at least one transition.")
    if policy_probabilities.shape[0] != horizon:
        raise ValueError("Policy sequence length must match reward horizon.")
    if target_probabilities.shape[0] != horizon:
        raise ValueError("Target sequence length must match reward horizon.")
    if terminals.shape[0] != horizon:
        raise ValueError("Terminal sequence length must match reward horizon.")
    if trace_actions.shape[0] != horizon - 1:
        raise ValueError("Retrace needs one intermediate action per trace step.")
    if behavior_probabilities.shape[0] != horizon - 1:
        raise ValueError(
            "Retrace needs one behavior probability per trace step.")

    dtype = target_probabilities.dtype
    gamma = jnp.asarray(gamma, dtype=dtype)
    lamb = jnp.asarray(lamb, dtype=dtype)
    target = jnp.zeros_like(support, dtype=dtype)
    path_weight = jnp.asarray(1.0, dtype=dtype)
    cumulative_reward = jnp.asarray(0.0, dtype=dtype)
    discount = jnp.asarray(1.0, dtype=dtype)
    num_actions = target_probabilities.shape[1]
    tiny = jnp.asarray(jnp.finfo(dtype).tiny, dtype=dtype)

    for endpoint in range(horizon):
        cumulative_reward = cumulative_reward + discount * rewards[endpoint]
        done = terminals[endpoint].astype(dtype)

        # A true terminal consumes all residual trace mass at the accumulated
        # real return. Later replay rows may belong to another episode and have
        # exactly zero influence after this point.
        terminal_weight = path_weight * done
        target = target + project_distribution(
            jnp.asarray([cumulative_reward]),
            jnp.asarray([terminal_weight]), support)
        live_weight = path_weight * (1.0 - done)

        policy = policy_probabilities[endpoint]
        if endpoint == horizon - 1:
            coefficients = live_weight * policy
        else:
            replay_action = trace_actions[endpoint].astype(jnp.int32)
            behavior_probability = jnp.maximum(
                behavior_probabilities[endpoint].astype(dtype), tiny)
            ratio = policy[replay_action] / behavior_probability
            coefficient = lamb * jnp.minimum(jnp.asarray(1.0, dtype=dtype),
                                              ratio)
            coefficients = live_weight * (
                policy - jax.nn.one_hot(
                    replay_action, num_actions, dtype=dtype) * coefficient)
            path_weight = live_weight * coefficient

        transformed_support = cumulative_reward + discount * gamma * support
        branch_supports = jnp.broadcast_to(
            transformed_support,
            target_probabilities[endpoint].shape,
        ).reshape(-1)
        branch_weights = (
            coefficients[:, None] * target_probabilities[endpoint]).reshape(-1)
        target = target + project_distribution(branch_supports, branch_weights,
                                               support)
        discount = discount * gamma

    return jax.lax.stop_gradient(target)


def categorical_retrace_priority(target, predicted_probabilities):
    """Returns a bounded squared total-variation priority loss.

    Distributional Retrace targets can contain negative bins, so their linear
    cross-entropy value is not a nonnegative loss and cannot safely be square
    rooted for prioritized replay. Total variation is the Distributional
    Retrace priority used by Reactor. We cap its signed-sample extension at one
    before squaring; the caller's existing square root then writes a finite
    priority in ``[0, 1]`` and prevents a single signed target from permanently
    dominating the replay tree.
    """
    if target.shape != predicted_probabilities.shape:
        raise ValueError("Retrace target and prediction shapes must match.")
    total_variation = 0.5 * jnp.sum(
        jnp.abs(target - predicted_probabilities), axis=-1)
    return jnp.square(jnp.clip(total_variation, 0.0, 1.0))


def categorical_n_step_target(policy_probabilities, target_probabilities,
                              rewards, terminals, support, gamma):
    """Builds an uncorrected fixed-horizon C51 target from raw replay rows.

    ``rewards`` and ``terminals`` contain ``R1..RH`` and ``D1..DH`` for an
    anchor at ``(S0, A0)``. ``policy_probabilities`` and
    ``target_probabilities`` are evaluated only at the endpoint ``SH``. The
    target is

      Pi_Z[sum_k gamma**k alive_k R{k+1}
           + gamma**H alive_H Z(SH, A), A ~ pi(.|SH)].

    A terminal reward is included, while every later reward and the endpoint
    bootstrap are masked. The exact endpoint-policy mixture deliberately
    matches the final branch of ``categorical_retrace``.
    """
    if policy_probabilities.ndim != 1:
        raise ValueError("Endpoint policy probabilities must have shape [A].")
    if target_probabilities.ndim != 2:
        raise ValueError(
            "Endpoint target probabilities must have shape [A, Z].")
    if target_probabilities.shape[0] != policy_probabilities.shape[0]:
        raise ValueError(
            "Endpoint policy and target action dimensions must match.")
    if target_probabilities.shape[1] != support.shape[0]:
        raise ValueError("Endpoint target and support atom counts must match.")
    if rewards.ndim != 1 or terminals.ndim != 1:
        raise ValueError("n-step rewards and terminals must be rank one.")
    if rewards.shape != terminals.shape:
        raise ValueError("n-step rewards and terminals must have equal shape.")
    if rewards.shape[0] < 1:
        raise ValueError("n-step target horizon must be positive.")

    dtype = target_probabilities.dtype
    gamma = jnp.asarray(gamma, dtype=dtype)
    cumulative_reward = jnp.asarray(0.0, dtype=dtype)
    discount = jnp.asarray(1.0, dtype=dtype)
    alive = jnp.asarray(1.0, dtype=dtype)
    for step_index in range(rewards.shape[0]):
        cumulative_reward = (
            cumulative_reward + discount * alive * rewards[step_index])
        alive = alive * (1.0 - terminals[step_index].astype(dtype))
        discount = discount * gamma

    endpoint_probabilities = jnp.einsum(
        "a,az->z", policy_probabilities, target_probabilities)
    transformed_support = cumulative_reward + discount * alive * support
    target = project_distribution(transformed_support,
                                  endpoint_probabilities, support)
    return jax.lax.stop_gradient(target)


def retrace_target_active(enabled, cycle_grad_steps,
                          warmup_n_step_updates):
    """Whether this cycle has finished its n-step target warmup."""
    return bool(enabled and
                int(cycle_grad_steps) >= int(warmup_n_step_updates))


def retrace_priority_reset_due(enabled, reset_on_target_switch,
                               cycle_grad_steps, warmup_n_step_updates):
    """Whether this update is the n-step-to-Retrace priority boundary."""
    return bool(enabled and reset_on_target_switch and
                int(warmup_n_step_updates) > 0 and
                int(cycle_grad_steps) == int(warmup_n_step_updates))


def imagination_training_scale(cycle_grad_steps, imag_warmup, ramp=True):
    """Returns the cycle-local multiplier for imagined actor/value losses."""
    cycle_grad_steps = int(cycle_grad_steps)
    imag_warmup = int(imag_warmup)
    if not ramp:
        return float(cycle_grad_steps >= imag_warmup)
    return float(
        np.clip(
            (cycle_grad_steps - imag_warmup) / max(1, imag_warmup),
            0.0,
            1.0,
        ))


def n_step_lower_bound_target(rewards, terminals, bootstrap_value, gamma):
    """Returns a terminal-truncated scalar n-step target.

    Replay rows follow ``(S_t, A_t, R_{t+1}, D_{t+1})``. The terminal reward
    is included, while all later rewards and the bootstrap are masked. This
    helper is also used to reconstruct the reward prefix for projected C51
    lower-bound and maximization targets from raw one-step replay rows.
    """
    if rewards.ndim != 1 or terminals.ndim != 1:
        raise ValueError("n-step rewards and terminals must be rank one.")
    if rewards.shape != terminals.shape:
        raise ValueError("n-step rewards and terminals must have equal shape.")
    if rewards.shape[0] < 1:
        raise ValueError("n-step target horizon must be positive.")

    dtype = rewards.dtype
    gamma = jnp.asarray(gamma, dtype=dtype)
    value = jnp.asarray(0.0, dtype=dtype)
    discount = jnp.asarray(1.0, dtype=dtype)
    alive = jnp.asarray(1.0, dtype=dtype)
    for step_index in range(rewards.shape[0]):
        value = value + discount * alive * rewards[step_index]
        alive = alive * (1.0 - terminals[step_index].astype(dtype))
        discount = discount * gamma
    return value + discount * alive * jnp.asarray(bootstrap_value, dtype=dtype)


def lower_bound_td_signals(q_value, one_step_target, n_step_target,
                           priority_eta=0.5, priority_epsilon=1e-6):
    """Builds the one-step error, upward-only gap, and proposal priority u_t.

    Target values are frozen, but the positive gap retains its derivative with
    respect to ``q_value`` so minimizing a loss of that gap can only raise the
    predicted root value. The priority is fully stopped and is never itself an
    optimization objective.
    """
    one_step_target = jax.lax.stop_gradient(jnp.asarray(one_step_target))
    n_step_target = jax.lax.stop_gradient(jnp.asarray(n_step_target))
    q_value = jnp.asarray(q_value)
    delta_one_step = one_step_target - q_value
    positive_lower_bound = jax.nn.relu(n_step_target - q_value)
    priority_score = jax.lax.stop_gradient(
        jnp.abs(delta_one_step) +
        jnp.asarray(priority_eta, dtype=q_value.dtype) * positive_lower_bound +
        jnp.asarray(priority_epsilon, dtype=q_value.dtype))
    return delta_one_step, positive_lower_bound, priority_score


def one_sided_huber_loss(positive_gap, delta=1.0):
    """Huber penalty for a nonnegative lower-bound gap."""
    delta = jnp.asarray(delta, dtype=positive_gap.dtype)
    quadratic = jnp.minimum(positive_gap, delta)
    linear = positive_gap - quadratic
    return 0.5 * jnp.square(quadratic) + delta * linear


def select_maximization_distribution(one_step_distribution,
                                     n_step_distribution, support):
    """Selects the whole C51 candidate with the larger projected mean.

    The maximum of two scalar expectations is not an elementwise maximum of
    their categorical probabilities. Selecting the complete distribution
    preserves unit mass and makes its support mean exactly the chosen target.
    Ties deliberately fall back to the policy-consistent one-step candidate.
    """
    one_step_distribution = jax.lax.stop_gradient(
        jnp.asarray(one_step_distribution))
    n_step_distribution = jax.lax.stop_gradient(
        jnp.asarray(n_step_distribution))
    support = jnp.asarray(support)
    one_step_value = jnp.sum(one_step_distribution * support, axis=-1)
    n_step_value = jnp.sum(n_step_distribution * support, axis=-1)
    use_n_step = n_step_value > one_step_value
    selected_distribution = jnp.where(
        use_n_step[..., None], n_step_distribution, one_step_distribution)
    return (jax.lax.stop_gradient(selected_distribution),
            jax.lax.stop_gradient(one_step_value),
            jax.lax.stop_gradient(n_step_value),
            jax.lax.stop_gradient(use_n_step))


def maximization_td_signals(q_value, one_step_target, n_step_target,
                            priority_epsilon=1e-6):
    """Returns Y_max, its scalar TD error, and absolute-error PER score."""
    q_value = jnp.asarray(q_value)
    one_step_target = jax.lax.stop_gradient(jnp.asarray(one_step_target))
    n_step_target = jax.lax.stop_gradient(jnp.asarray(n_step_target))
    maximization_target = jnp.maximum(one_step_target, n_step_target)
    delta_max = maximization_target - q_value
    priority_score = jax.lax.stop_gradient(
        jnp.abs(delta_max) +
        jnp.asarray(priority_epsilon, dtype=q_value.dtype))
    return maximization_target, delta_max, priority_score


def distributional_td_signals(target_distribution,
                              predicted_probabilities,
                              support,
                              priority_epsilon=1e-6):
    """Returns the projected C51 expectation error and its PER score.

    The target is the same projected distribution used by the categorical
    critic loss.  Computing its support mean therefore preserves C51's support
    clipping semantics.  As in the maximization-target priority, the scalar
    score is stopped so replay prioritization cannot become an optimization
    objective.
    """
    target_distribution = jax.lax.stop_gradient(
        jnp.asarray(target_distribution))
    predicted_probabilities = jnp.asarray(predicted_probabilities)
    support = jnp.asarray(support)
    target_value = jax.lax.stop_gradient(
        jnp.sum(target_distribution * support, axis=-1))
    q_value = jnp.sum(predicted_probabilities * support, axis=-1)
    delta = target_value - q_value
    priority_score = jax.lax.stop_gradient(
        jnp.abs(delta) +
        jnp.asarray(priority_epsilon, dtype=q_value.dtype))
    return delta, priority_score


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


def priority_weighted_masked_mean(values, mask, loss_weights, eps=1e-6):
    """Masked mean whose trajectory rows carry their sampled-s0 weights.

    ``values`` and ``mask`` have shape ``[B, H]`` and ``loss_weights`` has
    shape ``[B]``.  The original unweighted mask denominator is retained, so
    unit weights exactly recover ``masked_mean`` and PER changes each sampled
    trajectory's contribution rather than self-normalizing the weights away.
    """
    values = jnp.asarray(values)
    mask = jnp.asarray(mask)
    loss_weights = jnp.asarray(loss_weights)
    if values.ndim != 2 or mask.shape != values.shape:
        raise ValueError("Masked trajectory values/mask must have shape [B, H].")
    if loss_weights.ndim != 1:
        raise ValueError("Replay-anchor loss weights must have shape [B].")
    if values.shape[0] != loss_weights.shape[0]:
        raise ValueError(
            "Masked trajectory/weight batch mismatch: {} versus {}."
            .format(values.shape[0], loss_weights.shape[0]))
    mask = mask.astype(jnp.float32)
    weighted_values = loss_weights[:, None] * values
    return jnp.sum(weighted_values * mask) / (jnp.sum(mask) + eps)


def masked_correlation(x, y, mask, eps=1e-6):
    """Pearson correlation over masked entries."""
    mask = mask.astype(jnp.float32)
    count = jnp.sum(mask) + eps
    x_mean = jnp.sum(x * mask) / count
    y_mean = jnp.sum(y * mask) / count
    x_var = jnp.sum(jnp.square(x - x_mean) * mask) / count
    y_var = jnp.sum(jnp.square(y - y_mean) * mask) / count
    cov = jnp.sum((x - x_mean) * (y - y_mean) * mask) / count
    return cov / jnp.sqrt(x_var * y_var + 1e-12)


def imagined_lambda_return(rewards, continues, values, discount, lamb):
    """Lambda returns over imagined trajectories.

  Follows r2dreamer's _lambda_return: the return from state t uses the
  reward and continue predicted at the *next* state (arrival semantics),
  ret_t = r_{t+1} + disc * c_{t+1} * ((1 - lambda) * V_{t+1}
                                       + lambda * ret_{t+1}).

  Args:
    rewards: [B, H + 1] rewards predicted at each imagined state.
    continues: [B, H + 1] continuation probabilities at each state.
    values: [B, H + 1] state values.
    discount: Scalar discount factor.
    lamb: Lambda mixing parameter.

  Returns:
    [B, H] lambda returns for states 0..H-1.
  """
    next_rewards = rewards[:, 1:]
    live = continues[:, 1:] * discount
    interm = next_rewards + live * (1.0 - lamb) * values[:, 1:]
    ret = values[:, -1]
    outputs = []
    for i in reversed(range(next_rewards.shape[1])):
        ret = interm[:, i] + live[:, i] * lamb * ret
        outputs.append(ret)
    return jnp.stack(outputs[::-1], axis=1)


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


def copy_optimizer_state(source, target, keys=("encoder", "transition_model")):
    """Copies selected per-module optimizer state through Optax containers.

    Optax wraps Adam state in nested ``MaskedState`` and namedtuple/tuple
    containers.  ``copy_params`` intentionally treats those containers as
    leaves, which meant the reset path silently reinitialized *all* optimizer
    state.  This routine descends through the full container structure while
    retaining the mapping-key semantics used for parameter modules.

    Scalar transform state (for example Adam's bias-correction counter) is
    copied when its containing transform has state for at least one selected
    module.  A transform may share that scalar across preserved and freshly
    reset modules; preserving it is the only consistent choice for the copied
    first and second moments. Callers should therefore select complete
    optimizer transforms. The reset path does this by preserving the complete
    encoder/transition-model and reward/continue transforms while restarting
    the Q/SPR and policy transforms.
    """
    keys = frozenset(keys)

    def contains_selected_module(tree):
        if isinstance(tree, (dict, collections.OrderedDict, FrozenDict)):
            return any((k in keys and not isinstance(v, optax.MaskedNode)) or
                       contains_selected_module(v)
                       for k, v in tree.items())
        if isinstance(tree, tuple):
            return any(contains_selected_module(v) for v in tree)
        if isinstance(tree, list):
            return any(contains_selected_module(v) for v in tree)
        return False

    def merge(old, fresh, preserve_scalar=False):
        if isinstance(old, (dict, collections.OrderedDict, FrozenDict)):
            merged = {}
            for k, old_value in old.items():
                if k in keys:
                    merged[k] = old_value
                else:
                    # Scalar transform state lives in tuple/namedtuple fields;
                    # mapping leaves are parameter-shaped moments and must stay
                    # fresh unless their module key was explicitly selected.
                    merged[k] = merge(old_value, fresh[k], False)
            if isinstance(fresh, FrozenDict):
                return FrozenDict(merged)
            if isinstance(fresh, collections.OrderedDict):
                return collections.OrderedDict(merged)
            return merged

        if isinstance(old, tuple):
            is_namedtuple = hasattr(old, "_fields")
            # A plain tuple at the root is the Optax chain: its transforms
            # must decide independently whether they own selected modules.
            # Tuples nested inside a selected MaskedState inherit that choice.
            keep_scalar = (preserve_scalar or contains_selected_module(old)
                           if is_namedtuple else preserve_scalar)
            values = [
                merge(old_value, fresh_value, keep_scalar)
                for old_value, fresh_value in zip(old, fresh)
            ]
            if is_namedtuple:
                return type(fresh)(*values)
            return type(fresh)(values)

        if isinstance(old, list):
            keep_scalar = preserve_scalar or contains_selected_module(old)
            return [
                merge(old_value, fresh_value, keep_scalar)
                for old_value, fresh_value in zip(old, fresh)
            ]

        return old if preserve_scalar else fresh

    return merge(source, target)


def imagined_reach_weights(continues, discount):
    """Returns reach weights for imagined states, with weight(state 0) = 1."""
    transition_discounts = continues[:, 1:] * discount
    return jnp.concatenate(
        [
            jnp.ones_like(continues[:, :1]),
            jnp.cumprod(transition_discounts, axis=1),
        ],
        axis=1,
    )


def validate_priority_cardinality(indices, priority_losses):
    """Raises if a replay sample does not have one loss per sampled index."""
    num_indices = int(np.size(indices))
    num_losses = int(np.size(priority_losses))
    if num_indices != num_losses:
        raise AssertionError(
            "Replay priority cardinality mismatch: "
            f"{num_indices} indices but {num_losses} per-example losses.")


def replay_loss_weights(priorities, retrace=False, mean_priority=None):
    """Builds legacy auxiliary weights and, when needed, beta=1 TD weights."""
    priorities = np.asarray(priorities, dtype=np.float32)
    if priorities.size == 0:
        raise ValueError("Replay priorities must not be empty.")
    if (not np.all(np.isfinite(priorities)) or
            np.any(priorities <= 0.0)):
        raise FloatingPointError(
            "Sampled replay priorities must be finite and positive.")

    auxiliary_weights = 1.0 / np.sqrt(priorities + 1e-10)
    auxiliary_weights /= np.max(auxiliary_weights)
    if not retrace:
        return auxiliary_weights, auxiliary_weights

    if (mean_priority is None or not np.isfinite(mean_priority) or
            mean_priority <= 0.0):
        raise FloatingPointError(
            "Replay mean priority must be finite and positive.")
    td_weights = np.asarray(
        mean_priority / (priorities + 1e-10), dtype=np.float32)
    if not np.all(np.isfinite(td_weights)):
        raise FloatingPointError("Retrace TD importance weights must be finite.")
    return auxiliary_weights, td_weights


def apply_replay_anchor_weights(per_anchor_losses, loss_weights):
    """Applies replay weights to losses defined at sampled real states.

    Both arguments have shape ``[B]``: there is exactly one loss and one
    priority-derived weight for each sampled replay state ``s_0``.
    """
    per_anchor_losses = jnp.asarray(per_anchor_losses)
    loss_weights = jnp.asarray(loss_weights)
    if per_anchor_losses.ndim != 1:
        raise ValueError("Replay-anchor losses must have shape [B].")
    if loss_weights.ndim != 1:
        raise ValueError("Replay-anchor loss weights must have shape [B].")
    if per_anchor_losses.shape[0] != loss_weights.shape[0]:
        raise ValueError(
            "Replay-anchor loss/weight batch mismatch: {} versus {}."
            .format(per_anchor_losses.shape[0], loss_weights.shape[0]))
    return loss_weights * per_anchor_losses


def validate_expected_one_step_backup(enabled, update_horizon,
                                      max_update_horizon):
    """Requires a genuinely one-step replay target when the option is on."""
    if not enabled:
        return

    update_horizon = int(update_horizon)
    max_update_horizon = (update_horizon if max_update_horizon is None else
                          int(max_update_horizon))
    if update_horizon != 1 or max_update_horizon != 1:
        raise ValueError(
            "expected_one_step_backup requires update_horizon=1 and "
            "max_update_horizon=1 (or None), got update_horizon={} and "
            "max_update_horizon={}.".format(update_horizon,
                                            max_update_horizon))


def validate_retrace(enabled, expected_one_step_backup, update_horizon,
                     max_update_horizon, retrace_horizon, retrace_lambda,
                     min_gamma, cycle_steps, distributional,
                     warmup_n_step_updates=0, warmup_n_step_horizon=10,
                     warmup_n_step_final_horizon=None,
                     warmup_min_gamma=None, warmup_gamma=None):
    """Validates the raw-transition and optional warmup Retrace contract."""
    warmup_n_step_updates = int(warmup_n_step_updates)
    warmup_n_step_horizon = int(warmup_n_step_horizon)
    warmup_n_step_final_horizon = (
        warmup_n_step_horizon if warmup_n_step_final_horizon is None else
        int(warmup_n_step_final_horizon))
    warmup_min_gamma = (None if warmup_min_gamma is None else
                        float(warmup_min_gamma))
    warmup_gamma = (None if warmup_gamma is None else float(warmup_gamma))
    if warmup_n_step_updates < 0:
        raise ValueError(
            "retrace_warmup_n_step_updates must be nonnegative, got {}."
            .format(warmup_n_step_updates))
    if not enabled:
        if (warmup_n_step_updates or
                warmup_n_step_final_horizon != warmup_n_step_horizon or
                warmup_min_gamma is not None or warmup_gamma is not None):
            raise ValueError(
                "Retrace warmup settings require retrace=True.")
        return
    if warmup_n_step_updates and warmup_n_step_horizon < 1:
        raise ValueError(
            "retrace_warmup_n_step_horizon must be positive, got {}."
            .format(warmup_n_step_horizon))
    if warmup_n_step_final_horizon < 1:
        raise ValueError(
            "retrace_warmup_n_step_final_horizon must be positive, got {}."
            .format(warmup_n_step_final_horizon))
    if warmup_n_step_final_horizon > warmup_n_step_horizon:
        raise ValueError(
            "retrace_warmup_n_step_final_horizon ({}) must not exceed the "
            "initial horizon ({}).".format(warmup_n_step_final_horizon,
                                           warmup_n_step_horizon))
    nondefault_warmup = (
        warmup_n_step_final_horizon != warmup_n_step_horizon or
        warmup_min_gamma is not None or warmup_gamma is not None)
    if nondefault_warmup and warmup_n_step_updates == 0:
        raise ValueError(
            "Non-default Retrace warmup settings require "
            "retrace_warmup_n_step_updates > 0.")
    if (warmup_min_gamma is not None and
            (not np.isfinite(warmup_min_gamma) or
             not 0.0 <= warmup_min_gamma < 1.0)):
        raise ValueError(
            "retrace_warmup_min_gamma must be finite and in [0, 1), got {}."
            .format(warmup_min_gamma))
    if (warmup_gamma is not None and
            (not np.isfinite(warmup_gamma) or
             not 0.0 <= warmup_gamma < 1.0)):
        raise ValueError(
            "retrace_warmup_gamma must be finite and in [0, 1), got {}."
            .format(warmup_gamma))
    if warmup_gamma is not None and warmup_min_gamma is not None:
        raise ValueError(
            "retrace_warmup_gamma and retrace_warmup_min_gamma are mutually "
            "exclusive; use the former for a fixed warmup discount or the "
            "latter for an annealed warmup discount.")

    update_horizon = int(update_horizon)
    max_update_horizon = (update_horizon if max_update_horizon is None else
                          int(max_update_horizon))
    retrace_horizon = int(retrace_horizon)
    retrace_lambda = float(retrace_lambda)
    if expected_one_step_backup:
        raise ValueError(
            "retrace and expected_one_step_backup are mutually exclusive.")
    if update_horizon != 1 or max_update_horizon != 1:
        raise ValueError(
            "retrace reads raw one-step replay rows and therefore requires "
            "update_horizon=max_update_horizon=1, got {} and {}.".format(
                update_horizon, max_update_horizon))
    if retrace_horizon < 1:
        raise ValueError(
            "retrace_horizon must be positive, got {}.".format(
                retrace_horizon))
    if not 0.0 <= retrace_lambda <= 1.0:
        raise ValueError(
            "retrace_lambda must be in [0, 1], got {}.".format(
                retrace_lambda))
    if min_gamma is not None and int(cycle_steps) > 1:
        raise ValueError(
            "retrace requires a fixed replay-TD gamma; set min_gamma=None "
            "instead of using the cyclic gamma schedule. Use "
            "retrace_warmup_min_gamma for a scheduled n-step warmup.")
    if not distributional:
        raise ValueError("retrace currently requires distributional C51.")


def validate_td_lower_bound(weight, horizon, priority_eta, priority_epsilon,
                            retrace, update_horizon, max_update_horizon,
                            distributional):
    """Validates the raw-row contract for one-step TD plus an n-step floor."""
    weight = float(weight)
    if not np.isfinite(weight) or weight < 0.0:
        raise ValueError(
            "td_lower_bound_weight must be finite and nonnegative, got {}."
            .format(weight))
    if weight == 0.0:
        return

    horizon = int(horizon)
    priority_eta = float(priority_eta)
    priority_epsilon = float(priority_epsilon)
    update_horizon = int(update_horizon)
    max_update_horizon = (update_horizon if max_update_horizon is None else
                          int(max_update_horizon))
    if retrace:
        raise ValueError("td_lower_bound and retrace are mutually exclusive.")
    if update_horizon != 1 or max_update_horizon != 1:
        raise ValueError(
            "td_lower_bound reconstructs its n-step signal from raw replay "
            "rows and requires update_horizon=max_update_horizon=1, got {} "
            "and {}.".format(update_horizon, max_update_horizon))
    if horizon < 2:
        raise ValueError(
            "td_lower_bound_horizon must be at least 2, got {}.".format(
                horizon))
    if not np.isfinite(priority_eta) or priority_eta < 0.0:
        raise ValueError(
            "td_lower_bound_priority_eta must be finite and nonnegative, got "
            "{}.".format(priority_eta))
    if not np.isfinite(priority_epsilon) or priority_epsilon <= 0.0:
        raise ValueError(
            "td_lower_bound_priority_epsilon must be finite and positive, got "
            "{}.".format(priority_epsilon))
    if not distributional:
        raise ValueError(
            "td_lower_bound currently requires the distributional C51 critic.")


def validate_td_maximization_target(enabled, horizon, priority_epsilon,
                                    retrace, td_lower_bound,
                                    expected_one_step_backup, update_horizon,
                                    max_update_horizon, distributional):
    """Validates one-step versus n-step C51 maximization-target settings."""
    if not enabled:
        return

    horizon = int(horizon)
    priority_epsilon = float(priority_epsilon)
    update_horizon = int(update_horizon)
    max_update_horizon = (update_horizon if max_update_horizon is None else
                          int(max_update_horizon))
    if retrace or td_lower_bound:
        raise ValueError(
            "td_maximization_target, Retrace, and td_lower_bound are mutually "
            "exclusive.")
    if not expected_one_step_backup:
        raise ValueError(
            "td_maximization_target requires expected_one_step_backup=True so "
            "both candidate targets bootstrap from the same policy value.")
    if update_horizon != 1 or max_update_horizon != 1:
        raise ValueError(
            "td_maximization_target reconstructs its n-step candidate from raw "
            "replay rows and requires update_horizon=max_update_horizon=1, got "
            "{} and {}.".format(update_horizon, max_update_horizon))
    if horizon < 2:
        raise ValueError(
            "td_maximization_horizon must be at least 2, got {}.".format(
                horizon))
    if not np.isfinite(priority_epsilon) or priority_epsilon <= 0.0:
        raise ValueError(
            "td_maximization_priority_epsilon must be finite and positive, got "
            "{}.".format(priority_epsilon))
    if not distributional:
        raise ValueError(
            "td_maximization_target currently requires distributional C51.")


def validate_delta_based_priority(enabled, priority_epsilon, retrace,
                                  td_lower_bound, td_maximization_target,
                                  distributional):
    """Validates the opt-in absolute scalar TD priority for ordinary C51."""
    if not enabled:
        return

    priority_epsilon = float(priority_epsilon)
    if retrace or td_lower_bound or td_maximization_target:
        raise ValueError(
            "delta_based_priority is only for ordinary C51 and is mutually "
            "exclusive with Retrace, td_lower_bound, and "
            "td_maximization_target.")
    if not np.isfinite(priority_epsilon) or priority_epsilon <= 0.0:
        raise ValueError(
            "delta_priority_epsilon must be finite and positive, got {}."
            .format(priority_epsilon))
    if not distributional:
        raise ValueError(
            "delta_based_priority currently requires distributional C51.")


def validate_retrace_warmup_priority_settings(
        delta_based_priority, reset_on_target_switch, priority_epsilon,
        retrace, warmup_n_step_updates, distributional):
    """Validates phase-local priorities for an n-step Retrace warmup."""
    delta_based_priority = bool(delta_based_priority)
    reset_on_target_switch = bool(reset_on_target_switch)
    if not delta_based_priority and not reset_on_target_switch:
        return
    if not retrace or int(warmup_n_step_updates) <= 0:
        raise ValueError(
            "Retrace warmup priority settings require retrace=True and "
            "retrace_warmup_n_step_updates > 0.")
    if delta_based_priority and not reset_on_target_switch:
        raise ValueError(
            "retrace_warmup_delta_based_priority requires "
            "retrace_reset_priorities_on_target_switch=True so delta and "
            "Retrace priorities cannot mix.")
    if delta_based_priority:
        priority_epsilon = float(priority_epsilon)
        if not np.isfinite(priority_epsilon) or priority_epsilon <= 0.0:
            raise ValueError(
                "delta_priority_epsilon must be finite and positive, got {}."
                .format(priority_epsilon))
        if not distributional:
            raise ValueError(
                "Retrace warmup delta priority requires distributional C51.")


def td_backup_parameter_sets(online_params, target_params, use_target_backups,
                             double_dqn):
    """Selects value-evaluation and action-selection params for TD backups."""
    value_params = target_params if use_target_backups else online_params
    policy_params = online_params if double_dqn else value_params
    return value_params, policy_params


def behavior_parameter_set(online_params, target_params,
                           target_action_selection):
    """Selects the network used to sample environment actions."""
    return target_params if target_action_selection else online_params


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
        "optimizer_keys_to_copy",
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
    optimizer_keys_to_copy,
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
    keys_to_copy: Parameter keys to copy over without resetting.
    optimizer_keys_to_copy: Module keys whose optimizer state is retained.

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

    fresh_optimizer_state = optimizer.init(online_params)
    optimizer_state = copy_optimizer_state(optimizer_state,
                                           fresh_optimizer_state,
                                           keys=optimizer_keys_to_copy)

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
])
def select_action(
    network_def,
    params,
    state,
    rng,
):
    rng, key = jax.random.split(rng)
    state = spr_networks.process_inputs(state,
                                        rng=key,
                                        data_augmentation=False,
                                        dtype=jnp.float32)

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
    # On-policy categorical sampling for both training and evaluation.
    return rng, samples, jax.nn.softmax(logits)


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
    'expected_one_step_backup',
    'retrace',
    'retrace_warmup_n_step',
    'spr_horizon',
    'retrace_horizon',
    'retrace_warmup_n_step_horizon',
    'retrace_warmup_delta_based_priority',
    'td_lower_bound',
    'td_lower_bound_horizon',
    'td_maximization_target',
    'td_maximization_horizon',
    'delta_based_priority',
    'match_online_target_rngs',
    'target_eval_mode',
    'reward_weight',
    'continue_weight',
    'reward_readout',
    'continue_readout',
    'reward_grad_surgery',
    'imag_horizon',
]


def train(
    network_def,  # 0, static
    online_params,  # 1
    target_params,  # 2
    optimizer,  # 3, static
    optimizer_state,  # 4
    raw_states,  # 5
    actions,  # 6
    raw_next_states,  # 7
    rewards,  # 8
    terminals,  # 9
    same_traj_mask,  # 10
    loss_weights,  # 11
    td_loss_weights,
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
    expected_one_step_backup,  # static
    retrace,  # static
    retrace_warmup_n_step,  # static
    spr_horizon,  # static
    retrace_horizon,  # static
    retrace_warmup_n_step_horizon,  # static
    retrace_warmup_delta_based_priority,  # static
    retrace_lambda,
    td_lower_bound,  # static
    td_lower_bound_horizon,  # static
    td_lower_bound_weight,
    td_lower_bound_priority_eta,
    td_lower_bound_priority_epsilon,
    td_maximization_target,  # static
    td_maximization_horizon,  # static
    td_maximization_priority_epsilon,
    delta_based_priority,  # static
    delta_priority_epsilon,
    target_update_tau,
    target_update_every,
    step,
    match_online_target_rngs,  # static
    target_eval_mode,  # static
    #ent_targ,
    x_ent_coef,
    per_step_rewards,
    behavior_probabilities,
    reward_weight,  # static
    continue_weight,  # static
    reward_readout,  # static
    continue_readout,  # static
    reward_grad_surgery,  # static
    imag_horizon,  # static
    imag_actor_mult,
    imag_value_mult,
    imag_discount,
    imag_lambda,
    imag_entropy_coef,
    imag_value_trust,
    return_ema,
):

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
            rng,
            step,
            return_ema,
        ) = state
        (
            raw_states,
            actions,
            raw_next_states,
            rewards,
            terminals,
            same_traj_mask,
            loss_weights,
            td_loss_weights,
            cumulative_gamma,
            per_step_rewards,
            behavior_probabilities,
        ) = inputs
        # Multi-step replay targets can request a longer real sequence than
        # SPR/imagination. Keep representation and world-model learning at the
        # configured SPR depth.
        model_raw_states = raw_states[:, :spr_horizon + 1]
        model_actions = actions[:, :spr_horizon]
        model_same_traj = same_traj_mask[:, :spr_horizon + 1]
        model_step_rewards = per_step_rewards[:, :spr_horizon]

        # World-model targets, aligned to arrival semantics: the feature of
        # state s_{k+1} (column k below) predicts reward r_k and whether the
        # episode continues at s_{k+1}.
        continue_targets = model_same_traj[:, 1:].astype(jnp.float32)
        model_mask = jnp.concatenate(
            [
                jnp.ones_like(continue_targets[:, :1]),
                continue_targets[:, :-1],
            ],
            axis=1,
        )
        model_rewards = model_step_rewards.astype(jnp.float32)

        same_traj_mask = model_same_traj[:, 1:]
        root_rewards = rewards[:, 0]
        root_terminals = terminals[:, 0]
        root_discount = cumulative_gamma[:, 0]

        rng, rng1, rng2 = jax.random.split(rng, num=3)
        states = spr_networks.process_inputs(
            model_raw_states,
            rng=rng1,
            data_augmentation=data_augmentation,
            dtype=dtype)
        if retrace:
            # The behavior policy acts on unaugmented observations. Evaluate
            # the current policy on the same observation contract for an exact
            # pi/mu ratio, while retaining DrQ augmentation for target-C51.
            retrace_policy_states = spr_networks.process_inputs(
                raw_next_states[:, :retrace_horizon],
                data_augmentation=False,
                dtype=dtype,
            )
            retrace_value_states = spr_networks.process_inputs(
                raw_next_states[:, :retrace_horizon],
                rng=rng2,
                data_augmentation=data_augmentation,
                dtype=dtype,
            )
            next_states = retrace_value_states[:, 0]
        elif retrace_warmup_n_step:
            # The warmup changes only the critic target: use the same
            # unaugmented online-policy / augmented target-critic endpoint
            # contract as Retrace's final S_H branch.
            retrace_warmup_policy_states = spr_networks.process_inputs(
                raw_next_states[:, retrace_warmup_n_step_horizon - 1],
                data_augmentation=False,
                dtype=dtype,
            )
            retrace_warmup_value_states = spr_networks.process_inputs(
                raw_next_states[:, retrace_warmup_n_step_horizon - 1],
                rng=rng2,
                data_augmentation=data_augmentation,
                dtype=dtype,
            )
            next_states = retrace_warmup_value_states
        elif td_lower_bound or td_maximization_target:
            # Both modes need aligned one-step and H-step endpoints from the
            # same raw replay anchor. Give the endpoints independent crops.
            multi_step_horizon = (td_lower_bound_horizon if td_lower_bound else
                                  td_maximization_horizon)
            next_states = spr_networks.process_inputs(
                raw_next_states[:, 0],
                rng=rng2,
                data_augmentation=data_augmentation,
                dtype=dtype,
            )
            multi_step_next_states = spr_networks.process_inputs(
                raw_next_states[:, multi_step_horizon - 1],
                rng=jax.random.fold_in(rng2, 1),
                data_augmentation=data_augmentation,
                dtype=dtype,
            )
        else:
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
        use_spr = (spr_weight > 0 or reward_weight > 0 or continue_weight > 0
                   or imag_horizon > 0)

        # Double DQN uses online parameters for action selection and the
        # configured backup network for value evaluation.  With Double DQN
        # disabled, both sides use the configured value-backup parameters.
        backup_params, action_selection_params = td_backup_parameter_sets(
            online_params, target_params, use_target_backups, double_dqn)

        def policy_backup(state, action_sample_key):
            return network_def.apply(
                action_selection_params,
                state,
                rngs={"action_sample": action_sample_key},
                method=network_def.get_policy,
            )

        def current_policy(state, action_sample_key):
            # Retrace and its n-step warmup evaluate the current target
            # policy explicitly. This is independent of Double-DQN's
            # action-selection routing.
            return network_def.apply(
                online_params,
                state,
                rngs={"action_sample": action_sample_key},
                method=network_def.get_policy,
            )

        def q_backup(state):
            return network_def.apply(
                backup_params,
                state,
                support=support,
                eval_mode=target_eval_mode,
            )

        def encode_project(state):
            return network_def.apply(
                target_params,
                state,
                eval_mode=True,
                method=network_def.encode_project_with_latent,
            )

        def loss_fn(
            params,
            target,
            spr_targets,
            future_latents,
            one_step_target_value,
            n_step_target_value,
            loss_multipliers,
            td_loss_multipliers,
            key,
            imag_keys,
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
                                                    model_actions, use_spr)
            spr_predictions = x.latent
            q_logits = jnp.squeeze(x.logits)
            chosen_action_logits = q_logits[jnp.arange(q_logits.shape[0]),
                                            actions[:, 0]]
            dqn_loss = jax.vmap(softmax_cross_entropy_loss_with_logits)(
                target, chosen_action_logits)
            lower_bound_loss = jnp.zeros_like(dqn_loss)
            positive_lower_bound = jnp.zeros_like(dqn_loss)
            delta_one_step = jnp.zeros_like(dqn_loss)
            maximization_target_value = jnp.zeros_like(dqn_loss)
            delta_maximization = jnp.zeros_like(dqn_loss)
            maximization_uses_n_step = jnp.zeros_like(dqn_loss)
            if retrace:
                predicted_probabilities = jax.nn.softmax(chosen_action_logits)
                retrace_tv = 0.5 * jnp.sum(
                    jnp.abs(target - predicted_probabilities), axis=-1)
                priority_loss = categorical_retrace_priority(
                    target, predicted_probabilities)
                td_error = retrace_tv
            elif (retrace_warmup_n_step and
                  retrace_warmup_delta_based_priority):
                predicted_probabilities = jax.nn.softmax(
                    chosen_action_logits)
                delta_priority, priority_loss = distributional_td_signals(
                    target,
                    predicted_probabilities,
                    support[None, :],
                    delta_priority_epsilon,
                )
                td_error = jnp.abs(delta_priority)
            elif retrace_warmup_n_step:
                # Preserve the original row-38 warmup priority by default.
                predicted_probabilities = jax.nn.softmax(chosen_action_logits)
                retrace_tv = 0.5 * jnp.sum(
                    jnp.abs(target - predicted_probabilities), axis=-1)
                priority_loss = categorical_retrace_priority(
                    target, predicted_probabilities)
                td_error = retrace_tv
            elif td_lower_bound:
                predicted_probabilities = jax.nn.softmax(chosen_action_logits)
                q_value = jnp.sum(
                    predicted_probabilities * support[None, :], axis=-1)
                (delta_one_step, positive_lower_bound,
                 priority_loss) = lower_bound_td_signals(
                     q_value,
                     one_step_target_value,
                     n_step_target_value,
                     td_lower_bound_priority_eta,
                     td_lower_bound_priority_epsilon,
                 )
                lower_bound_loss = one_sided_huber_loss(
                    positive_lower_bound)
                td_error = jnp.abs(delta_one_step)
            elif td_maximization_target:
                predicted_probabilities = jax.nn.softmax(chosen_action_logits)
                q_value = jnp.sum(
                    predicted_probabilities * support[None, :], axis=-1)
                (maximization_target_value, delta_maximization,
                 priority_loss) = maximization_td_signals(
                     q_value,
                     one_step_target_value,
                     n_step_target_value,
                     td_maximization_priority_epsilon,
                 )
                maximization_uses_n_step = (
                    n_step_target_value > one_step_target_value).astype(
                        jnp.float32)
                td_error = jnp.abs(delta_maximization)
            elif delta_based_priority:
                predicted_probabilities = jax.nn.softmax(
                    chosen_action_logits)
                delta_priority, priority_loss = distributional_td_signals(
                    target,
                    predicted_probabilities,
                    support[None, :],
                    delta_priority_epsilon,
                )
                td_error = jnp.abs(delta_priority)
            else:
                priority_loss = dqn_loss
                td_error = dqn_loss + jnp.nan_to_num(
                    target * jnp.log(target)).sum(-1)

            spr_predictions = spr_predictions.transpose(1, 0, 2)
            spr_predictions = spr_networks.split_spr_branches(
                spr_predictions, network_def.hidden_dim)

            spr_predictions = spr_predictions / jnp.linalg.norm(
                spr_predictions, 2, -1, keepdims=True)

            spr_targets = spr_networks.split_spr_branches(
                spr_targets, network_def.hidden_dim)
            spr_targets = spr_targets / jnp.linalg.norm(
                spr_targets, 2, -1, keepdims=True)
            spr_loss = jnp.power(spr_predictions - spr_targets,
                                 2).sum((-1, -2))
            #logging.info("spr_loss.shape: {}".format(spr_loss.shape))
            spr_loss = (spr_loss * same_traj_mask.transpose(1, 0)).mean(0) * .5
            #logging.info("spr_loss.shape: {}".format(spr_loss.shape))
            #exit(0)
            # Keep full beta=1 correction for the complete hybrid experiment,
            # including its n-step warmup. SPR (and weighted
            # actor/imagination losses below) retain historical beta=0.5.
            if retrace or retrace_warmup_n_step:
                loss = (td_loss_multipliers * dqn_loss +
                        loss_multipliers * spr_weight * spr_loss)
            elif td_lower_bound:
                loss = apply_replay_anchor_weights(
                    dqn_loss +
                    td_lower_bound_weight * lower_bound_loss +
                    spr_weight * spr_loss,
                    loss_multipliers)
            else:
                # Preserve the historical operation order for seed-level
                # reproducibility when Retrace is disabled.
                loss = apply_replay_anchor_weights(
                    dqn_loss + spr_weight * spr_loss,
                    loss_multipliers)

            mean_loss = jnp.mean(loss)

            # === World-model heads: reward and continue prediction ===
            aux_model = {}
            model_loss = jnp.asarray(0.0, dtype=jnp.float32)
            if reward_weight > 0 or continue_weight > 0:
                # By default rollout features keep gradients so reward errors
                # also shape the encoder/transition model (r2dreamer-style
                # grounding). reward_readout makes the heads strict readouts:
                # SPR becomes the transition model's sole supervisor. The
                # real-frame features come from the target encoder and only
                # train the heads either way.
                reward_reps = (jax.lax.stop_gradient(x.rollout_reps)
                               if reward_readout else x.rollout_reps)
                # continue_readout detaches ONLY the continue head: its BCE
                # is dense on every game (life-loss terminals), so it grounds
                # the trunk everywhere even when rewards are sparse. reward
                # stays per reward_readout.
                continue_reps = (jax.lax.stop_gradient(x.rollout_reps)
                                 if (reward_readout or continue_readout) else
                                 x.rollout_reps)
                real_reps = jax.lax.stop_gradient(future_latents)

                def reward_fn(feature):
                    return network_def.apply(
                        params,
                        feature,
                        method=network_def.reward_from_feature)

                def continue_fn(feature):
                    return network_def.apply(
                        params,
                        feature,
                        method=network_def.continue_from_feature)

                pred_reward_roll = jax.vmap(jax.vmap(reward_fn))(reward_reps)
                pred_reward_real = jax.vmap(jax.vmap(reward_fn))(real_reps)
                # Real frames past a terminal belong to the next episode.
                real_mask = model_mask * continue_targets
                reward_loss = (priority_weighted_masked_mean(
                    jnp.square(pred_reward_roll - model_rewards),
                    model_mask,
                    loss_multipliers) + priority_weighted_masked_mean(
                        jnp.square(pred_reward_real - model_rewards),
                        real_mask,
                        loss_multipliers))
                continue_logits = jax.vmap(
                    jax.vmap(continue_fn))(continue_reps)
                continue_loss = priority_weighted_masked_mean(
                    sigmoid_binary_cross_entropy(continue_logits,
                                                 continue_targets),
                    model_mask,
                    loss_multipliers)
                model_loss = (reward_weight * reward_loss +
                              continue_weight * continue_loss)
                aux_model.update({
                    "RewardLoss":
                        reward_loss,
                    "ContinueLoss":
                        continue_loss,
                    "RewardCorr":
                        masked_correlation(pred_reward_roll, model_rewards,
                                           model_mask),
                })

            # === Imagination: on-policy actor(-critic) over imagined rollouts ===
            imag_metrics = {}
            imag_actor_loss = jnp.asarray(0.0, dtype=jnp.float32)
            imag_value_loss = jnp.asarray(0.0, dtype=jnp.float32)
            new_return_ema = return_ema
            if imag_horizon > 0:

                def imagine_one(latent, imagine_key):
                    return network_def.apply(
                        params,
                        latent,
                        imag_horizon,
                        rngs={"action_sample": imagine_key},
                        method=network_def.imagine_from_latent,
                    )

                imagined = jax.vmap(imagine_one)(x.spatial_latent, imag_keys)

                def value_fn(feature):
                    return network_def.apply(
                        backup_params,
                        feature,
                        support,
                        method=network_def.q_values_from_feature)

                # V(z) = sum_a pi(a|z) Q_target(z, a); grounded by real TD.
                imag_q_target = jax.vmap(jax.vmap(value_fn))(
                    imagined['features'])
                imag_probs = jax.lax.stop_gradient(imagined['probs'])
                imag_values = jnp.sum(imag_probs * imag_q_target, -1)

                # Starts are real replay states: force continue = 1 at step 0.
                imag_continues = jax.lax.stop_gradient(
                    jnp.concatenate(
                        [
                            jnp.ones_like(imagined['continues'][:, :1]),
                            imagined['continues'][:, 1:],
                        ],
                        axis=1,
                    ))
                imag_rewards = jax.lax.stop_gradient(imagined['rewards'])
                ret = jax.lax.stop_gradient(
                    imagined_lambda_return(imag_rewards, imag_continues,
                                           imag_values, imag_discount,
                                           imag_lambda))
                weight = jax.lax.stop_gradient(
                    imagined_reach_weights(imag_continues, imag_discount))

                percentiles = jnp.percentile(ret, jnp.asarray([5.0, 95.0]))
                new_return_ema = jnp.where(
                    imag_actor_mult > 0,
                    0.01 * percentiles + 0.99 * return_ema,
                    return_ema,
                )
                scale = jnp.maximum(new_return_ema[1] - new_return_ema[0],
                                    1.0)
                adv = jax.lax.stop_gradient(
                    (ret - imag_values[:, :-1]) / scale)

                # Carries the same PER weights as the replay actor loss (the
                # rollouts start from those states), so imag_actor_mult = 1
                # really does weight the two actor losses equally.
                imag_actor_loss = jnp.mean(
                    loss_multipliers[:, None] * weight[:, :-1] *
                    -(imagined['log_probs'][:, :-1] * adv +
                      imag_entropy_coef * imagined['entropies'][:, :-1]))

                def q_logits_fn(feature):
                    return network_def.apply(
                        params,
                        feature,
                        method=network_def.q_logits_from_feature)

                imag_q_logits = jax.vmap(jax.vmap(q_logits_fn))(
                    imagined['features'][:, :-1])
                chosen_imag_logits = jnp.squeeze(
                    jnp.take_along_axis(
                        imag_q_logits,
                        imagined['actions'][:, :-1, None, None].astype(
                            jnp.int32),
                        axis=2,
                    ), 2)
                # Trust region for the imagined critic targets: bound the
                # lambda-return within imag_value_trust * scale of the target
                # critic's estimate for the taken action (inf = off). The
                # actor's advantages keep the raw return.
                q_choice = jnp.squeeze(
                    jnp.take_along_axis(
                        imag_q_target[:, :-1],
                        imagined['actions'][:, :-1, None].astype(jnp.int32),
                        axis=2), 2)
                trust_band = imag_value_trust * scale
                value_ret = jnp.where(
                    jnp.isfinite(trust_band),
                    q_choice + jnp.clip(ret - q_choice, -trust_band,
                                        trust_band),
                    ret)
                trust_clip_frac = jnp.mean(
                    jnp.where(
                        jnp.isfinite(trust_band),
                        (jnp.abs(ret - q_choice) >= trust_band).astype(
                            jnp.float32), 0.0))

                imag_target_dist = jax.vmap(
                    jax.vmap(lambda r: project_distribution(
                        r[None], jnp.ones(1), support)))(value_ret)
                # Same PER weights as the replay critic loss -- the imagined
                # targets train the same Q head from the same start states.
                imag_value_loss = jnp.mean(
                    loss_multipliers[:, None] * weight[:, :-1] * -jnp.sum(
                        imag_target_dist *
                        jax.nn.log_softmax(chosen_imag_logits), -1))

                imag_metrics.update({
                    "ImagActorLoss": imag_actor_loss,
                    "ImagValueLoss": imag_value_loss,
                    "ImagRet": jnp.mean(ret),
                    "ImagValue": jnp.mean(imag_values),
                    "ImagReward": jnp.mean(imag_rewards),
                    "ImagContinue": jnp.mean(imag_continues),
                    "ImagEntropy": jnp.mean(imagined['entropies']),
                    "ImagTrustClip": trust_clip_frac,
                })

            policy_out = jax.vmap(policy_loss, in_axes=0,
                                  axis_name="batch")(x.q_values, logits, key)
            replay_actor_loss = jnp.mean(
                apply_replay_anchor_weights(policy_out[0], loss_multipliers))
            total_loss = (mean_loss +
                          replay_actor_loss +
                          model_loss + imag_actor_mult * imag_actor_loss +
                          imag_value_mult * imag_value_loss)
            aux_losses = {
                "TotalLoss": total_loss,
                "ModelLoss": model_loss,
                "DQNLoss": jnp.mean(dqn_loss),
                # Keep the mode-specific per-example priority score until
                # after the grouped minibatch scan so every sampled replay
                # index receives a corresponding priority update. DQNLoss
                # remains the scalar logging metric above.
                "PriorityLoss": priority_loss,
                "TD Error": jnp.mean(td_error),
                "SPRLoss": jnp.mean(spr_loss),
                "ent": jnp.mean(policy_out[1]),
                "ReturnEMAState": new_return_ema,
            }
            aux_losses.update(aux_model)
            aux_losses.update(imag_metrics)
            if retrace:
                aux_losses["RetraceNegativeMass"] = jnp.mean(
                    jnp.sum(jnp.maximum(-target, 0.0), axis=-1))
                aux_losses["RetraceTargetMass"] = jnp.mean(
                    jnp.sum(target, axis=-1))
            if td_lower_bound:
                aux_losses["LowerBoundLoss"] = jnp.mean(lower_bound_loss)
                aux_losses["LowerBoundGap"] = jnp.mean(positive_lower_bound)
                aux_losses["LowerBoundActive"] = jnp.mean(
                    (positive_lower_bound > 0.0).astype(jnp.float32))
                aux_losses["OneStepAbsTD"] = jnp.mean(
                    jnp.abs(delta_one_step))
                aux_losses["NStepTarget"] = jnp.mean(n_step_target_value)
            if td_maximization_target:
                aux_losses["MaxTargetOneStep"] = jnp.mean(
                    one_step_target_value)
                aux_losses["MaxTargetNStep"] = jnp.mean(n_step_target_value)
                aux_losses["MaxTargetValue"] = jnp.mean(
                    maximization_target_value)
                aux_losses["MaxTargetLift"] = jnp.mean(
                    jax.nn.relu(n_step_target_value - one_step_target_value))
                aux_losses["MaxTargetNStepSelected"] = jnp.mean(
                    maximization_uses_n_step)
                aux_losses["MaxTargetAbsTD"] = jnp.mean(
                    jnp.abs(delta_maximization))
            return total_loss, (aux_losses)

        # Use the weighted mean loss for gradient computation.
        one_step_target_value = jnp.zeros_like(root_rewards)
        n_step_target_value = jnp.zeros_like(root_rewards)
        if retrace:
            retrace_keys = jax.random.split(
                rng1, states.shape[0] * retrace_horizon).reshape(
                    states.shape[0], retrace_horizon, 2)
            retrace_policy_logits, _ = jax.vmap(
                jax.vmap(current_policy, in_axes=(0, 0), axis_name="time"),
                in_axes=(0, 0), axis_name="batch")(
                    retrace_policy_states, retrace_keys)
            retrace_policy_probabilities = jax.nn.softmax(
                retrace_policy_logits, axis=-1)
            retrace_target_output = jax.vmap(
                jax.vmap(q_backup, in_axes=0, axis_name="time"),
                in_axes=0, axis_name="batch")(retrace_value_states)
            retrace_target_probabilities = retrace_target_output.probabilities
            target = jax.vmap(
                categorical_retrace,
                in_axes=(0, 0, 0, 0, 0, 0, None, 0, None),
                axis_name="batch")(
                    retrace_policy_probabilities,
                    retrace_target_probabilities,
                    actions[:, 1:retrace_horizon],
                    behavior_probabilities[:, 1:retrace_horizon],
                    per_step_rewards[:, :retrace_horizon],
                    terminals[:, :retrace_horizon],
                    support,
                    root_discount,
                    retrace_lambda,
                )
        elif retrace_warmup_n_step:
            warmup_keys = jax.random.split(rng1, states.shape[0])
            warmup_policy_logits, _ = jax.vmap(
                current_policy, in_axes=(0, 0), axis_name="batch")(
                    retrace_warmup_policy_states, warmup_keys)
            warmup_policy_probabilities = jax.nn.softmax(
                warmup_policy_logits, axis=-1)
            warmup_target_output = jax.vmap(
                q_backup, in_axes=0, axis_name="batch")(
                    retrace_warmup_value_states)
            target = jax.vmap(
                categorical_n_step_target,
                in_axes=(0, 0, 0, 0, None, 0),
                axis_name="batch")(
                    warmup_policy_probabilities,
                    warmup_target_output.probabilities,
                    per_step_rewards[:, :retrace_warmup_n_step_horizon],
                    terminals[:, :retrace_warmup_n_step_horizon],
                    support,
                    root_discount,
                )
        elif td_lower_bound or td_maximization_target:
            # Primary one-step C51 target: retain the configured current-code
            # behavior (exact policy mixture when expected_one_step_backup is
            # enabled, otherwise one sampled bootstrap action).
            one_step_distribution = jax.vmap(
                target_output,
                in_axes=(None, None, 0, 0, 0, None, 0, 0, None),
                axis_name="batch")(
                    policy_backup,
                    q_backup,
                    next_states,
                    root_rewards,
                    root_terminals,
                    support,
                    root_discount,
                    target_rng,
                    expected_one_step_backup,
                )

            # Reconstruct the uncorrected H-step reward prefix from raw rows.
            # Use the same configured endpoint backup as the one-step target.
            # In the expected-backup ablations this exact policy mixture avoids
            # adding sampled-action variance before a one-sided target is used.
            multi_step_rewards = jax.vmap(
                n_step_lower_bound_target,
                in_axes=(0, 0, None, 0),
                axis_name="batch")(
                    per_step_rewards[:, :multi_step_horizon],
                    terminals[:, :multi_step_horizon],
                    jnp.asarray(0.0, dtype=per_step_rewards.dtype),
                    root_discount,
                )
            multi_step_terminals = jnp.max(
                terminals[:, :multi_step_horizon], axis=-1)
            multi_step_discount = jnp.power(
                root_discount, multi_step_horizon)
            multi_step_rng = jax.random.split(
                jax.random.fold_in(rng1, 1), states.shape[0])
            n_step_distribution = jax.vmap(
                target_output,
                in_axes=(None, None, 0, 0, 0, None, 0, 0, None),
                axis_name="batch")(
                    policy_backup,
                    q_backup,
                    multi_step_next_states,
                    multi_step_rewards,
                    multi_step_terminals,
                    support,
                    multi_step_discount,
                    multi_step_rng,
                    expected_one_step_backup,
                )
            if td_maximization_target:
                (target, one_step_target_value, n_step_target_value,
                 _) = select_maximization_distribution(
                     one_step_distribution, n_step_distribution, support)
            else:
                target = one_step_distribution
                one_step_target_value = jax.lax.stop_gradient(
                    jnp.sum(target * support[None, :], axis=-1))
                n_step_target_value = jax.lax.stop_gradient(
                    jnp.sum(n_step_distribution * support[None, :], axis=-1))
        else:
            target = jax.vmap(
                target_output,
                in_axes=(None, None, 0, 0, 0, None, 0, 0, None),
                axis_name="batch")(
                    policy_backup,
                    q_backup,
                    next_states,
                    root_rewards,
                    root_terminals,
                    support,
                    root_discount,
                    target_rng,
                    expected_one_step_backup,
                )

        future_states = states[:, 1:spr_horizon + 1]
        spr_targets, future_latents = jax.vmap(jax.vmap(encode_project,
                                                        in_axes=0,
                                                        axis_name="time"),
                                               in_axes=0,
                                               axis_name="batch")(future_states)
        spr_targets = spr_targets.transpose(1, 0, 2)

        n_samples = current_state.shape[0]
        splits = jax.random.split(rng2, 2 * n_samples + 1)
        rng2 = splits[0]
        key = splits[1:n_samples + 1]
        imag_keys = splits[n_samples + 1:]
        loss_args = (target, spr_targets, future_latents,
                     one_step_target_value, n_step_target_value, loss_weights,
                     td_loss_weights, key, imag_keys)
        if reward_grad_surgery:
            # Two backward passes: main = everything except the grounding
            # losses, model = reward/continue only. The component of the
            # grounding gradient that conflicts with the main direction is
            # projected out (PCGrad applied to one loss), so grounding acts
            # where it agrees with TD+SPR and is disarmed where it fights.
            def main_loss_fn(params, *args):
                total, aux = loss_fn(params, *args)
                return total - aux["ModelLoss"], aux

            def model_loss_fn(params, *args):
                total, aux = loss_fn(params, *args)
                return aux["ModelLoss"], aux

            (_, aux_losses), g_main = jax.value_and_grad(
                main_loss_fn, has_aux=True)(online_params, *loss_args)
            (_, _), g_model = jax.value_and_grad(
                model_loss_fn, has_aux=True)(online_params, *loss_args)
            # v2 (2026-07-23): reductions AND the correction are restricted
            # to the shared trunk (encoder + transition_model) -- the module
            # set the intervention was always meant to act on. v1 flattened
            # the entire tree, which (a) diluted the cosine and undersized
            # the trunk correction with Q/policy/head gradient mass that
            # carries no model-loss gradient, and (b) leaked the correction
            # into those modules as a (1+|coef|) rescale of the main
            # gradient on conflict steps. Pre-v2 GroundCos logs are the
            # diluted global quantity; do not compare across the change.
            trunk = ("encoder", "transition_model")
            tdot = lambda a, b: jax.tree_util.tree_reduce(
                lambda u, v: u + v, jax.tree_util.tree_map(jnp.vdot, a, b))
            stats = {
                m: (tdot(g_model["params"][m], g_main["params"][m]),
                    tdot(g_main["params"][m], g_main["params"][m]),
                    tdot(g_model["params"][m], g_model["params"][m]))
                for m in trunk
            }
            dot = sum(s[0] for s in stats.values())
            nm = sum(s[1] for s in stats.values())
            mm = sum(s[2] for s in stats.values())
            coef = jnp.where(dot < 0, dot / (nm + 1e-12), 0.0)
            # Corrected combination on trunk leaves, plain sum elsewhere;
            # copy_params grafts the corrected trunk subtrees over the
            # plain-sum tree, preserving structure.
            corrected = jax.tree_util.tree_map(
                lambda gm, gd: gm + gd - coef * gm, g_main, g_model)
            plain = jax.tree_util.tree_map(lambda gm, gd: gm + gd, g_main,
                                           g_model)
            grad = copy_params(corrected, plain, keys=trunk)
            if isinstance(g_main, FrozenDict):
                grad = FrozenDict(grad)
            eps = 1e-12
            aux_losses["GroundCos"] = dot * jax.lax.rsqrt(nm * mm + eps)
            aux_losses["GroundCosEnc"] = stats["encoder"][0] * jax.lax.rsqrt(
                stats["encoder"][1] * stats["encoder"][2] + eps)
            aux_losses["GroundCosTM"] = (
                stats["transition_model"][0] *
                jax.lax.rsqrt(stats["transition_model"][1] *
                              stats["transition_model"][2] + eps))
            aux_losses["GroundNormRatio"] = jnp.sqrt(mm / (nm + eps))
        else:
            # Get the unweighted loss without taking its mean for updating
            # priorities.
            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            (_, aux_losses), grad = grad_fn(online_params, *loss_args)
        new_return_ema = aux_losses.pop("ReturnEMAState")

        updates, new_optimizer_state = optimizer.update(grad,
                                                        optimizer_state,
                                                        params=online_params)
        new_online_params = optax.apply_updates(online_params, updates)

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
                rng2,
                step + 1,
                new_return_ema,
            ),
            aux_losses,
        )

    init_state = (
        online_params,
        target_params,
        optimizer_state,
        rng,
        step,
        return_ema,
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
        td_loss_weights.reshape(num_batches, batch_size,
                                *td_loss_weights.shape[1:]),
        cumulative_gamma.reshape(num_batches, batch_size,
                                 *cumulative_gamma.shape[1:]),
        per_step_rewards.reshape(num_batches, batch_size,
                                 *per_step_rewards.shape[1:]),
        behavior_probabilities.reshape(
            num_batches, batch_size, *behavior_probabilities.shape[1:]),
    )

    (
        (
            online_params,
            target_params,
            optimizer_state,
            rng,
            step,
            return_ema,
        ),
        aux_losses,
    ) = jax.lax.scan(train_one_batch, init_state, inputs)

    return (
        online_params,
        target_params,
        optimizer_state,
        {k: jnp.reshape(v, (-1,)) for k, v in aux_losses.items()},
        return_ema,
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
    expected_one_step_backup=False,
):
    gamma_with_terminal = (cumulative_gamma *
                           (1.0 - terminals.astype(jnp.float32)))
    target_dist = target_network(next_states)
    policy_logits, sampled_action = policy_info(next_states, rng)

    # Compute the target Q-value distribution
    probabilities = jnp.squeeze(target_dist.probabilities)
    if expected_one_step_backup:
        # For the finite Atari action set, form the target-policy mixture
        # exactly instead of drawing one bootstrap action.  Mixing the full
        # categorical distributions preserves C51; reducing each action to
        # its scalar expectation here would discard distributional structure.
        policy_probabilities = jax.nn.softmax(policy_logits)
        next_probabilities = jnp.einsum("a,az->z", policy_probabilities,
                                        probabilities)
    else:
        next_probabilities = probabilities[sampled_action]
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
        expected_one_step_backup=False,
        retrace=False,
        retrace_horizon=10,
        retrace_lambda=1.0,
        retrace_warmup_n_step_updates=0,
        retrace_warmup_n_step_horizon=10,
        retrace_warmup_n_step_final_horizon=None,
        retrace_warmup_min_gamma=None,
        retrace_warmup_delta_based_priority=False,
        retrace_reset_priorities_on_target_switch=False,
        td_lower_bound_weight=0.0,
        td_lower_bound_horizon=10,
        td_lower_bound_priority_eta=0.5,
        td_lower_bound_priority_epsilon=1e-6,
        td_maximization_target=False,
        td_maximization_horizon=10,
        td_maximization_priority_epsilon=1e-6,
        delta_based_priority=False,
        delta_priority_epsilon=1e-6,
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
        reward_weight=1.0,
        continue_weight=1.0,
        reward_readout=False,
        continue_readout=False,
        reward_grad_surgery=False,
        imag_horizon=0,
        imag_actor_weight=0.0,
        imag_value_weight=0.0,
        imag_value_trust=None,
        imag_discount=None,
        imag_lambda=0.95,
        imag_warmup=2000,
        imag_entropy_weight=None,
        x_ent_decay_steps=80_000,
        x_ent_floor=0.0,
        half_precision=False,
        late_update_after=-1,
        late_update_until=-1,
        late_update_multiplier=1,
        seed=None,
        log_every=None,
        explore_end_steps=None,
        retrace_warmup_gamma=None,
        imag_warmup_ramp=True,
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
        self.expected_one_step_backup = bool(expected_one_step_backup)
        validate_expected_one_step_backup(self.expected_one_step_backup,
                                          self.update_horizon,
                                          max_update_horizon)
        self.retrace = bool(retrace)
        self.retrace_horizon = int(retrace_horizon)
        self.retrace_lambda = float(retrace_lambda)
        self.retrace_warmup_n_step_updates = int(
            retrace_warmup_n_step_updates)
        self.retrace_warmup_n_step_horizon = int(
            retrace_warmup_n_step_horizon)
        self.retrace_warmup_n_step_final_horizon = (
            self.retrace_warmup_n_step_horizon
            if retrace_warmup_n_step_final_horizon is None else
            int(retrace_warmup_n_step_final_horizon))
        self.retrace_warmup_min_gamma = (
            None if retrace_warmup_min_gamma is None else
            float(retrace_warmup_min_gamma))
        self.retrace_warmup_gamma = (
            None if retrace_warmup_gamma is None else
            float(retrace_warmup_gamma))
        self.retrace_warmup_delta_based_priority = bool(
            retrace_warmup_delta_based_priority)
        self.retrace_reset_priorities_on_target_switch = bool(
            retrace_reset_priorities_on_target_switch)
        validate_retrace(self.retrace,
                         self.expected_one_step_backup,
                         self.update_horizon,
                         max_update_horizon,
                         self.retrace_horizon,
                         self.retrace_lambda,
                         min_gamma,
                         cycle_steps,
                         self._distributional,
                         self.retrace_warmup_n_step_updates,
                         self.retrace_warmup_n_step_horizon,
                         self.retrace_warmup_n_step_final_horizon,
                         self.retrace_warmup_min_gamma,
                         self.retrace_warmup_gamma)
        self.td_lower_bound_weight = float(td_lower_bound_weight)
        self.td_lower_bound = self.td_lower_bound_weight > 0.0
        self.td_lower_bound_horizon = int(td_lower_bound_horizon)
        self.td_lower_bound_priority_eta = float(
            td_lower_bound_priority_eta)
        self.td_lower_bound_priority_epsilon = float(
            td_lower_bound_priority_epsilon)
        validate_td_lower_bound(
            self.td_lower_bound_weight,
            self.td_lower_bound_horizon,
            self.td_lower_bound_priority_eta,
            self.td_lower_bound_priority_epsilon,
            self.retrace,
            self.update_horizon,
            max_update_horizon,
            self._distributional,
        )
        self.td_maximization_target = bool(td_maximization_target)
        self.td_maximization_horizon = int(td_maximization_horizon)
        self.td_maximization_priority_epsilon = float(
            td_maximization_priority_epsilon)
        validate_td_maximization_target(
            self.td_maximization_target,
            self.td_maximization_horizon,
            self.td_maximization_priority_epsilon,
            self.retrace,
            self.td_lower_bound,
            self.expected_one_step_backup,
            self.update_horizon,
            max_update_horizon,
            self._distributional,
        )
        self.delta_based_priority = bool(delta_based_priority)
        self.delta_priority_epsilon = float(delta_priority_epsilon)
        validate_delta_based_priority(
            self.delta_based_priority,
            self.delta_priority_epsilon,
            self.retrace,
            self.td_lower_bound,
            self.td_maximization_target,
            self._distributional,
        )
        validate_retrace_warmup_priority_settings(
            self.retrace_warmup_delta_based_priority,
            self.retrace_reset_priorities_on_target_switch,
            self.delta_priority_epsilon,
            self.retrace,
            self.retrace_warmup_n_step_updates,
            self._distributional,
        )
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

        # Late-phase update multiplier: from env step late_update_after
        # onwards, do late_update_multiplier x the usual number of update
        # phases per env step (-1 / 1 = off). Counted in env steps, like
        # reset_every, because that is the unit the schedule is specified in;
        # what it actually changes is the gradient-step rate, so it moves the
        # post-reset recovery budget the same way replay_ratio does. Anything
        # keyed on cycle_grad_steps (update horizon, gamma, imag_warmup)
        # therefore anneals proportionally faster in the multiplied phase, as
        # it does under a higher replay_ratio; reset_every does NOT adapt on
        # its own (it is in env steps), so multiplying without shortening it
        # lengthens the final un-reset stretch in gradient steps.
        # late_update_until bounds the window on the right (-1 = run to the end
        # of training). A bounded window is what lets the multiplier target the
        # post-reset recovery phase specifically, rather than everything after
        # a threshold.
        self.late_update_after = int(late_update_after)
        self.late_update_until = int(late_update_until)
        self.late_update_multiplier = int(late_update_multiplier)
        self.late_updates_active = False

        self.learning_rate = learning_rate
        self.encoder_learning_rate = encoder_learning_rate

        self.shrink_perturb_keys = [
            s for s in shrink_perturb_keys.lower().split(",") if s
        ]
        self.shrink_perturb_keys = tuple(self.shrink_perturb_keys)
        self.shrink_factor = shrink_factor
        self.perturb_factor = perturb_factor

        # Select which slowly/rapidly changing policy collects environment
        # actions.  TD action selection is controlled separately by double_dqn.
        self.target_action_selection = bool(target_action_selection)
        # Controls value-evaluation parameters for TD and imagined bootstraps.
        self.use_target_network = bool(use_target_network)
        self.match_online_target_rngs = match_online_target_rngs
        self.target_eval_mode = target_eval_mode

        self.reward_weight = float(reward_weight)
        self.continue_weight = float(continue_weight)
        self.reward_readout = bool(reward_readout)
        # Detach ONLY the continue head from the trunk (its BCE is dense on
        # every game via life-loss terminals, so it grounds the trunk even
        # when rewards are sparse); reward stays per reward_readout. Only
        # meaningful with reward_readout=False.
        self.continue_readout = bool(continue_readout)
        # PCGrad on one loss: project the conflicting component out of the
        # reward/continue gradient at the shared encoder/TM. Costs a second
        # backward pass; only meaningful with reward_readout=False.
        self.reward_grad_surgery = bool(reward_grad_surgery)
        self.imag_horizon = int(imag_horizon)
        self.imag_actor_weight = float(imag_actor_weight)
        self.imag_value_weight = float(imag_value_weight)
        # None -> unbounded (original behavior); a float bounds the imagined
        # critic targets within imag_value_trust * return-scale of the
        # target critic's estimate for the taken action.
        self.imag_value_trust = (None if imag_value_trust is None else
                                 float(imag_value_trust))
        # None tracks the annealed TD discount (resolved per gradient step in
        # _training_step_update); a float pins it to that value instead.
        self.imag_discount = (None if imag_discount is None else
                              float(imag_discount))
        self.imag_lambda = float(imag_lambda)
        self.imag_warmup = int(imag_warmup)
        self.imag_warmup_ramp = bool(imag_warmup_ramp)
        # None -> imagination entropy follows the decaying x_ent_coef
        # schedule (original behavior); a float decouples it. 3e-4 is
        # DreamerV3's eta, calibrated for advantages normalized by the same
        # 5th-95th percentile return EMA this loss uses.
        self.imag_entropy_weight = (None if imag_entropy_weight is None else
                                    float(imag_entropy_weight))
        # Decay period of x_ent_coef, the (only live) actor entropy
        # coefficient: 1e-2 -> 0 linearly over this many env steps, then
        # clipped at exactly 0. 80_000 is the shipped schedule, so the last
        # 20k env steps of a 100k run train with no entropy term at all --
        # in the imagined actor loss too, since imag_entropy_weight=None
        # couples it to this same value. Setting it to 100_000 makes the
        # coefficient reach zero only at the end of training; note that it
        # also raises the coefficient at every earlier step, so it is not a
        # tail-only change.
        self.x_ent_decay_steps = int(x_ent_decay_steps)
        # Lower bound on x_ent_coef, applied AFTER the schedule, so the ramp
        # itself is untouched and the floor only bites once the line crosses
        # it: at the default 80_000 decay, a 1e-3 floor takes effect from
        # 72_000 env steps on (1e-2 * (80000-s)/80000 = 1e-3). 0.0 = off =
        # the shipped anneal-to-exactly-zero.
        self.x_ent_floor = float(x_ent_floor)
        self.imag_return_ema = np.zeros((2,), dtype=np.float32)
        self.use_world_model = (self.spr_weight > 0 or self.reward_weight > 0
                                or self.continue_weight > 0 or
                                self.imag_horizon > 0)

        # debug - start
        print('*' * 20)
        print(' self.target_eval_mode: {}'.format(self.target_eval_mode))
        print(' self.target_action_selection: {}'.format(
            self.target_action_selection))
        print(' self.expected_one_step_backup: {}'.format(
            self.expected_one_step_backup))
        print(' self.retrace: {}'.format(self.retrace))
        print(' self.retrace_horizon: {}'.format(self.retrace_horizon))
        print(' self.retrace_lambda: {}'.format(self.retrace_lambda))
        print(' self.retrace_warmup_n_step_updates: {}'.format(
            self.retrace_warmup_n_step_updates))
        print(' self.retrace_warmup_n_step_horizon: {}'.format(
            self.retrace_warmup_n_step_horizon))
        print(' self.retrace_warmup_n_step_final_horizon: {}'.format(
            self.retrace_warmup_n_step_final_horizon))
        print(' self.retrace_warmup_min_gamma: {}'.format(
            self.retrace_warmup_min_gamma))
        print(' self.retrace_warmup_gamma: {}'.format(
            self.retrace_warmup_gamma))
        print(' self.retrace_warmup_delta_based_priority: {}'.format(
            self.retrace_warmup_delta_based_priority))
        print(' self.retrace_reset_priorities_on_target_switch: {}'.format(
            self.retrace_reset_priorities_on_target_switch))
        print(' self.td_lower_bound: {}'.format(self.td_lower_bound))
        print(' self.td_lower_bound_weight: {}'.format(
            self.td_lower_bound_weight))
        print(' self.td_lower_bound_horizon: {}'.format(
            self.td_lower_bound_horizon))
        print(' self.td_lower_bound_priority_eta: {}'.format(
            self.td_lower_bound_priority_eta))
        print(' self.td_maximization_target: {}'.format(
            self.td_maximization_target))
        print(' self.td_maximization_horizon: {}'.format(
            self.td_maximization_horizon))
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

        if (self.retrace_warmup_n_step_final_horizon ==
                self.retrace_warmup_n_step_horizon):
            self.retrace_warmup_horizon_scheduler = (
                lambda x: self.retrace_warmup_n_step_horizon)
        else:
            retrace_warmup_n_schedule = exponential_decay_scheduler(
                self.retrace_warmup_n_step_updates,
                0,
                1,
                (self.retrace_warmup_n_step_final_horizon /
                 self.retrace_warmup_n_step_horizon),
            )
            self.retrace_warmup_horizon_scheduler = lambda x: int(  # pylint: disable=g-long-lambda
                np.round(retrace_warmup_n_schedule(x) *
                         self.retrace_warmup_n_step_horizon))

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

        if self.retrace_warmup_gamma is not None:
            self.retrace_warmup_gamma_scheduler = (
                lambda x: self.retrace_warmup_gamma)
        elif self.retrace_warmup_min_gamma is None:
            self.retrace_warmup_gamma_scheduler = lambda x: self.gamma
        else:
            final_gamma = float(self.gamma)
            valid_gamma_range = (
                0.0 <= self.retrace_warmup_min_gamma <= final_gamma < 1.0)
            if not np.isfinite(final_gamma) or not valid_gamma_range:
                raise ValueError(
                    "retrace_warmup_min_gamma ({}) and final gamma ({}) "
                    "must satisfy 0 <= warmup gamma <= final gamma < 1."
                    .format(self.retrace_warmup_min_gamma, final_gamma))
            self.retrace_warmup_gamma_scheduler = (
                exponential_decay_scheduler(
                    self.retrace_warmup_n_step_updates,
                    0,
                    self.retrace_warmup_min_gamma,
                    final_gamma,
                    reverse=True,
                ))

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
                do_rollout=self.use_world_model,
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

        head_keys = {"projection", "head", "predictor"}
        head_mask = FrozenDict({
            "params": {k: k in head_keys for k in self.online_params["params"]}
        })

        grounding_keys = {"reward_head", "continue_head"}
        grounding_mask = FrozenDict({
            "params": {
                k: k in grounding_keys for k in self.online_params["params"]
            }
        })

        policy_key = {"policy_projection", "policy", "predict_policy"}
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
        optimizer_key_groups = (
            encoder_keys,
            head_keys,
            grounding_keys,
            policy_key,
            alpha_key,
        )
        parameter_keys = set(self.online_params["params"])
        optimizer_membership = {
            key: sum(key in group for group in optimizer_key_groups)
            for key in parameter_keys
        }
        invalid_membership = {
            key: count
            for key, count in optimizer_membership.items() if count != 1
        }
        if invalid_membership:
            raise ValueError(
                "Optimizer masks must be disjoint and cover every parameter "
                "module exactly once; invalid memberships: {}".format(
                    invalid_membership))
        retained_optimizer_keys = encoder_keys | grounding_keys
        if set(RESET_OPTIMIZER_KEYS_TO_COPY) != retained_optimizer_keys:
            raise ValueError(
                "RESET_OPTIMIZER_KEYS_TO_COPY must exactly match the complete "
                "optimizer transforms retained across reset: {}".format(
                    sorted(retained_optimizer_keys)))
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
            optax.masked(optimizer, grounding_mask),
            optax.masked(policy_optim, policy_mask),
            optax.masked(alpha_optim, alpha_mask),
        )

        self.optimizer_state = self.optimizer.init(self.online_params)
        self.target_network_params = copy.deepcopy(self.online_params)
        self.random_params = copy.deepcopy(self.online_params)

        #print(' so far so good')
        #exit(0)

    def _build_replay_buffer(self):
        # Multi-step targets consume raw transitions. Endpoint states S1..SH
        # come from the one-step `next_state` columns, so only H replay rows are
        # needed. SPR keeps its independent jumps+1 state sequence.
        subseq_len = max(
            self._jumps + 1,
            self.retrace_horizon if self.retrace else 1,
            (getattr(self, "retrace_warmup_n_step_horizon", 1)
             if (self.retrace and getattr(
                 self, "retrace_warmup_n_step_updates", 0) > 0) else 1),
            (getattr(self, "td_lower_bound_horizon", 1)
             if getattr(self, "td_lower_bound", False) else 1),
            (getattr(self, "td_maximization_horizon", 1)
             if getattr(self, "td_maximization_target", False) else 1),
        )
        extra_storage_types = None
        if self.retrace:
            extra_storage_types = [
                circular_replay_buffer.ReplayElement(
                    'behavior_probability', (), np.float32)
            ]
        prioritized_buffer = subsequence_replay_buffer.PrioritizedJaxSubsequenceParallelEnvReplayBuffer(
            observation_shape=self.observation_shape,
            stack_size=self.stack_size,
            update_horizon=self.max_update_horizon,
            gamma=self.gamma,
            subseq_len=subseq_len,
            batch_size=self._batch_size,
            extra_storage_types=extra_storage_types,
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
        warmup_updates = getattr(
            self, "retrace_warmup_n_step_updates", 0)
        if warmup_updates % self._batches_to_group:
            raise ValueError(
                "retrace_warmup_n_step_updates ({}) must be divisible by "
                "the effective batches_to_group ({}) so one grouped JAX "
                "call cannot straddle the target switch.".format(
                    warmup_updates, self._batches_to_group))
        hard_imagination_gate = (
            not getattr(self, "imag_warmup_ramp", True) and
            getattr(self, "imag_horizon", 0) > 0 and
            (getattr(self, "imag_actor_weight", 0.0) != 0.0 or
             getattr(self, "imag_value_weight", 0.0) != 0.0))
        if hard_imagination_gate:
            if self.imag_warmup < 0:
                raise ValueError(
                    "An abrupt imagination gate requires imag_warmup to be "
                    "nonnegative, got {}.".format(self.imag_warmup))
            if self.imag_warmup % self._batches_to_group:
                raise ValueError(
                    "imag_warmup ({}) must be divisible by the effective "
                    "batches_to_group ({}) so one grouped JAX call cannot "
                    "straddle the abrupt imagination boundary.".format(
                        self.imag_warmup, self._batches_to_group))
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

    def _uses_ere_replay(self):
        """Whether the configured replay buffer uses ERE anchor sampling."""
        # ``is True`` keeps lightweight Mock-based tests and alternate replay
        # implementations on the legacy prefetch path unless they explicitly
        # opt into ERE.
        replay = getattr(self, "_replay", None)
        return getattr(replay, "ere_sampling", False) is True

    def _sample_replay_batch(self,
                             ere_update_index=None,
                             ere_num_updates=None):
        types = self._replay.get_transition_elements()
        self._rng, rng = jax.random.split(self._rng)

        # Retrace and the lower-bound target both consume raw rows and own
        # separate multi-step horizons. A configured Retrace warmup can use
        # its cycle-local gamma schedule; active Retrace uses the fixed final
        # gamma. The lower-bound mode may use the ordinary gamma schedule,
        # with the sampled one-step discount serving as gamma.
        raw_replay_rows = (
            self.retrace or self.td_lower_bound or
            getattr(self, "td_maximization_target", False))
        replay_horizon = (1 if raw_replay_rows else
                          self.update_horizon_scheduler(
                              self.cycle_grad_steps))
        retrace_warmup_n_step = (
            self.retrace and not retrace_target_active(
                self.retrace,
                self.cycle_grad_steps,
                self.retrace_warmup_n_step_updates,
            ))
        if retrace_warmup_n_step:
            replay_gamma = self.retrace_warmup_gamma_scheduler(
                self.cycle_grad_steps)
        elif self.retrace:
            replay_gamma = self.gamma
        else:
            replay_gamma = self.gamma_scheduler(self.cycle_grad_steps)
        sampling_kwargs = {}
        if ere_update_index is not None or ere_num_updates is not None:
            if ere_update_index is None or ere_num_updates is None:
                raise ValueError(
                    "ERE update index and update count must be provided "
                    "together.")
            sampling_kwargs = {
                "ere_update_index": ere_update_index,
                "ere_num_updates": ere_num_updates,
                "ere_batch_size": self._batch_size,
            }
        samples = self._replay.sample_transition_batch(
            rng,
            batch_size=self._batch_size * self._batches_to_group,
            update_horizon=replay_horizon,
            gamma=replay_gamma,
            **sampling_kwargs,
        )
        replay_elements = collections.OrderedDict()
        for element, element_type in zip(samples, types):
            replay_elements[element_type.name] = element
        return replay_elements

    def _replay_sampler_generator(self):
        while True:
            yield self._sample_replay_batch()

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
                                             1)

    def _sample_from_replay_buffer(self):
        self.replay_elements = next(self.prefetcher)

    def _discard_pending_replay_sample(self):
        """Drops any batch selected under obsolete schedules or priorities."""
        if hasattr(self, "prefetcher") and not self._uses_ere_replay():
            self.initialize_prefetcher()
        if hasattr(self, "replay_elements"):
            del self.replay_elements

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
            self.use_world_model,
            self._support,
            self.reset_target,
            self.shrink_perturb_keys,
            self.shrink_factor,
            self.perturb_factor,
            RESET_PARAMETER_KEYS_TO_COPY,
            RESET_OPTIMIZER_KEYS_TO_COPY,
        )

        self.cycle_grad_steps = 0
        if getattr(self, "retrace", False):
            # Q heads were reset, so their old error-based priorities no longer
            # describe the current critic. Restore uniform sampling until fresh
            # priorities from the new warmup cycle are written.
            self._replay.reset_priorities()
        # Returns/discounts are materialized by the replay sampler using the
        # cycle schedule at sampling time.  Discard the pre-reset sample and
        # generator so the first post-reset update is sampled at cycle step 0.
        self._discard_pending_replay_sample()

    def _training_step_update(self,
                              step_index,
                              offline=False,
                              num_update_groups=1):
        """Gradient update during every training step."""
        self.start = time.time()

        ere_sampling = self._uses_ere_replay()
        retrace_enabled = getattr(self, "retrace", False)
        warmup_n_step_updates = getattr(
            self, "retrace_warmup_n_step_updates", 0)
        retrace_target_enabled = retrace_target_active(
            retrace_enabled,
            self.cycle_grad_steps,
            warmup_n_step_updates,
        )
        if retrace_priority_reset_due(
                retrace_enabled,
                getattr(self,
                        "retrace_reset_priorities_on_target_switch", False),
                self.cycle_grad_steps,
                warmup_n_step_updates):
            logging.info(
                "\t Resetting replay priorities at the n-step-to-Retrace "
                "target switch (cycle gradient step %s).",
                self.cycle_grad_steps)
            self._replay.reset_priorities()
            # The legacy depth-one prefetcher may already hold a batch selected
            # with warmup delta priorities. Recreate it and force a new sample
            # so the first Retrace update is selected from the uniform tree.
            self._discard_pending_replay_sample()

        if ere_sampling:
            # One replay call feeds ``_batches_to_group`` sequential JAX
            # minibatches. Give each contiguous chunk its own ERE k/K window.
            self.replay_elements = self._sample_replay_batch(
                ere_update_index=step_index * self._batches_to_group,
                ere_num_updates=(num_update_groups *
                                 self._batches_to_group),
            )
        elif not hasattr(self, "replay_elements"):
            self._sample_from_replay_buffer()

        probs = self.replay_elements["sampling_probabilities"]
        retrace_warmup_n_step = (
            retrace_enabled and not retrace_target_enabled)
        retrace_warmup_n_step_horizon = int(
            self.retrace_warmup_horizon_scheduler(self.cycle_grad_steps))
        mean_priority = None
        if retrace_enabled:
            # Signed Distributional Retrace coefficients are nonnegative only
            # in expectation over behavior trajectories, so its TD loss needs
            # beta=1 PER correction. Preserve that correction during the
            # n-step warmup so the schedule changes only the target. A
            # replay-wide normalizer preserves inverse-priority relative
            # weights without a sampled-batch factor.
            mean_priority = float(self._replay.mean_priority())
        # Preserve historical beta=0.5 weighting for SPR, actor, and imagined
        # objectives. The hybrid changes only the replay critic target.
        loss_weights, td_loss_weights = replay_loss_weights(
            probs, retrace=retrace_enabled, mean_priority=mean_priority)
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

        # Imagination is gated off right after each shrink-and-perturb reset
        # (the Q target the imagined values come from is also reset). The
        # historical mode ramps it back in over another imag_warmup updates;
        # the opt-in hard gate turns it on at full weight at that boundary.
        imag_ramp = imagination_training_scale(
            self.cycle_grad_steps,
            self.imag_warmup,
            ramp=getattr(self, "imag_warmup_ramp", True),
        )
        imag_actor_mult = self.imag_actor_weight * imag_ramp
        imag_value_mult = self.imag_value_weight * imag_ramp

        # Keep imagination aligned with the critic discount. A dynamic Retrace
        # warmup follows its phase-local schedule; active Retrace uses the fixed
        # final gamma.
        if retrace_warmup_n_step:
            critic_discount = self.retrace_warmup_gamma_scheduler(
                self.cycle_grad_steps)
        elif retrace_enabled:
            critic_discount = self.gamma
        else:
            critic_discount = self.gamma_scheduler(self.cycle_grad_steps)
        imag_discount = (critic_discount if self.imag_discount is None else
                         self.imag_discount)

        if retrace_enabled and "behavior_probability" not in self.replay_elements:
            raise KeyError(
                "Retrace replay samples must include behavior_probability.")
        behavior_probabilities = self.replay_elements.get(
            "behavior_probability",
            np.ones_like(self.replay_elements["action"], dtype=np.float32),
        )

        self._rng, train_rng = jax.random.split(self._rng)
        (
            new_online_params,
            new_target_params,
            new_optimizer_state,
            aux_losses,
            new_return_ema,
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
            self.replay_elements["terminal"],
            self.replay_elements["same_trajectory"],
            loss_weights,
            td_loss_weights,
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
            self.expected_one_step_backup,
            retrace_target_enabled,
            retrace_warmup_n_step,
            getattr(self, "_jumps", 0),
            getattr(self, "retrace_horizon", 1),
            retrace_warmup_n_step_horizon,
            getattr(self, "retrace_warmup_delta_based_priority", False),
            getattr(self, "retrace_lambda", 1.0),
            getattr(self, "td_lower_bound", False),
            getattr(self, "td_lower_bound_horizon", 1),
            getattr(self, "td_lower_bound_weight", 0.0),
            getattr(self, "td_lower_bound_priority_eta", 0.5),
            getattr(self, "td_lower_bound_priority_epsilon", 1e-6),
            getattr(self, "td_maximization_target", False),
            getattr(self, "td_maximization_horizon", 1),
            getattr(self, "td_maximization_priority_epsilon", 1e-6),
            getattr(self, "delta_based_priority", False),
            getattr(self, "delta_priority_epsilon", 1e-6),
            self.target_update_tau_scheduler(self.cycle_grad_steps),
            self.target_update_period,
            self.grad_steps,
            self.match_online_target_rngs,
            self.target_eval_mode,
            #self.ent_targ,
            self.x_ent_coef,
            self.replay_elements["reward"],
            behavior_probabilities,
            self.reward_weight,
            self.continue_weight,
            self.reward_readout,
            self.continue_readout,
            self.reward_grad_surgery,
            self.imag_horizon,
            imag_actor_mult,
            imag_value_mult,
            imag_discount,
            self.imag_lambda,
            (self.x_ent_coef if self.imag_entropy_weight is None else
             self.imag_entropy_weight),
            (float('inf')
             if self.imag_value_trust is None else self.imag_value_trust),
            self.imag_return_ema,
        )
        self.imag_return_ema = np.asarray(new_return_ema)
        self.grad_steps += self._batches_to_group
        self.cycle_grad_steps += self._batches_to_group

        # Sample the next legacy PER/uniform batch while we wait for training.
        # ERE samples synchronously at the start of each update so its k/K
        # phase cannot lag behind through the depth-one prefetch queue.
        if not ere_sampling:
            self._sample_from_replay_buffer()
        # Rainbow and prioritized replay use alpha=0.5 here, implemented by
        # storing sqrt(raw_score). The ordinary path supplies C51
        # cross-entropy unless a delta-priority mode supplies absolute scalar TD
        # plus its configured epsilon. Maximization also supplies absolute
        # scalar TD. Retrace and the default n-step warmup supply squared TV;
        # row 40 opts its warmup into scalar delta instead. The lower-bound path
        # supplies its documented u_t score. Add a small nonzero value before
        # sqrt so zero scores cannot produce zero sampling probabilities or NaN
        # correction terms.
        indices = np.reshape(np.asarray(indices), (-1,))
        priority_loss = np.reshape(
            np.asarray(aux_losses["PriorityLoss"]), (-1))
        validate_priority_cardinality(indices, priority_loss)

        # debug - start
        #if random.uniform(0, 1) < 1e-3:
        if False:
            logging.info("ent: {}".format(aux_losses["ent"]))
        # debug - end

        priorities = np.sqrt(priority_loss + 1e-10)
        if not np.all(np.isfinite(priorities)):
            raise FloatingPointError("Replay priorities must be finite.")
        validate_priority_cardinality(indices, priorities)
        self._replay.set_priority(indices, priorities)

        if self.grad_steps % 500 < self._batches_to_group:
            log_keys = ("RewardLoss", "ContinueLoss", "RewardCorr",
                        "GroundCos", "GroundCosEnc", "GroundCosTM",
                        "GroundNormRatio", "ImagActorLoss", "ImagValueLoss",
                        "ImagRet", "ImagReward", "ImagContinue",
                        "ImagEntropy", "ImagTrustClip",
                        "RetraceNegativeMass", "RetraceTargetMass",
                        "LowerBoundLoss", "LowerBoundGap",
                        "LowerBoundActive", "OneStepAbsTD", "NStepTarget",
                        "MaxTargetOneStep", "MaxTargetNStep",
                        "MaxTargetValue", "MaxTargetLift",
                        "MaxTargetNStepSelected", "MaxTargetAbsTD")
            msgs = ["grad_step {}".format(self.grad_steps)]
            for k in log_keys:
                if k in aux_losses:
                    msgs.append("{}: {:.4f}".format(
                        k, float(np.mean(np.asarray(aux_losses[k])))))
            if len(msgs) > 1:
                logging.info(" | ".join(msgs))

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

        self.x_ent_coef = linearly_decaying_epsilon(self.x_ent_decay_steps,
                                                    self.training_steps, 0, .0)
        if self.x_ent_floor > 0.0:
            self.x_ent_coef = jnp.maximum(self.x_ent_coef, self.x_ent_floor)
        if random.uniform(0, 1) < 1e-3:
            logging.info("step: {}, x_ent_coef: {}".format(
                self.training_steps, self.x_ent_coef))

        if (self._replay.add_count == self.min_replay_history and
                not self._uses_ere_replay()):
            self.initialize_prefetcher()

        if self._replay.add_count > self.min_replay_history:
            if self.training_steps % self.update_period == 0:
                num_updates = self._num_updates_per_train_step
                late_on = (self.late_update_after >= 0
                           and self.training_steps >= self.late_update_after
                           and (self.late_update_until < 0 or
                                self.training_steps < self.late_update_until))
                if late_on:
                    num_updates *= self.late_update_multiplier
                if late_on != self.late_updates_active:
                    self.late_updates_active = late_on
                    logging.info(
                        "\t Late-phase updates %s at step %s: %s update phase(s)"
                        " per env step (x%s over [%s, %s)).",
                        "ON" if late_on else "OFF", self.training_steps,
                        num_updates, self.late_update_multiplier,
                        self.late_update_after, self.late_update_until)
                for i in range(num_updates):
                    self._training_step_update(
                        i,
                        offline=False,
                        num_update_groups=num_updates,
                    )
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
        self._behavior_probabilities = None
        self._record_observation(new_obs)

    def reset_one(self, env_id):
        self.state[env_id].fill(0)

    def delete_one(self, env_id):
        self.state = np.concatenate(
            [self.state[:env_id], self.state[env_id + 1:]], 0)
        if getattr(self, "_behavior_probabilities", None) is not None:
            self._behavior_probabilities = np.concatenate(
                [self._behavior_probabilities[:env_id],
                 self._behavior_probabilities[env_id + 1:]], 0)

    def cache_train_state(self):
        self.training_state = (
            copy.deepcopy(self.state),
            copy.deepcopy(self._last_observation),
            copy.deepcopy(self._observation),
        )

    def restore_train_state(self):
        (self.state, self._last_observation,
         self._observation) = (self.training_state)

    def log_transition(self, observation, action, reward, terminal,
                       episode_end):
        self._last_observation = self._observation
        self._record_observation(observation)

        if not self.eval_mode:
            extra_replay_values = ()
            if self.retrace:
                if self._behavior_probabilities is None:
                    raise RuntimeError(
                        "Retrace requires the behavior probability recorded "
                        "when the replay action was selected.")
                behavior_probabilities = np.asarray(
                    self._behavior_probabilities, dtype=np.float32)
                if behavior_probabilities.shape != np.asarray(action).shape:
                    raise ValueError(
                        "Behavior probabilities must align with replay actions: "
                        "{} versus {}.".format(behavior_probabilities.shape,
                                              np.asarray(action).shape))
                extra_replay_values = (behavior_probabilities,)
            self._store_transition(
                self._last_observation,
                action,
                reward,
                terminal,
                *extra_replay_values,
                episode_end=episode_end,
            )

    def select_action(
        self,
        state,
        select_params,
        eval_mode,
    ):
        if not eval_mode and self.training_steps < self.min_replay_history:
            self._rng, key = jax.random.split(self._rng)
            action = jax.random.randint(
                key,
                (state.shape[0],),
                0,
                self.num_actions,
            )
            self._behavior_probabilities = np.full(
                (state.shape[0],),
                1.0 / self.num_actions,
                dtype=np.float32,
            )
            return action
        self._rng, action, probs = select_action(
            self.network_def,
            select_params,
            state,
            self._rng,
        )
        selected_probabilities = jnp.take_along_axis(
            probs, action[:, None].astype(jnp.int32), axis=-1)[:, 0]
        self._behavior_probabilities = np.asarray(
            selected_probabilities, dtype=np.float32)
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
        state = self.state
        select_params = behavior_parameter_set(
            self.online_params, self.target_network_params,
            self.target_action_selection)
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
        #time_delta = time.time() - start_time
        #print('time_delta: {}'.format(time_delta))
        #exit(0)
        return self.action
