# coding=utf-8

import collections
import copy
import functools
import time

from absl import logging
from flax.core.frozen_dict import FrozenDict
import gin
import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax

from bbf import spr_networks
from bbf.replay_memory import subsequence_replay_buffer

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
)
RESET_OPTIMIZER_KEYS_TO_COPY = (
    "encoder",
    "transition_model",
    "reward_head",
    "continue_head",
)

# This branch intentionally supports one TD schedule only. Keep these values
# in code so an obsolete Gin binding cannot silently change the experiment.
INITIAL_TD_HORIZON = 10
FINAL_TD_HORIZON = 1
INITIAL_TD_GAMMA = 0.97
FINAL_TD_GAMMA = 0.997
TD_SCHEDULE_GRAD_UPDATES = 20_000
DELTA_PRIORITY_EPSILON = 1e-6


def td_schedule(cycle_grad_steps):
    """Returns the historical exponential horizon/gamma schedule.

    The cycle begins at H=10 and gamma=.97. Both schedules anneal over 20,000
    gradient updates; after that they remain at H=1 and gamma=.997.
    """
    progress = np.clip(
        float(cycle_grad_steps) / TD_SCHEDULE_GRAD_UPDATES, 0.0, 1.0)
    decay = 0.1 ** progress
    horizon = int(np.round(INITIAL_TD_HORIZON * decay))
    horizon = int(np.clip(horizon, FINAL_TD_HORIZON, INITIAL_TD_HORIZON))
    gamma = 1.0 - (1.0 - INITIAL_TD_GAMMA) * decay
    gamma = float(np.clip(gamma, INITIAL_TD_GAMMA, FINAL_TD_GAMMA))
    return horizon, gamma


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


def distributional_td_signals(target_distribution,
                              predicted_probabilities,
                              support,
                              priority_epsilon=1e-6):
    """Returns the projected C51 expectation error and its PER score.

    The target is the same projected distribution used by the categorical
    critic loss.  Computing its support mean therefore preserves C51's support
    clipping semantics. The scalar score is stopped so replay prioritization
    cannot become an optimization objective.
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


def priority_weighted_masked_mean(values, mask, loss_weights, eps=1e-6):
    """Masked mean whose trajectory rows carry their sampled-s0 weights.

    ``values`` and ``mask`` have shape ``[B, H]`` and ``loss_weights`` has
    shape ``[B]``.  The original unweighted mask denominator is retained, so
    unit weights exactly recover the unweighted masked mean and PER changes each sampled
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


def replay_loss_weights(priorities, mean_priority):
    """Builds fixed beta=.5 auxiliary and beta=1 critic PER weights."""
    priorities = np.asarray(priorities, dtype=np.float32)
    if priorities.size == 0:
        raise ValueError("Replay priorities must not be empty.")
    if (not np.all(np.isfinite(priorities)) or
            np.any(priorities <= 0.0)):
        raise FloatingPointError(
            "Sampled replay priorities must be finite and positive.")

    auxiliary_weights = 1.0 / np.sqrt(priorities + 1e-10)
    auxiliary_weights /= np.max(auxiliary_weights)
    if (mean_priority is None or not np.isfinite(mean_priority) or
            mean_priority <= 0.0):
        raise FloatingPointError(
            "Replay mean priority must be finite and positive.")
    td_weights = np.asarray(
        mean_priority / (priorities + 1e-10), dtype=np.float32)
    if not np.all(np.isfinite(td_weights)):
        raise FloatingPointError("Critic importance weights must be finite.")
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

    return online_params, target_network_params, optimizer_state


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
    _, samples = jax.vmap(logits_w_samples, in_axes=0,
                          axis_name="batch")(state, key)
    # On-policy categorical sampling for both training and evaluation.
    return rng, samples


train_static_argnames = [
    'network_def',
    'optimizer',
    'spr_weight',
    'data_augmentation',
    'dtype',
    'batch_size',
    'spr_horizon',
    'target_eval_mode',
    'reward_weight',
    'continue_weight',
    'reward_readout',
    'continue_readout',
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
    rng,  # 16
    spr_weight,  # 17, static (gates rollouts)
    data_augmentation,  # static
    dtype,  # static
    batch_size,  # static
    spr_horizon,  # static
    delta_priority_epsilon,
    target_update_tau,
    target_update_every,
    step,
    target_eval_mode,  # static
    #ent_targ,
    x_ent_coef,
    per_step_rewards,
    reward_weight,  # static
    continue_weight,  # static
    reward_readout,  # static
    continue_readout,  # static
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
        ) = inputs
        # Replay materializes only the model sequence plus one root TD endpoint.
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
        root_rewards = rewards
        root_terminals = terminals
        root_discount = cumulative_gamma

        rng, rng1, rng2 = jax.random.split(rng, num=3)
        states = spr_networks.process_inputs(
            model_raw_states,
            rng=rng1,
            data_augmentation=data_augmentation,
            dtype=dtype)
        # Replay has already applied the current horizon and gamma, selected
        # S_H, and accumulated its root return. The online policy sees an
        # unaugmented S_H and the target critic an independently augmented S_H.
        endpoint_policy_states = spr_networks.process_inputs(
            raw_next_states,
            data_augmentation=False,
            dtype=dtype,
        )
        endpoint_value_states = spr_networks.process_inputs(
            raw_next_states,
            rng=rng2,
            data_augmentation=data_augmentation,
            dtype=dtype,
        )
        current_state = states[:, 0]

        # Split the current rng to update the rng after this call
        rng, _, rng2 = jax.random.split(rng, num=3)
        use_spr = (spr_weight > 0 or reward_weight > 0 or continue_weight > 0
                   or imag_horizon > 0)

        backup_params = target_params

        def policy_backup(state):
            return network_def.apply(
                online_params,
                state,
                method=network_def.get_policy_logits,
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
                x_ent = -(prob * log_prob).sum()
                return (-(jax.lax.stop_gradient(q_values) *
                          log_prob[samples]) - x_ent_coef * x_ent)

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
            predicted_probabilities = jax.nn.softmax(chosen_action_logits)
            _, priority_loss = distributional_td_signals(
                target,
                predicted_probabilities,
                support[None, :],
                delta_priority_epsilon,
            )

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
            spr_loss = (spr_loss * same_traj_mask.transpose(1, 0)).mean(0) * .5
            # Full beta=1 correction is fixed for the real C51 critic. SPR and
            # every other replay-anchored auxiliary retain beta=0.5.
            loss = (td_loss_multipliers * dqn_loss +
                    loss_multipliers * spr_weight * spr_loss)

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
                # Imagined values use the beta=.5 auxiliary PER weights because
                # their rollouts are anchored at the sampled replay states.
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
                apply_replay_anchor_weights(policy_out, loss_multipliers))
            total_loss = (mean_loss +
                          replay_actor_loss +
                          model_loss + imag_actor_mult * imag_actor_loss +
                          imag_value_mult * imag_value_loss)
            aux_losses = {
                # Keep the per-example delta priority until after the grouped
                # minibatch scan so every replay anchor is updated.
                "PriorityLoss": priority_loss,
                "ReturnEMAState": new_return_ema,
            }
            aux_losses.update(aux_model)
            aux_losses.update(imag_metrics)
            return total_loss, (aux_losses)

        # Exact C51 policy-mixture bootstrap at S_H. The online policy sees the
        # unaugmented endpoint while the target critic sees its augmented view.
        endpoint_policy_logits = jax.vmap(
            policy_backup, in_axes=0, axis_name="batch")(
                endpoint_policy_states)
        endpoint_target_output = jax.vmap(
            q_backup, in_axes=0, axis_name="batch")(endpoint_value_states)
        endpoint_probabilities = jnp.einsum(
            "ba,baz->bz",
            jax.nn.softmax(endpoint_policy_logits, axis=-1),
            endpoint_target_output.probabilities,
        )
        gamma_with_terminal = root_discount * (
            1.0 - root_terminals.astype(jnp.float32))
        target_supports = (root_rewards[:, None] +
                           gamma_with_terminal[:, None] * support[None, :])
        target = jax.vmap(
            project_distribution,
            in_axes=(0, 0, None),
            axis_name="batch",
        )(target_supports, endpoint_probabilities, support)
        target = jax.lax.stop_gradient(target)

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
        loss_args = (target, spr_targets, future_latents, loss_weights,
                     td_loss_weights, key, imag_keys)
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


@gin.configurable
def create_scaling_optimizer(
    learning_rate=6.25e-5,
    beta1=0.9,
    beta2=0.999,
    eps=1.5e-4,
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
        min_replay_history=20000,
        update_period=4,
        eval_mode=False,
        seed=None,
    ):
        assert isinstance(observation_shape, tuple)
        seed = int(time.time() * 1e6) if seed is None else seed
        logging.info('Creating %s agent with the following parameters:',
                     self.__class__.__name__)
        logging.info('\t min_replay_history: %d', min_replay_history)
        logging.info('\t update_period: %d', update_period)
        logging.info('\t seed: %d', seed)

        self.num_actions = num_actions
        self.observation_shape = tuple(observation_shape)
        self.observation_dtype = observation_dtype
        self.stack_size = stack_size
        self.network_def = network(num_actions=num_actions)
        self.min_replay_history = min_replay_history
        self.update_period = update_period
        self.eval_mode = eval_mode
        self.training_steps = 0

        self._rng = jax.random.PRNGKey(seed)
        state_shape = self.observation_shape + (stack_size,)
        self.state = np.zeros(state_shape, dtype=self.observation_dtype)
        self._replay = self._build_replay_buffer()
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
        data_augmentation=False,
        network=spr_networks.RainbowDQNNetwork,
        num_atoms=51,
        vmax=10.0,
        vmin=None,
        jumps=0,
        spr_weight=0,
        batch_size=32,
        replay_ratio=64,
        batches_to_group=1,
        reset_every=-1,
        no_resets_after=-1,
        reset_offset=1,
        learning_rate=0.0001,
        encoder_learning_rate=0.0001,
        reset_target=True,
        shrink_perturb_keys="",
        perturb_factor=0.2,  # original was 0.1
        shrink_factor=0.8,  # original was 0.4
        target_update_tau=1.0,
        target_update_period=1,
        target_eval_mode=False,
        reward_weight=1.0,
        continue_weight=1.0,
        reward_readout=False,
        continue_readout=False,
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
        seed=None,
    ):
        logging.info(
            "Creating %s agent with the following parameters:",
            self.__class__.__name__,
        )
        logging.info("\t data_augmentation: %s", data_augmentation)
        # We need casting because passing arguments can convert ints to floats
        vmax = float(vmax)
        self._num_atoms = int(num_atoms)
        vmin = float(vmin) if vmin else -vmax
        self._support = jnp.linspace(vmin, vmax, self._num_atoms)
        self._data_augmentation = bool(data_augmentation)
        self._replay_ratio = int(replay_ratio)
        self._batch_size = int(batch_size)
        self._batches_to_group = int(batches_to_group)
        self.delta_priority_epsilon = DELTA_PRIORITY_EPSILON
        self._jumps = int(jumps)
        self.spr_weight = spr_weight

        self.reset_every = int(reset_every)
        self.reset_target = reset_target
        self.no_resets_after = int(no_resets_after)
        self.cumulative_resets = 0
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

        self.target_eval_mode = target_eval_mode

        self.reward_weight = float(reward_weight)
        self.continue_weight = float(continue_weight)
        self.reward_readout = bool(reward_readout)
        # Detach ONLY the continue head from the trunk (its BCE is dense on
        # every game via life-loss terminals, so it grounds the trunk even
        # when rewards are sparse); reward stays per reward_readout. Only
        # meaningful with reward_readout=False.
        self.continue_readout = bool(continue_readout)
        self.imag_horizon = int(imag_horizon)
        self.imag_actor_weight = float(imag_actor_weight)
        self.imag_value_weight = float(imag_value_weight)
        # None -> unbounded (original behavior); a float bounds the imagined
        # critic targets within imag_value_trust * return-scale of the
        # target critic's estimate for the taken action.
        self.imag_value_trust = (None if imag_value_trust is None else
                                 float(imag_value_trust))
        # None follows the scheduled TD discount; a float overrides it.
        self.imag_discount = (None if imag_discount is None else
                              float(imag_discount))
        self.imag_lambda = float(imag_lambda)
        self.imag_warmup = int(imag_warmup)
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

        self.grad_steps = 0
        self.cycle_grad_steps = 0
        self.target_update_period = int(target_update_period)
        self.target_update_tau = target_update_tau

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
                distributional=True,
                dtype=self.dtype,
            ),
            seed=seed,
        )

        self.set_replay_settings()

        self.train_fn = jax.jit(train,
                                static_argnames=train_static_argnames,
                                device=jax.local_devices()[0])

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

        optimizer_key_groups = (
            encoder_keys,
            head_keys,
            grounding_keys,
            policy_key,
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
        self.optimizer = optax.chain(
            optax.masked(encoder_optimizer, encoder_mask),
            optax.masked(optimizer, head_mask),
            optax.masked(optimizer, grounding_mask),
            optax.masked(policy_optim, policy_mask),
        )

        self.optimizer_state = self.optimizer.init(self.online_params)
        self.target_network_params = copy.deepcopy(self.online_params)

    def _build_replay_buffer(self):
        # Replay returns the six-step model sequence plus one root TD endpoint.
        # Its compact output shape is fixed while H and gamma are scheduled.
        subseq_len = self._jumps + 1
        prioritized_buffer = subsequence_replay_buffer.PrioritizedJaxSubsequenceParallelEnvReplayBuffer(
            observation_shape=self.observation_shape,
            stack_size=self.stack_size,
            update_horizon=INITIAL_TD_HORIZON,
            subseq_len=subseq_len,
            batch_size=self._batch_size,
            observation_dtype=self.observation_dtype,
        )

        self.n_envs = prioritized_buffer._n_envs  # pylint: disable=protected-access
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
        if TD_SCHEDULE_GRAD_UPDATES % self._batches_to_group:
            raise ValueError(
                "TD_SCHEDULE_GRAD_UPDATES ({}) must be divisible by the "
                "effective batches_to_group ({}) so the schedule reaches its "
                "endpoint between grouped JAX calls.".format(
                    TD_SCHEDULE_GRAD_UPDATES, self._batches_to_group))
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

    def _next_replay_rng(self):
        """Returns a CPU-resident key without depending on pending GPU work."""
        self._rng, replay_rng = jax.random.split(self._rng)
        return jax.device_put(replay_rng, jax.devices("cpu")[0])

    def _sample_replay_batch(self, replay_rng):
        batch_size = self._batch_size * self._batches_to_group
        update_horizon, gamma = td_schedule(self.cycle_grad_steps)
        types = self._replay.get_transition_elements(batch_size=batch_size)
        samples = self._replay.sample_transition_batch(
            replay_rng,
            batch_size=batch_size,
            update_horizon=update_horizon,
            gamma=gamma,
        )
        replay_elements = collections.OrderedDict()
        for element, element_type in zip(samples, types):
            replay_elements[element_type.name] = element
        return replay_elements

    def _discard_pending_replay_sample(self):
        """Drops any materialized batch selected before a global reset."""
        if hasattr(self, "replay_elements"):
            del self.replay_elements
        if hasattr(self, "replay_mean_priority"):
            del self.replay_mean_priority

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
        # The reset critic invalidates every old delta score. Restart the
        # H=10/gamma=.97 schedule from uniform sampling, with no materialized
        # batch surviving the reset.
        self._replay.reset_priorities()
        self._discard_pending_replay_sample()

    def _training_step_update(self):
        """Gradient update during every training step."""
        if not hasattr(self, "replay_elements"):
            self.replay_elements = self._sample_replay_batch(
                self._next_replay_rng())
            self.replay_mean_priority = float(self._replay.mean_priority())

        sampled_priorities = self.replay_elements["priorities"]
        # The real critic uses beta=1; replay-anchored auxiliary and imagined
        # objectives retain beta=.5.
        loss_weights, td_loss_weights = replay_loss_weights(
            sampled_priorities, self.replay_mean_priority)
        indices = self.replay_elements["indices"]

        # Imagination is gated off right after each shrink-and-perturb reset
        # (the Q target the imagined values come from is also reset), then
        # ramped back in linearly over another imag_warmup gradient steps.
        imag_ramp = float(
            np.clip(
                (self.cycle_grad_steps - self.imag_warmup) /
                max(1, self.imag_warmup), 0.0, 1.0))
        imag_actor_mult = self.imag_actor_weight * imag_ramp
        imag_value_mult = self.imag_value_weight * imag_ramp

        _, scheduled_gamma = td_schedule(self.cycle_grad_steps)
        imag_discount = (scheduled_gamma if self.imag_discount is None else
                         self.imag_discount)

        self._rng, train_rng = jax.random.split(self._rng)
        # Prepare the tiny replay key before dispatching GPU work. Otherwise a
        # key split requested after train_fn can queue behind that work and
        # serialize the intended CPU/GPU overlap.
        lookahead_rng = self._next_replay_rng()
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
            train_rng,
            self.spr_weight,
            self._data_augmentation,
            self.dtype,
            self._batch_size,
            self._jumps,
            self.delta_priority_epsilon,
            self.target_update_tau,
            self.target_update_period,
            self.grad_steps,
            self.target_eval_mode,
            self.x_ent_coef,
            self.replay_elements["reward"],
            self.reward_weight,
            self.continue_weight,
            self.reward_readout,
            self.continue_readout,
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
        self.grad_steps += self._batches_to_group
        self.cycle_grad_steps += self._batches_to_group

        # Assemble the next group while the dispatched accelerator work can
        # still be running.  Sampling happens before this group's priority
        # writeback, intentionally introducing one group of ordinary PER lag.
        # The lookahead uses the next group's schedule, including across the
        # smooth 20k endpoint; there is no target-boundary reset or flush.
        self.replay_elements = self._sample_replay_batch(lookahead_rng)
        self.replay_mean_priority = float(self._replay.mean_priority())

        self.imag_return_ema = np.asarray(new_return_ema)

        # Store alpha=.5 delta priorities for the current scheduled C51 target.
        indices = np.reshape(np.asarray(indices), (-1,))
        priority_loss = np.reshape(
            np.asarray(aux_losses["PriorityLoss"]), (-1))
        validate_priority_cardinality(indices, priority_loss)

        priorities = np.sqrt(priority_loss + 1e-10)
        if not np.all(np.isfinite(priorities)):
            raise FloatingPointError("Replay priorities must be finite.")
        validate_priority_cardinality(indices, priorities)
        self._replay.set_priority(indices, priorities)

        if self.grad_steps % 500 < self._batches_to_group:
            log_keys = ("RewardLoss", "ContinueLoss",
                        "ImagActorLoss", "ImagValueLoss",
                        "ImagRet", "ImagReward", "ImagContinue",
                        "ImagEntropy", "ImagTrustClip")
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
                priority=priority,
                episode_end=episode_end,
            )

    def _train_step(self):
        decay_fraction = np.clip(
            (self.x_ent_decay_steps - self.training_steps) /
            max(1, self.x_ent_decay_steps), 0.0, 1.0)
        self.x_ent_coef = max(1e-2 * decay_fraction, self.x_ent_floor)
        if self._replay.add_count > self.min_replay_history:
            if self.training_steps % self.update_period == 0:
                num_updates = self._num_updates_per_train_step
                for _ in range(num_updates):
                    self._training_step_update()
        if self.reset_every > 0 and self.training_steps > self.next_reset:
            self.reset_weights()
        self.training_steps += 1

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
        self.state = np.zeros(
            (n_envs, *self.state_shape), dtype=self.observation_dtype)
        self._record_observation(new_obs)

    def reset_one(self, env_id):
        self.state[env_id].fill(0)

    def delete_one(self, env_id):
        self.state = np.concatenate(
            [self.state[:env_id], self.state[env_id + 1:]], 0)

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
            return action
        self._rng, action = select_action(
            self.network_def,
            self.target_network_params,
            state,
            self._rng,
        )
        return action

    def step(self):
        """Records the most recent transition, returns the agent's next action, and trains if appropriate.
    """
        if not self.eval_mode:
            self._train_step()
        state = self.state
        action = self.select_action(
            state,
            self.eval_mode,
        )
        self.action = np.asarray(action)
        return self.action
