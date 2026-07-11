# coding=utf-8
# Copyright 2026 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A discrete recurrent state-space model for Dreamer-style imagination.

The module uses batch-major sequences at its public ``observe`` boundary:

* ``embed`` has shape ``[..., time, embed_size]``.
* ``prev_actions`` has shape ``[..., time, action_size]``.  The action at
  index ``t`` led to the observation whose embedding is at index ``t``.
* ``is_first`` has shape ``[..., time]`` and resets the latent state and the
  previous action before processing that observation.

An :class:`RSSMState` stores deterministic features ``[..., deter_size]`` and
``stoch_size`` independent categorical variables, each with ``discrete_size``
classes.  Its stochastic state and logits therefore both have shape
``[..., stoch_size, discrete_size]``.  State objects are Flax pytrees and can
be carried by ``jax.lax.scan``, sliced, vmapped, and jitted.

All stochastic public methods use Flax's ``"sample"`` RNG collection.  For
example, call ``module.apply(variables, ..., rngs={"sample": key})``.
"""

from typing import Any, Sequence, Tuple, Union

from flax import linen as nn
from flax import struct
import jax
import jax.numpy as jnp


Array = Any
BatchShape = Union[int, Sequence[int]]


def _sample_key(module: nn.Module, deterministic: bool) -> Array:
    """Returns a dummy key for argmax paths without requiring an RNG stream."""
    if deterministic:
        return jax.random.PRNGKey(0)
    return module.make_rng('sample')


@struct.dataclass
class RSSMState:
    """Latent state of a discrete RSSM.

    Attributes:
      deter: Deterministic recurrent state with shape ``[..., D]``.
      stoch: Straight-through one-hot samples with shape ``[..., S, K]``.
      logits: Uniform-mixed categorical logits with shape ``[..., S, K]``.
    """

    deter: Array
    stoch: Array
    logits: Array


def unimix_logits(logits: Array, unimix: float = 0.01) -> Array:
    """Mixes categorical probabilities with uniform mass and returns logits.

    Mixing probabilities, rather than adding a constant to logits, guarantees
    every class probability is at least ``unimix / num_classes``.  The result
    is normalized log-probability and can still be passed anywhere logits are
    expected.

    Args:
      logits: Categorical logits with classes on the final axis.
      unimix: Fraction of uniform probability mass to mix in.

    Returns:
      Log probabilities with the same shape as ``logits``.
    """
    if not 0.0 <= unimix <= 1.0:
        raise ValueError(f'unimix must be in [0, 1], got {unimix}.')
    if logits.shape[-1] < 1:
        raise ValueError('Categorical logits must have at least one class.')
    probs = jax.nn.softmax(logits, axis=-1)
    probs = ((1.0 - unimix) * probs +
             unimix / jnp.asarray(logits.shape[-1], logits.dtype))
    return jnp.log(probs)


def straight_through_one_hot(logits: Array,
                             key: Array,
                             deterministic: bool = False) -> Array:
    """Draws a one-hot categorical sample with straight-through gradients.

    The forward value is exactly one-hot.  Its backward derivative is the
    derivative of the categorical probabilities, which lets representation
    and dynamics losses train the logits that produced a sampled state.
    """
    sample_index = jax.lax.cond(
        jnp.asarray(deterministic),
        lambda _: jnp.argmax(logits, axis=-1),
        lambda rng: jax.random.categorical(rng, logits, axis=-1),
        key,
    )
    sample = jax.nn.one_hot(
        sample_index, logits.shape[-1], dtype=logits.dtype)
    probs = jax.nn.softmax(logits, axis=-1)
    return probs + jax.lax.stop_gradient(sample - probs)


def categorical_kl(lhs_logits: Array, rhs_logits: Array) -> Array:
    """KL(lhs || rhs), summed over classes and stochastic variables.

    Inputs have shape ``[..., S, K]`` and the result has shape ``[...]``.
    Both arguments are treated as logits; normalized log probabilities (as
    returned by :func:`unimix_logits`) are also valid logits.
    """
    if lhs_logits.shape != rhs_logits.shape:
        raise ValueError(
            'KL arguments must have equal shape, got '
            f'{lhs_logits.shape} and {rhs_logits.shape}.')
    if lhs_logits.ndim < 2:
        raise ValueError('KL logits must have shape [..., stoch, discrete].')
    lhs_log_probs = jax.nn.log_softmax(lhs_logits, axis=-1)
    rhs_log_probs = jax.nn.log_softmax(rhs_logits, axis=-1)
    lhs_probs = jnp.exp(lhs_log_probs)
    per_variable = jnp.sum(
        lhs_probs * (lhs_log_probs - rhs_log_probs), axis=-1)
    return jnp.sum(per_variable, axis=-1)


def balanced_kl_loss(post_logits: Array,
                     prior_logits: Array,
                     free_nats: float = 1.0,
                     balance: float = 0.8) -> Tuple[Array, Array, Array]:
    """Computes balanced Dreamer dynamics and representation KL losses.

    The dynamics term stops gradients through the posterior and trains the
    prior.  The representation term stops gradients through the prior and
    trains the posterior.  Free nats are applied separately to both terms.
    No batch or time reduction is performed.

    Args:
      post_logits: Posterior logits with shape ``[..., S, K]``.
      prior_logits: Prior logits with shape ``[..., S, K]``.
      free_nats: Per-state lower bound applied to each KL term.
      balance: Weight of the dynamics term.  The representation weight is
        ``1 - balance``.

    Returns:
      ``(balanced, dynamics, representation)``, each with shape ``[...]``.
    """
    dynamics = categorical_kl(
        jax.lax.stop_gradient(post_logits), prior_logits)
    representation = categorical_kl(
        post_logits, jax.lax.stop_gradient(prior_logits))
    free_nats = jnp.asarray(free_nats, dtype=dynamics.dtype)
    dynamics = jnp.maximum(dynamics, free_nats)
    representation = jnp.maximum(representation, free_nats)
    balance = jnp.asarray(balance, dtype=dynamics.dtype)
    balanced = balance * dynamics + (1.0 - balance) * representation
    return balanced, dynamics, representation


class _GRUCell(nn.Module):
    """Small GRU-style recurrent cell with a Dreamer-compatible update bias."""

    features: int

    @nn.compact
    def __call__(self, carry: Array, inputs: Array) -> Array:
        joined = jnp.concatenate([carry, inputs], axis=-1)
        gates = nn.Dense(2 * self.features, name='gates')(joined)
        reset, update = jnp.split(gates, 2, axis=-1)
        reset = jax.nn.sigmoid(reset)
        # A negative update bias initially favors retaining the recurrent
        # state, as in DreamerV3's deterministic transition.
        update = jax.nn.sigmoid(update - 1.0)
        candidate_inputs = jnp.concatenate(
            [reset * carry, inputs], axis=-1)
        candidate = jnp.tanh(
            nn.Dense(self.features, name='candidate')(candidate_inputs))
        return update * candidate + (1.0 - update) * carry


class RSSM(nn.Module):
    """Discrete recurrent state-space model.

    ``initial_mode="zeros"`` matches R2-Dreamer's zero initial state.
    ``initial_mode="learned"`` learns the initial deterministic state and
    samples its stochastic state from the shared prior network.
    """

    stoch_size: int = 32
    discrete_size: int = 32
    deter_size: int = 512
    hidden_size: int = 512
    initial_mode: str = 'zeros'
    unimix: float = 0.01
    free_nats: float = 1.0
    kl_balance: float = 0.8

    def setup(self):
        if self.stoch_size < 1 or self.discrete_size < 1:
            raise ValueError('stoch_size and discrete_size must be positive.')
        if self.deter_size < 1 or self.hidden_size < 1:
            raise ValueError('deter_size and hidden_size must be positive.')
        if self.initial_mode not in ('zeros', 'learned'):
            raise ValueError(
                'initial_mode must be "zeros" or "learned", got '
                f'{self.initial_mode!r}.')
        if not 0.0 <= self.unimix <= 1.0:
            raise ValueError(f'unimix must be in [0, 1], got {self.unimix}.')
        if not 0.0 <= self.kl_balance <= 1.0:
            raise ValueError(
                f'kl_balance must be in [0, 1], got {self.kl_balance}.')
        if self.free_nats < 0.0:
            raise ValueError('free_nats must be nonnegative.')

        self._dynamics_input = nn.Dense(
            self.hidden_size, name='dynamics_input')
        self._dynamics_norm = nn.LayerNorm(name='dynamics_norm')
        self._gru = _GRUCell(self.deter_size, name='gru')

        self._prior_hidden = nn.Dense(self.hidden_size, name='prior_hidden')
        self._prior_norm = nn.LayerNorm(name='prior_norm')
        self._prior_output = nn.Dense(
            self.stoch_size * self.discrete_size, name='prior_output')

        self._posterior_hidden = nn.Dense(
            self.hidden_size, name='posterior_hidden')
        self._posterior_norm = nn.LayerNorm(name='posterior_norm')
        self._posterior_output = nn.Dense(
            self.stoch_size * self.discrete_size, name='posterior_output')
        if self.initial_mode == 'learned':
            self._initial_deter = self.param(
                'initial_deter', nn.initializers.zeros, (self.deter_size,))

    @property
    def feat_size(self) -> int:
        return self.deter_size + self.stoch_size * self.discrete_size

    def _sample(self, logits: Array, key: Array,
                deterministic: bool) -> Array:
        return straight_through_one_hot(logits, key, deterministic)

    def _prior_logits(self, deter: Array) -> Array:
        hidden = self._prior_hidden(deter)
        hidden = jax.nn.silu(self._prior_norm(hidden))
        logits = self._prior_output(hidden)
        logits = logits.reshape(
            deter.shape[:-1] + (self.stoch_size, self.discrete_size))
        return unimix_logits(logits, self.unimix)

    def _posterior_logits(self, deter: Array, embed: Array) -> Array:
        hidden = jnp.concatenate([deter, embed], axis=-1)
        hidden = self._posterior_hidden(hidden)
        hidden = jax.nn.silu(self._posterior_norm(hidden))
        logits = self._posterior_output(hidden)
        logits = logits.reshape(
            deter.shape[:-1] + (self.stoch_size, self.discrete_size))
        return unimix_logits(logits, self.unimix)

    def _initial(self, batch_shape: Tuple[int, ...], key: Array,
                 deterministic: bool) -> RSSMState:
        shape = batch_shape + (self.deter_size,)
        if self.initial_mode == 'zeros':
            deter = jnp.zeros(shape, jnp.float32)
            stoch_shape = batch_shape + (
                self.stoch_size, self.discrete_size)
            return RSSMState(
                deter=deter,
                stoch=jnp.zeros(stoch_shape, jnp.float32),
                logits=jnp.zeros(stoch_shape, jnp.float32),
            )

        initial_deter = jnp.tanh(self._initial_deter)
        deter = jnp.broadcast_to(initial_deter, shape)
        logits = self._prior_logits(deter)
        stoch = self._sample(logits, key, deterministic)
        return RSSMState(deter=deter, stoch=stoch, logits=logits)

    def initial(self,
                batch_size: BatchShape,
                deterministic: bool = False) -> RSSMState:
        """Returns an initial state for an int or tuple batch shape."""
        if isinstance(batch_size, int):
            batch_shape = (batch_size,)
        else:
            batch_shape = tuple(batch_size)
        # Zero initialization does not need randomness.  Keeping the dummy key
        # local lets callers use ``initial`` without supplying an RNG in the
        # default mode.
        key = (jax.random.PRNGKey(0) if self.initial_mode == 'zeros' else
               _sample_key(self, deterministic))
        return self._initial(batch_shape, key, deterministic)

    def _deter_step(self, prev_state: RSSMState, prev_action: Array) -> Array:
        stoch = prev_state.stoch.reshape(
            prev_state.stoch.shape[:-2] +
            (self.stoch_size * self.discrete_size,))
        prev_action = prev_action.astype(prev_state.deter.dtype)
        action_scale = jax.lax.stop_gradient(
            jnp.maximum(1.0, jnp.abs(prev_action)))
        prev_action = prev_action / action_scale
        hidden = jnp.concatenate([stoch, prev_action], axis=-1)
        hidden = self._dynamics_input(hidden)
        hidden = jax.nn.silu(self._dynamics_norm(hidden))
        return self._gru(prev_state.deter, hidden)

    def _prior(self, deter: Array, key: Array,
               deterministic: bool) -> RSSMState:
        logits = self._prior_logits(deter)
        stoch = self._sample(logits, key, deterministic)
        return RSSMState(deter=deter, stoch=stoch, logits=logits)

    def _posterior(self, deter: Array, embed: Array, key: Array,
                   deterministic: bool) -> RSSMState:
        logits = self._posterior_logits(deter, embed)
        stoch = self._sample(logits, key, deterministic)
        return RSSMState(deter=deter, stoch=stoch, logits=logits)

    def prior(self, deter: Array,
              deterministic: bool = False) -> RSSMState:
        """Samples a prior state for deterministic features ``[..., D]``."""
        return self._prior(
            deter, _sample_key(self, deterministic), deterministic)

    def posterior(self,
                  deter: Array,
                  embed: Array,
                  deterministic: bool = False) -> RSSMState:
        """Samples a posterior conditioned on ``deter`` and ``embed``."""
        return self._posterior(
            deter, embed, _sample_key(self, deterministic), deterministic)

    def _img_step(self, prev_state: RSSMState, prev_action: Array, key: Array,
                  deterministic: bool) -> RSSMState:
        deter = self._deter_step(prev_state, prev_action)
        return self._prior(deter, key, deterministic)

    def img_step(self,
                 prev_state: RSSMState,
                 prev_action: Array,
                 deterministic: bool = False) -> RSSMState:
        """Advances the prior one step using an action with shape ``[..., A]``."""
        return self._img_step(
            prev_state, prev_action, _sample_key(self, deterministic),
            deterministic)

    @staticmethod
    def _reset_where(is_first: Array, initial: RSSMState,
                     state: RSSMState) -> RSSMState:
        is_first = is_first.astype(jnp.bool_)
        return RSSMState(
            deter=jnp.where(is_first[..., None], initial.deter, state.deter),
            stoch=jnp.where(
                is_first[..., None, None], initial.stoch, state.stoch),
            logits=jnp.where(
                is_first[..., None, None], initial.logits, state.logits),
        )

    def _obs_step(self, prev_state: RSSMState, prev_action: Array,
                  embed: Array, is_first: Array, key: Array,
                  deterministic: bool) -> Tuple[RSSMState, RSSMState]:
        initial_key, prior_key, posterior_key = jax.random.split(key, 3)
        initial = self._initial(
            tuple(prev_state.deter.shape[:-1]), initial_key, deterministic)
        prev_state = self._reset_where(is_first, initial, prev_state)
        prev_action = jnp.where(
            is_first[..., None], jnp.zeros_like(prev_action), prev_action)
        prior = self._img_step(
            prev_state, prev_action, prior_key, deterministic)
        post = self._posterior(
            prior.deter, embed, posterior_key, deterministic)
        return post, prior

    def obs_step(self,
                 prev_state: RSSMState,
                 prev_action: Array,
                 embed: Array,
                 is_first: Array,
                 deterministic: bool = False
                 ) -> Tuple[RSSMState, RSSMState]:
        """Performs one posterior update and also returns its matching prior."""
        return self._obs_step(
            prev_state, prev_action, embed, is_first,
            _sample_key(self, deterministic), deterministic)

    def observe(self,
                embed: Array,
                prev_actions: Array,
                is_first: Array,
                deterministic: bool = False
                ) -> Tuple[RSSMState, RSSMState]:
        """Runs posterior inference over a batch-major sequence.

        Returns:
          ``(post, prior)`` states.  Every state field preserves all leading
          batch dimensions and has time immediately before its feature axes,
          e.g. posterior ``deter`` is ``[..., T, D]`` and ``stoch`` is
          ``[..., T, S, K]``.
        """
        if embed.ndim < 2 or prev_actions.ndim < 2:
            raise ValueError(
                'embed and prev_actions must include time and feature axes.')
        if embed.shape[:-1] != prev_actions.shape[:-1]:
            raise ValueError(
                'embed and prev_actions leading shapes must match, got '
                f'{embed.shape[:-1]} and {prev_actions.shape[:-1]}.')
        if is_first.shape != embed.shape[:-1]:
            raise ValueError(
                f'is_first must have shape {embed.shape[:-1]}, got '
                f'{is_first.shape}.')

        batch_shape = tuple(embed.shape[:-2])
        time_steps = embed.shape[-2]
        if time_steps < 1:
            raise ValueError('observe requires a sequence with at least one step.')
        all_keys = jax.random.split(
            _sample_key(self, deterministic), time_steps + 1)
        initial = self._initial(batch_shape, all_keys[0], deterministic)
        scan_inputs = (
            jnp.moveaxis(prev_actions, -2, 0),
            jnp.moveaxis(embed, -2, 0),
            jnp.moveaxis(is_first, -1, 0),
            all_keys[1:],
        )

        def scan_step(prev_state, inputs):
            action, step_embed, step_is_first, key = inputs
            post, prior = self._obs_step(
                prev_state, action, step_embed, step_is_first, key,
                deterministic)
            return post, (post, prior)

        # Initialize every shared submodule on the first step outside raw
        # lax.scan; creating Flax parameters for the first time within the
        # transform would leak initialization tracers from its scope.
        first_inputs = jax.tree_util.tree_map(lambda value: value[0], scan_inputs)
        first_post, (first_post_output, first_prior) = scan_step(
            initial, first_inputs)
        rest_inputs = jax.tree_util.tree_map(lambda value: value[1:], scan_inputs)
        _, (rest_posts, rest_priors) = jax.lax.scan(
            scan_step, first_post, rest_inputs)

        def prepend(first, rest):
            return jnp.concatenate([first[None], rest], axis=0)

        posts = jax.tree_util.tree_map(
            prepend, first_post_output, rest_posts)
        priors = jax.tree_util.tree_map(prepend, first_prior, rest_priors)
        time_axis = len(batch_shape)
        posts = jax.tree_util.tree_map(
            lambda value: jnp.moveaxis(value, 0, time_axis), posts)
        priors = jax.tree_util.tree_map(
            lambda value: jnp.moveaxis(value, 0, time_axis), priors)
        return posts, priors

    def __call__(self,
                 embed: Array,
                 prev_actions: Array,
                 is_first: Array,
                 deterministic: bool = False
                 ) -> Tuple[RSSMState, RSSMState]:
        return self.observe(embed, prev_actions, is_first, deterministic)

    def get_feat(self, state_or_stoch: Union[RSSMState, Array],
                 deter: Array = None) -> Array:
        """Flattens stochastic variables and concatenates deterministic state.

        Accepts either ``get_feat(state)`` or ``get_feat(stoch, deter)``.
        """
        if isinstance(state_or_stoch, RSSMState):
            if deter is not None:
                raise ValueError(
                    'Do not pass deter when the first argument is a state.')
            stoch = state_or_stoch.stoch
            deter = state_or_stoch.deter
        else:
            if deter is None:
                raise ValueError('deter is required when passing stoch directly.')
            stoch = state_or_stoch
        stoch = stoch.reshape(
            stoch.shape[:-2] + (self.stoch_size * self.discrete_size,))
        return jnp.concatenate([stoch, deter], axis=-1)

    def kl_loss(self,
                post: Union[RSSMState, Array],
                prior: Union[RSSMState, Array],
                free_nats: float = None,
                balance: float = None) -> Tuple[Array, Array, Array]:
        """Returns balanced, dynamics, and representation KL per state."""
        post_logits = post.logits if isinstance(post, RSSMState) else post
        prior_logits = prior.logits if isinstance(prior, RSSMState) else prior
        if free_nats is None:
            free_nats = self.free_nats
        if balance is None:
            balance = self.kl_balance
        return balanced_kl_loss(
            post_logits, prior_logits, free_nats=free_nats, balance=balance)
