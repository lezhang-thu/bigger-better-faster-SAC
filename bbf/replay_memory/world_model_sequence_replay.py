# coding=utf-8
"""Sequence replay for a recurrent Atari world model.

This module deliberately depends only on NumPy.  In particular, it does not
import the sibling ``r2dreamer`` checkout, Torch, TensorDict, or TorchRL.  The
arrays returned by :meth:`WorldModelSequenceReplayBuffer.sample` can be passed
to JAX with ``jnp.asarray`` (or device-put by the caller).

Row semantics match R2-Dreamer.  A stored row is::

    (observation_t, reward_t, action_t, is_terminal_t, is_first_t, latent_t)

``reward_t`` arrived *with* ``observation_t`` and ``action_t`` is chosen after
observing it.  Consequently a length-T training sequence needs T+1 stored
rows.  The first row supplies only a cached, stop-gradient RSSM context.  The
returned observations/rewards/flags are rows 1..T, while returned actions are
rows 0..T-1.  Training must recompute all T posterior rows and may write those
new posterior states back with :meth:`write_back`.

At a life or game boundary, preserve the final observation and reward in a
dedicated boundary row, then append a separate reset row.  For example::

    buffer.add_boundary_transition(final_state, final_reward, terminal, ...)
    buffer.add_reset_transition(reset_state, first_action, ...)

The boundary row has a zero action sentinel and ``is_first=False``.  The reset
row has ``reward=0``, ``is_terminal=False``, and ``is_first=True``.  Samples
are allowed to cross this pair; the RSSM reset mask makes the cached context
and shifted zero action irrelevant at the reset row.
"""

from __future__ import absolute_import, division, print_function

import numbers

import numpy as np


class StaleReplayIndexError(IndexError):
    """Raised when a sampled logical row has since been overwritten."""


class StaleLatentGenerationError(RuntimeError):
    """Raised when pre-reset posterior results are written after invalidation."""


class WorldModelSequenceReplayBuffer(object):
    """A single-environment NumPy ring buffer for world-model sequences.

    Args:
      capacity: Maximum number of rows retained.
      state_shape: Shape of one Atari state, normally ``(84, 84, stack_size)``.
      num_actions: Number of discrete Atari actions.  Actions are stored as
        one-hot float32 vectors; integer actions passed to :meth:`add` are
        converted automatically.
      stoch_shape: Shape of one cached stochastic posterior, e.g. ``(32, 48)``.
      deter_shape: Shape of one cached deterministic posterior, e.g. ``(6144,)``.
      batch_size: Default B used by :meth:`sample`.
      batch_length: Default T used by :meth:`sample`.
      state_dtype: Storage dtype for stacked Atari states.
      latent_dtype: Storage dtype for cached stochastic/deterministic context.
      seed: Optional seed for uniform sequence sampling.

    This buffer intentionally supports one environment.  Parallel environments
    need independent chronological rings (or an explicit environment axis),
    because flattening their rows would create false temporal adjacency.
    """

    def __init__(self,
                 capacity,
                 state_shape,
                 num_actions,
                 stoch_shape,
                 deter_shape,
                 batch_size,
                 batch_length,
                 state_dtype=np.uint8,
                 latent_dtype=np.float32,
                 seed=None):
        self.capacity = self._positive_int(capacity, 'capacity')
        self.batch_size = self._positive_int(batch_size, 'batch_size')
        self.batch_length = self._positive_int(batch_length, 'batch_length')
        self.num_actions = self._positive_int(num_actions, 'num_actions')
        self.state_shape = self._shape(state_shape, 'state_shape')
        self.stoch_shape = self._shape(stoch_shape, 'stoch_shape')
        self.deter_shape = self._shape(deter_shape, 'deter_shape')
        if self.capacity < self.batch_length + 1:
            raise ValueError(
                'capacity must hold at least batch_length + 1 rows; got {} and '
                '{}'.format(self.capacity, self.batch_length))

        self.state_dtype = np.dtype(state_dtype)
        self.latent_dtype = np.dtype(latent_dtype)
        self._state = np.empty((self.capacity,) + self.state_shape,
                               dtype=self.state_dtype)
        self._action = np.empty((self.capacity, self.num_actions),
                                dtype=np.float32)
        self._reward = np.empty((self.capacity, 1), dtype=np.float32)
        self._is_terminal = np.empty((self.capacity, 1), dtype=np.bool_)
        self._is_first = np.empty((self.capacity, 1), dtype=np.bool_)
        self._stoch = np.empty((self.capacity,) + self.stoch_shape,
                               dtype=self.latent_dtype)
        self._deter = np.empty((self.capacity,) + self.deter_shape,
                               dtype=self.latent_dtype)

        # Logical IDs make wrap-around windows simple and make write-back safe:
        # a physical slot is updated only if it still contains the sampled row.
        self._row_id = np.full((self.capacity,), -1, dtype=np.int64)
        self._latent_generation = np.full((self.capacity,), -1, dtype=np.int64)
        self._size = 0
        self._total_added = 0
        self._generation = 0
        self._invalid_latent_count = 0
        self._expect_first = False
        self._rng = np.random.default_rng(seed)

    @staticmethod
    def _positive_int(value, name):
        value = int(value)
        if value <= 0:
            raise ValueError('{} must be positive, got {}'.format(name, value))
        return value

    @staticmethod
    def _shape(value, name):
        if isinstance(value, numbers.Integral):
            value = (int(value),)
        else:
            value = tuple(int(dim) for dim in value)
        if not value or any(dim <= 0 for dim in value):
            raise ValueError('{} must contain positive dimensions, got {}'.format(
                name, value))
        return value

    @property
    def total_added(self):
        """Total rows ever appended, including rows later overwritten."""
        return self._total_added

    @property
    def oldest_row_id(self):
        """Logical ID of the oldest retained row, or None when empty."""
        if not self._size:
            return None
        return self._total_added - self._size

    @property
    def newest_row_id(self):
        """Logical ID of the newest retained row, or None when empty."""
        if not self._size:
            return None
        return self._total_added - 1

    @property
    def latent_generation(self):
        """Current generation of cached posterior states."""
        return self._generation

    def __len__(self):
        return self._size

    def count(self):
        """Returns the number of currently retained rows."""
        return self._size

    def can_sample(self, batch_length=None):
        """Whether at least one T+1 window has a valid cached context."""
        batch_length = self._resolve_batch_length(batch_length)
        if self._size < batch_length + 1:
            return False
        if not self._invalid_latent_count:
            return True
        first_start = self._total_added - self._size
        last_start = self._total_added - batch_length - 1
        starts = np.arange(first_start, last_start + 1, dtype=np.int64)
        slots = np.remainder(starts, self.capacity)
        return bool(np.any(self._latent_generation[slots] == self._generation))

    def invalidate_latents(self, clear=False):
        """Invalidates every retained cached posterior after a model reset.

        Sequence data remain available, but :meth:`sample` will not use an old
        posterior as context.  Newly collected rows and rows refreshed by
        :meth:`write_back` belong to the new generation.  Until a valid context
        has T following rows, sampling raises instead of silently treating a
        stale or zero vector as a posterior.

        Args:
          clear: Also overwrite retained latent arrays with zeros.  This is
            normally unnecessary because generation checks prevent access and
            leaving it False makes invalidation O(1).

        Returns:
          The new integer generation token.
        """
        self._generation += 1
        self._invalid_latent_count = self._size
        if clear and self._size:
            row_ids = np.arange(self._total_added - self._size,
                                self._total_added,
                                dtype=np.int64)
            slots = np.remainder(row_ids, self.capacity)
            self._stoch[slots] = 0
            self._deter[slots] = 0
        return self._generation

    def _resolve_batch_length(self, batch_length):
        if batch_length is None:
            return self.batch_length
        return self._positive_int(batch_length, 'batch_length')

    def _resolve_batch_size(self, batch_size):
        if batch_size is None:
            return self.batch_size
        return self._positive_int(batch_size, 'batch_size')

    @staticmethod
    def _scalar(value, name, dtype):
        array = np.asarray(value)
        if array.size != 1:
            raise ValueError('{} must be scalar, got shape {}'.format(
                name, array.shape))
        return np.asarray(array.reshape(()), dtype=dtype).item()

    @staticmethod
    def _array(value, shape, dtype, name):
        array = np.asarray(value, dtype=dtype)
        if array.shape != shape:
            raise ValueError('{} must have shape {}, got {}'.format(
                name, shape, array.shape))
        return array

    def _one_hot_action(self, action):
        action_array = np.asarray(action)
        if action_array.size == 1 and action_array.shape != (self.num_actions,):
            scalar_action = action_array.reshape(()).item()
            if (not isinstance(scalar_action, numbers.Integral) and
                    (not np.isfinite(scalar_action) or
                     not float(scalar_action).is_integer())):
                raise ValueError('scalar action must be an integer index')
            action_index = int(scalar_action)
            if action_index < 0 or action_index >= self.num_actions:
                raise ValueError('action index {} outside [0, {})'.format(
                    action_index, self.num_actions))
            result = np.zeros((self.num_actions,), dtype=np.float32)
            result[action_index] = 1.0
            return result

        result = self._array(action, (self.num_actions,), np.float32,
                             'action')
        if not np.all(np.isfinite(result)):
            raise ValueError('action must contain finite values')
        is_zero = np.allclose(result, 0.0)
        is_one_hot = (np.all(np.logical_or(np.isclose(result, 0.0),
                                           np.isclose(result, 1.0))) and
                      np.isclose(result.sum(), 1.0))
        if not (is_zero or is_one_hot):
            raise ValueError(
                'action must be one-hot, or all-zero for a boundary row')
        return result

    def add(self,
            state,
            action,
            reward,
            is_terminal,
            is_first,
            stoch,
            deter):
        """Appends one chronological row and returns its logical row ID.

        Integer actions are converted to one-hot.  An all-zero action is a
        boundary sentinel and requires the following appended row to have
        ``is_first=True``.  ``is_terminal=True`` imposes the same requirement.

        A first row must have zero arrival reward and cannot also be terminal.
        These checks prevent the failure fixed by commit 581e345, where the
        terminal observation/reward and reset observation were collapsed into
        one contradictory row.
        """
        state = self._array(state, self.state_shape, self.state_dtype, 'state')
        action = self._one_hot_action(action)
        reward = self._scalar(reward, 'reward', np.float32)
        is_terminal = bool(self._scalar(is_terminal, 'is_terminal', np.bool_))
        is_first = bool(self._scalar(is_first, 'is_first', np.bool_))
        stoch = self._array(stoch, self.stoch_shape, self.latent_dtype,
                            'stoch')
        deter = self._array(deter, self.deter_shape, self.latent_dtype,
                            'deter')

        if is_terminal and is_first:
            raise ValueError(
                'a terminal observation and a reset observation must be '
                'stored as separate rows')
        if is_first and not np.isclose(reward, 0.0):
            raise ValueError(
                'an is_first row must have zero arrival reward; preserve the '
                'terminal reward in a preceding boundary row')
        if self._expect_first and not is_first:
            raise ValueError(
                'the previous row ended a life/game segment; the next row '
                'must be a separate is_first reset row')

        row_id = self._total_added
        slot = row_id % self.capacity
        if (self._row_id[slot] >= 0 and
                self._latent_generation[slot] != self._generation):
            self._invalid_latent_count -= 1
        self._state[slot] = state
        self._action[slot] = action
        self._reward[slot, 0] = reward
        self._is_terminal[slot, 0] = is_terminal
        self._is_first[slot, 0] = is_first
        self._stoch[slot] = stoch
        self._deter[slot] = deter
        self._row_id[slot] = row_id
        self._latent_generation[slot] = self._generation

        self._total_added += 1
        self._size = min(self._size + 1, self.capacity)
        # A zero vector is not a valid Atari action, so it unambiguously marks a
        # final observation for which no next action will be executed.
        zero_action = bool(np.allclose(action, 0.0))
        self._expect_first = bool(is_terminal or zero_action)
        if is_first:
            self._expect_first = False
        return row_id

    def add_atari_transition(self,
                              state,
                              action,
                              reward,
                              is_terminal,
                              is_first,
                              stoch,
                              deter):
        """Compatibility wrapper accepting either unbatched or leading-B=1 data.

        This remains a single-environment buffer.  A leading batch dimension
        larger than one is rejected rather than silently interleaved.
        """
        state = self._remove_singleton_batch(state, self.state_shape, 'state')
        action_array = np.asarray(action)
        if action_array.shape == (1, self.num_actions):
            action = action_array[0]
        elif action_array.shape == (1,):
            action = action_array[0]
        stoch = self._remove_singleton_batch(stoch, self.stoch_shape, 'stoch')
        deter = self._remove_singleton_batch(deter, self.deter_shape, 'deter')
        return self.add(state, action, reward, is_terminal, is_first, stoch,
                        deter)

    @staticmethod
    def _remove_singleton_batch(value, item_shape, name):
        array = np.asarray(value)
        if array.shape == (1,) + item_shape:
            return array[0]
        if array.shape == item_shape:
            return array
        if array.ndim == len(item_shape) + 1:
            raise ValueError(
                '{} has leading batch size {}; only one environment is '
                'supported'.format(name, array.shape[0]))
        return array

    def add_boundary_transition(self,
                                state,
                                reward,
                                is_terminal,
                                stoch,
                                deter):
        """Appends a life/game-ending observation as a reward-bearing row.

        ``is_terminal`` distinguishes a true MDP terminal from a truncation.
        Both kinds still require a subsequent ``is_first`` row, which is
        enforced by the zero action sentinel.
        """
        return self.add_atari_transition(
            state=state,
            action=np.zeros((self.num_actions,), dtype=np.float32),
            reward=reward,
            is_terminal=is_terminal,
            is_first=False,
            stoch=stoch,
            deter=deter)

    def add_reset_transition(self, state, action, stoch, deter):
        """Appends the separate zero-reward first row after a boundary."""
        return self.add_atari_transition(
            state=state,
            action=action,
            reward=0.0,
            is_terminal=False,
            is_first=True,
            stoch=stoch,
            deter=deter)

    def add_initial(self, state, action, stoch, deter):
        """Alias for :meth:`add_reset_transition` used at environment reset."""
        return self.add_reset_transition(state, action, stoch, deter)

    def sample(self, batch_size=None, batch_length=None, rng=None):
        """Uniformly samples B chronological T+1 windows and R2-aligns them.

        Every retained start row whose context latent belongs to the current
        generation has equal probability; sampling is with replacement.  In
        the usual case (no model reset), this is every valid chronological
        start.  The returned dictionary contains NumPy arrays:

        ``state``:
          Rows 1..T, shape ``(B, T) + state_shape``.
        ``action``:
          Rows 0..T-1, shape ``(B, T, num_actions)``.  This one-step shift is
          the action that causes the corresponding returned state.
        ``reward``, ``is_terminal``, ``is_first``:
          Rows 1..T, with reward-on-arrival semantics and shape ``(B, T, 1)``.
        ``initial_stoch``, ``initial_deter``:
          Cached row-0 context only.  Cached latents for rows 1..T are not
          returned; those posterior states must be recomputed by training.
        ``row_ids``:
          Stable logical IDs for rows 1..T, shape ``(B, T)``.  Pass these with
          recomputed posteriors to :meth:`write_back`.
        ``context_row_ids``:
          Logical IDs of the cached row-0 contexts, shape ``(B,)``.
        ``generation``:
          Scalar latent-generation token.  Pass it to :meth:`write_back` so an
          in-flight pre-reset batch cannot contaminate post-reset caches.
        """
        batch_size = self._resolve_batch_size(batch_size)
        batch_length = self._resolve_batch_length(batch_length)
        window_length = batch_length + 1
        if self._size < window_length:
            raise RuntimeError(
                'need at least {} rows to sample, have {}'.format(
                    window_length, self._size))

        first_start = self._total_added - self._size
        last_start = self._total_added - window_length
        sample_rng = self._rng if rng is None else rng
        if isinstance(sample_rng, numbers.Integral):
            sample_rng = np.random.default_rng(int(sample_rng))
        if not hasattr(sample_rng, 'integers'):
            raise TypeError('rng must provide numpy Generator.integers')
        if not self._invalid_latent_count:
            starts = np.asarray(
                sample_rng.integers(first_start,
                                    last_start + 1,
                                    size=batch_size),
                dtype=np.int64)
        else:
            all_starts = np.arange(first_start,
                                   last_start + 1,
                                   dtype=np.int64)
            all_slots = np.remainder(all_starts, self.capacity)
            valid = self._latent_generation[all_slots] == self._generation
            eligible_starts = all_starts[valid]
            if not eligible_starts.size:
                raise RuntimeError(
                    'no sampleable sequence has a current-generation cached '
                    'context; collect or refresh more rows after invalidation')
            selected = np.asarray(
                sample_rng.integers(0, eligible_starts.size, size=batch_size),
                dtype=np.int64)
            starts = eligible_starts[selected]
        if starts.shape != (batch_size,):
            raise ValueError('rng returned starts with shape {}, expected {}'.format(
                starts.shape, (batch_size,)))

        window_ids = starts[:, None] + np.arange(window_length,
                                                  dtype=np.int64)[None, :]
        slots = np.remainder(window_ids, self.capacity)
        if not np.array_equal(self._row_id[slots], window_ids):
            raise RuntimeError('internal replay chronology is inconsistent')

        context_slots = slots[:, 0]
        data_slots = slots[:, 1:]
        action_slots = slots[:, :-1]
        return {
            'state': self._state[data_slots].copy(),
            'action': self._action[action_slots].copy(),
            'reward': self._reward[data_slots].copy(),
            'is_terminal': self._is_terminal[data_slots].copy(),
            'is_first': self._is_first[data_slots].copy(),
            'initial_stoch': self._stoch[context_slots].copy(),
            'initial_deter': self._deter[context_slots].copy(),
            'row_ids': window_ids[:, 1:].copy(),
            'context_row_ids': window_ids[:, 0].copy(),
            'generation': np.asarray(self._generation, dtype=np.int64),
        }

    def write_back(self,
                   row_ids,
                   post_stoch,
                   post_deter,
                   ignore_stale=False,
                   generation=None):
        """Writes recomputed posterior states into their sampled replay rows.

        Args:
          row_ids: Integer array with leading shape P, normally ``(B, T)``.
          post_stoch: Recomputed current-model posteriors, shape
            ``P + stoch_shape``.
          post_deter: Recomputed current-model posteriors, shape
            ``P + deter_shape``.
          ignore_stale: If False, raise :class:`StaleReplayIndexError` when any
            sampled row has been overwritten.  If True, update only retained
            rows and return their count.
          generation: Optional scalar token returned by :meth:`sample`.  A
            token from before :meth:`invalidate_latents` is rejected even when
            the physical replay rows still exist.

        Duplicate row IDs can arise from overlapping sampled windows.  They are
        applied in flattened batch order, so the last occurrence wins
        deterministically.

        Returns:
          Number of retained rows written (including duplicate occurrences).
        """
        row_ids = np.asarray(row_ids)
        if generation is not None:
            generation = self._scalar(generation, 'generation', np.int64)
            if generation != self._generation:
                raise StaleLatentGenerationError(
                    'posterior batch generation {} does not match current '
                    'generation {}'.format(generation, self._generation))
        if not np.issubdtype(row_ids.dtype, np.integer):
            raise ValueError('row_ids must have an integer dtype')
        leading_shape = row_ids.shape
        post_stoch = np.asarray(post_stoch, dtype=self.latent_dtype)
        post_deter = np.asarray(post_deter, dtype=self.latent_dtype)
        expected_stoch = leading_shape + self.stoch_shape
        expected_deter = leading_shape + self.deter_shape
        if post_stoch.shape != expected_stoch:
            raise ValueError('post_stoch must have shape {}, got {}'.format(
                expected_stoch, post_stoch.shape))
        if post_deter.shape != expected_deter:
            raise ValueError('post_deter must have shape {}, got {}'.format(
                expected_deter, post_deter.shape))

        flat_ids = row_ids.astype(np.int64, copy=False).reshape(-1)
        flat_stoch = post_stoch.reshape((-1,) + self.stoch_shape)
        flat_deter = post_deter.reshape((-1,) + self.deter_shape)
        slots = np.remainder(flat_ids, self.capacity)
        retained = np.logical_and(flat_ids >= 0,
                                  self._row_id[slots] == flat_ids)
        if not np.all(retained) and not ignore_stale:
            stale = flat_ids[np.logical_not(retained)]
            raise StaleReplayIndexError(
                'sampled replay rows were overwritten before write-back: {}'.format(
                    stale.tolist()))

        written = 0
        for slot, stoch, deter, keep in zip(slots, flat_stoch, flat_deter,
                                             retained):
            if not keep:
                continue
            self._stoch[slot] = stoch
            self._deter[slot] = deter
            if self._latent_generation[slot] != self._generation:
                self._latent_generation[slot] = self._generation
                self._invalid_latent_count -= 1
            written += 1
        return written

    def update(self,
               row_ids,
               post_stoch,
               post_deter,
               ignore_stale=False,
               generation=None):
        """Compatibility alias for :meth:`write_back`."""
        return self.write_back(
            row_ids=row_ids,
            post_stoch=post_stoch,
            post_deter=post_deter,
            ignore_stale=ignore_stale,
            generation=generation)


# Short alias for call sites that do not need the full descriptive class name.
WorldModelSequenceReplay = WorldModelSequenceReplayBuffer
