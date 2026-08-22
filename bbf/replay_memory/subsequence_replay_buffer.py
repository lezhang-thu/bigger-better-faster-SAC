# coding=utf-8
"""Compact prioritized replay for scheduled n-step TD targets."""

import collections

from absl import logging
import gin
import jax
import numpy as np

from bbf.replay_memory import deterministic_sum_tree as sum_tree

ReplayElement = collections.namedtuple('ReplayElement', ['name', 'shape', 'type'])


def modulo_range(start, length, modulo):
    for offset in range(length):
        yield (start + offset) % modulo


def invalid_range(cursor, replay_length, stack_history, future_rows):
    """Returns anchors whose stack or future window crosses the cursor."""
    return np.asarray([
        (cursor - future_rows + offset) % replay_length
        for offset in range(stack_history + future_rows)
    ])


@gin.configurable
class PrioritizedJaxSubsequenceParallelEnvReplayBuffer(object):
    """Prioritized circular replay with compact root-only TD fields.

    The replay stores one observation frame per transition and reconstructs
    stacked states when sampled. A batch contains a fixed-length state/action/
    reward sequence for SPR and the world model, but only one n-step return,
    discount, terminal, and endpoint state for the root TD target. The TD
    horizon and gamma are selected per sample call, without changing any output
    shape.
    """

    def __init__(
            self,
            observation_shape,
            stack_size,
            replay_capacity,
            batch_size,
            update_horizon=10,
            subseq_len=1,
            n_envs=1,
            max_sample_attempts=1000,
            observation_dtype=np.uint8,
            terminal_dtype=np.uint8,
            action_shape=(),
            action_dtype=np.int32,
            reward_shape=(),
            reward_dtype=np.float32,
    ):
        if not isinstance(observation_shape, tuple):
            raise TypeError('observation_shape must be a tuple.')
        replay_capacity = int(replay_capacity)
        stack_size = int(stack_size)
        batch_size = int(batch_size)
        update_horizon = int(update_horizon)
        subseq_len = int(subseq_len)
        n_envs = int(n_envs)
        if n_envs < 1:
            raise ValueError('n_envs must be positive, got {}'.format(n_envs))
        if stack_size < 1:
            raise ValueError(
                'stack_size must be positive, got {}'.format(stack_size))
        if batch_size < 1:
            raise ValueError(
                'batch_size must be positive, got {}'.format(batch_size))
        if update_horizon < 1:
            raise ValueError('update_horizon must be positive, got {}'.format(
                update_horizon))
        if subseq_len < 1:
            raise ValueError(
                'subseq_len must be positive, got {}'.format(subseq_len))
        replay_length = replay_capacity // n_envs
        required_future = max(update_horizon, subseq_len - 1)
        minimum_replay_length = stack_size + required_future
        if replay_length < minimum_replay_length:
            raise ValueError(
                'Replay capacity must provide at least stack_size ({}) + '
                'max(update_horizon ({}), subseq_len - 1 ({})) rows per '
                'environment.'.format(stack_size, update_horizon,
                                      subseq_len - 1))

        self._action_shape = tuple(action_shape)
        self._action_dtype = action_dtype
        self._reward_shape = tuple(reward_shape)
        self._reward_dtype = reward_dtype
        self._observation_shape = observation_shape
        self._observation_dtype = observation_dtype
        self._terminal_dtype = terminal_dtype
        self._stack_size = stack_size
        self._state_shape = observation_shape + (stack_size,)
        self._batch_size = batch_size
        self._update_horizon = update_horizon
        self._subseq_len = subseq_len
        self._max_sample_attempts = int(max_sample_attempts)
        self._n_envs = n_envs
        self._replay_length = replay_length
        self._replay_capacity = replay_length * n_envs

        logging.info(
            'Creating compact prioritized replay: capacity=%s, n_envs=%s, '
            'batch=%s, subseq_len=%s, max_horizon=%s',
            self._replay_capacity, n_envs, batch_size, subseq_len,
            update_horizon)

        self._store = {}
        for element in self.get_storage_signature():
            shape = ((self._replay_length, self._n_envs) +
                     tuple(element.shape))
            self._store[element.name] = np.empty(shape, dtype=element.type)
        self.sum_tree = sum_tree.DeterministicSumTree(self._replay_capacity)
        self.add_count = np.array(0)
        self.total_steps = 0
        self._episode_end_indices = set()
        self._timeout_indices = set()
        self._rng = None

    def get_storage_signature(self):
        """Returns the four transition fields retained in replay storage."""
        return [
            ReplayElement('observation', self._observation_shape,
                          self._observation_dtype),
            ReplayElement('action', self._action_shape, self._action_dtype),
            ReplayElement('reward', self._reward_shape, self._reward_dtype),
            ReplayElement('terminal', (), self._terminal_dtype),
        ]

    def get_add_args_signature(self):
        """Returns the transition fields plus the non-stored PER priority."""
        return self.get_storage_signature() + [
            ReplayElement('priority', (), np.float32)
        ]

    def _check_add_types(self, *args):
        signature = self.get_add_args_signature()
        if len(args) != len(signature):
            raise ValueError('Add expects {} elements, received {}'.format(
                len(signature), len(args)))
        for index, (argument, element) in enumerate(zip(args, signature)):
            argument_shape = np.asarray(argument).shape
            if not argument_shape or argument_shape[0] != self._n_envs:
                raise ValueError(
                    'arg {} must have leading n_envs dimension {}, got {}.'
                    .format(index, self._n_envs, argument_shape))
            if argument_shape[1:] != tuple(element.shape):
                raise ValueError('arg {} has shape {}, expected ({}, {}).'
                                 .format(index, argument_shape, self._n_envs,
                                         tuple(element.shape)))

    def add(self,
            observation,
            action,
            reward,
            terminal,
            priority=None,
            episode_end=False):
        """Adds one row for every environment at the current cursor."""
        if priority is None:
            priority = np.full(
                (self._n_envs,),
                self.sum_tree.max_recorded_priority,
                dtype=np.float32,
            )
        self._check_add_types(observation, action, reward, terminal, priority)
        priority = np.asarray(priority, dtype=np.float32)
        if not np.all(np.isfinite(priority)) or np.any(priority < 0.0):
            raise ValueError('Priorities must be finite and nonnegative.')

        terminal = np.asarray(terminal)
        episode_end = np.broadcast_to(
            np.asarray(episode_end, dtype=bool), terminal.shape)
        resets = episode_end + terminal
        cursor = self.cursor()
        for env_index in range(self._n_envs):
            key = (cursor, env_index)
            if resets[env_index]:
                self._episode_end_indices.add(key)
            else:
                self._episode_end_indices.discard(key)
            if episode_end[env_index] and not terminal[env_index]:
                self._timeout_indices.add(key)
            else:
                self._timeout_indices.discard(key)

        flat_indices = self.ravel_indices(
            np.full((self._n_envs,), cursor, dtype=np.int64),
            np.arange(self._n_envs),
        )
        for flat_index, value in zip(flat_indices, priority):
            self.sum_tree.set(flat_index, value)

        self._store['observation'][cursor] = observation
        self._store['action'][cursor] = action
        self._store['reward'][cursor] = reward
        self._store['terminal'][cursor] = terminal
        self.add_count += 1
        self.total_steps += self._n_envs

    def is_empty(self):
        return self.add_count == 0

    def is_full(self):
        return self.add_count >= self._replay_length

    def num_elements(self):
        if self.is_full():
            return self._replay_capacity
        return self.cursor() * self._n_envs

    def cursor(self):
        """Returns the row where the next transition will be written."""
        return int(self.add_count % self._replay_length)

    def ravel_indices(self, indices_t, indices_b):
        return np.ravel_multi_index(
            (indices_t, indices_b),
            (self._replay_length, self._n_envs),
            mode='wrap',
        )

    def unravel_indices(self, indices):
        return np.unravel_index(
            indices, (self._replay_length, self._n_envs))

    def _stack_censor_before(self, index_t, index_b):
        """Returns the first valid frame row for one unwrapped state index."""
        index_t = int(index_t)
        index_b = int(index_b)
        first_valid = index_t - self._stack_size + 1
        for offset in range(-self._stack_size + 1, 0):
            unwrapped_index = index_t + offset
            stored_index = unwrapped_index % self._replay_length
            if (self._store['terminal'][stored_index, index_b] or
                    (stored_index, index_b) in self._episode_end_indices):
                first_valid = unwrapped_index + 1
        return first_valid

    def _get_stacked_observations(self, indices_t, indices_b):
        """Builds frame stacks for parallel unwrapped time coordinates."""
        indices_t = np.asarray(indices_t)
        indices_b = np.asarray(indices_b)
        first_valid = np.asarray([
            self._stack_censor_before(t, b)
            for t, b in zip(indices_t, indices_b)
        ])
        frame_indices = (
            indices_t[:, None] +
            np.arange(-self._stack_size + 1, 1, dtype=np.int64)[None, :])

        # Fill the final channel-last layout directly. The previous
        # frame-major gather allocated another full array for `result * mask`
        # and returned a non-contiguous moveaxis view, which then had to be
        # packed before transfer to JAX.
        result = np.empty(
            (indices_t.size,) + self._observation_shape +
            (self._stack_size,),
            dtype=self._observation_dtype,
        )
        for stack_index in range(self._stack_size):
            result[..., stack_index] = self._store['observation'][
                frame_indices[:, stack_index] % self._replay_length,
                indices_b,
            ]
            censored = frame_indices[:, stack_index] < first_valid
            if np.any(censored):
                result[censored, ..., stack_index] = 0
        return result

    def _required_future(self, update_horizon):
        return max(int(update_horizon), self._subseq_len - 1)

    def is_valid_transition(self, index_t, index_b, update_horizon=None):
        """Checks cursor and non-terminal episode-boundary collisions."""
        update_horizon = (self._update_horizon if update_horizon is None else
                          int(update_horizon))
        required_future = self._required_future(update_horizon)
        index_t = np.asarray(index_t).reshape(-1)
        index_b = np.asarray(index_b).reshape(-1)
        if index_t.size != 1 or index_b.size != 1:
            raise ValueError('is_valid_transition expects one replay index.')
        start_index = int(index_t[0])
        env_index = int(index_b[0])
        if start_index < 0 or start_index >= self._replay_length:
            return False, 0
        if env_index < 0 or env_index >= self._n_envs:
            return False, 0
        if not self.is_full():
            # Need rows through start + required_future, inclusive.
            if start_index > self.cursor() - required_future - 1:
                return False, 0
            if start_index < self._stack_size - 1:
                return False, 0

        cursor_invalid = invalid_range(
            self.cursor(), self._replay_length, self._stack_size - 1,
            required_future)
        if start_index in set(cursor_invalid):
            return False, 0

        first_valid = self._stack_censor_before(start_index, env_index)
        # A timeout boundary may not be crossed by either the model sequence or
        # the root TD window. True terminals are allowed; masks handle them.
        for row in modulo_range(start_index, required_future,
                                self._replay_length):
            if ((row, env_index) in self._episode_end_indices and
                    not self._store['terminal'][row, env_index]):
                return False, 0
        return True, first_valid

    def sample_index_batch(self, batch_size, update_horizon=None):
        """Samples valid anchors from the sole supported PER distribution."""
        batch_size = int(batch_size)
        update_horizon = (self._update_horizon if update_horizon is None else
                          int(update_horizon))
        if update_horizon < 1 or update_horizon > self._update_horizon:
            raise ValueError(
                'update_horizon must be in [1, {}], got {}.'.format(
                    self._update_horizon, update_horizon))
        if self._rng is None:
            raise RuntimeError('A PRNG key must be supplied before sampling.')

        flat_indices = np.asarray(
            self.sum_tree.stratified_sample(batch_size, self._rng))
        t_indices, b_indices = self.unravel_indices(flat_indices)
        t_indices = np.asarray(t_indices)
        b_indices = np.asarray(b_indices)
        attempts_left = self._max_sample_attempts

        for batch_index in range(batch_size):
            valid, _ = self.is_valid_transition(
                t_indices[batch_index:batch_index + 1],
                b_indices[batch_index:batch_index + 1],
                update_horizon=update_horizon,
            )
            while not valid and attempts_left > 0:
                self._rng, retry_rng = jax.random.split(self._rng)
                flat_index = int(
                    self.sum_tree.stratified_sample(1, retry_rng).item())
                t_index, b_index = self.unravel_indices(flat_index)
                t_indices[batch_index] = t_index
                b_indices[batch_index] = b_index
                attempts_left -= 1
                valid, _ = self.is_valid_transition(
                    t_indices[batch_index:batch_index + 1],
                    b_indices[batch_index:batch_index + 1],
                    update_horizon=update_horizon,
                )
            if not valid:
                raise RuntimeError(
                    'Unable to sample {} valid replay anchors after {} retry '
                    'attempts.'.format(batch_size, self._max_sample_attempts))
        return t_indices, b_indices

    def sample(self, *args, **kwargs):
        return self.sample_transition_batch(*args, **kwargs)

    def sample_transition_batch(self,
                                rng,
                                batch_size=None,
                                indices=None,
                                update_horizon=None,
                                gamma=None):
        """Returns compact model sequences and one root n-step TD transition."""
        self._rng = rng
        batch_size = self._batch_size if batch_size is None else int(batch_size)
        update_horizon = (self._update_horizon if update_horizon is None else
                          int(update_horizon))
        if update_horizon < 1 or update_horizon > self._update_horizon:
            raise ValueError(
                'update_horizon must be in [1, {}], got {}.'.format(
                    self._update_horizon, update_horizon))
        if gamma is None:
            raise ValueError('gamma must be supplied for every replay sample.')
        gamma = float(gamma)
        if not np.isfinite(gamma) or gamma < 0.0 or gamma >= 1.0:
            raise ValueError(
                'gamma must be finite and in [0, 1), got {}.'.format(gamma))

        if indices is None:
            root_t, root_b = self.sample_index_batch(
                batch_size, update_horizon=update_horizon)
        else:
            flat_indices = np.asarray(indices).reshape(-1)
            if flat_indices.size != batch_size:
                raise ValueError('Expected {} indices, got {}.'.format(
                    batch_size, flat_indices.size))
            if not np.issubdtype(flat_indices.dtype, np.integer):
                raise TypeError('Replay indices must be integers, got {}.'
                                .format(flat_indices.dtype))
            root_t, root_b = self.unravel_indices(flat_indices)
            root_t = np.asarray(root_t)
            root_b = np.asarray(root_b)
            for batch_index in range(batch_size):
                valid, _ = self.is_valid_transition(
                    root_t[batch_index:batch_index + 1],
                    root_b[batch_index:batch_index + 1],
                    update_horizon=update_horizon,
                )
                if not valid:
                    raise ValueError('Invalid replay anchor: {}.'.format(
                        flat_indices[batch_index]))

        # Fixed-length real sequence used by SPR and the world-model losses.
        sequence_t = (root_t[:, None] +
                      np.arange(self._subseq_len)[None, :])
        sequence_b = np.broadcast_to(root_b[:, None], sequence_t.shape)
        flat_sequence_t = sequence_t.reshape(-1)
        flat_sequence_b = sequence_b.reshape(-1)
        states = self._get_stacked_observations(
            flat_sequence_t, flat_sequence_b).reshape(
                (batch_size, self._subseq_len) + self._state_shape)
        stored_sequence_t = sequence_t % self._replay_length
        actions = self._store['action'][stored_sequence_t, sequence_b]
        rewards = self._store['reward'][stored_sequence_t, sequence_b]
        dones = self._store['terminal'][stored_sequence_t, sequence_b]
        previous_dones = np.zeros_like(dones)
        previous_dones[:, 1:] = dones[:, :-1]
        same_trajectory = (1 - previous_dones).cumprod(axis=1).astype(
            self._terminal_dtype, copy=False)

        # Root-only H-step return and terminal. A terminal reward is included;
        # rewards and the bootstrap after that terminal are masked.
        trajectory_t = (np.arange(update_horizon)[:, None] + root_t[None, :])
        trajectory_b = np.broadcast_to(root_b[None, :], trajectory_t.shape)
        stored_trajectory_t = trajectory_t % self._replay_length
        trajectory_terminals = self._store['terminal'][
            stored_trajectory_t, trajectory_b].astype(bool, copy=False)
        terminal = trajectory_terminals.any(axis=0).astype(
            self._terminal_dtype, copy=False)
        alive = np.ones(trajectory_terminals.shape, dtype=np.float32)
        if update_horizon > 1:
            alive[1:] = np.cumprod(
                1.0 - trajectory_terminals[:-1].astype(np.float32), axis=0)
        powers = np.power(
            np.float32(gamma), np.arange(update_horizon + 1),
        ).astype(np.float32)
        reward_weights = alive * powers[:update_horizon, None]
        trajectory_rewards = self._store['reward'][
            stored_trajectory_t, trajectory_b]
        weight_shape = reward_weights.shape + (1,) * len(self._reward_shape)
        returns = np.sum(
            reward_weights.reshape(weight_shape) * trajectory_rewards,
            axis=0,
            dtype=self._reward_dtype,
        )
        discounts = np.full(
            (batch_size,), powers[update_horizon], dtype=self._reward_dtype)

        if update_horizon < self._subseq_len:
            # S_H is already part of the fixed model sequence. Copy its view
            # into a compact transfer buffer instead of gathering four frames
            # from replay a second time.
            next_states = np.ascontiguousarray(states[:, update_horizon])
        else:
            endpoint_t = root_t + update_horizon
            next_states = self._get_stacked_observations(endpoint_t, root_b)
        root_indices = self.ravel_indices(
            root_t % self._replay_length, root_b).astype(np.int32)
        priorities = self.get_priority(root_indices)

        return [
            states,
            actions,
            rewards,
            returns,
            discounts,
            next_states,
            terminal,
            same_trajectory,
            root_indices,
            priorities,
        ]

    def get_transition_elements(self, batch_size=None):
        """Returns the exact compact batch signature."""
        batch_size = self._batch_size if batch_size is None else int(batch_size)
        sequence_prefix = (batch_size, self._subseq_len)
        return [
            ReplayElement('state', sequence_prefix + self._state_shape,
                          self._observation_dtype),
            ReplayElement('action', sequence_prefix + self._action_shape,
                          self._action_dtype),
            ReplayElement('reward', sequence_prefix + self._reward_shape,
                          self._reward_dtype),
            ReplayElement('return', (batch_size,) + self._reward_shape,
                          self._reward_dtype),
            ReplayElement('discount', (batch_size,), self._reward_dtype),
            ReplayElement('next_state', (batch_size,) + self._state_shape,
                          self._observation_dtype),
            ReplayElement('terminal', (batch_size,), self._terminal_dtype),
            ReplayElement('same_trajectory', sequence_prefix,
                          self._terminal_dtype),
            ReplayElement('indices', (batch_size,), np.int32),
            # These are raw sum-tree leaves, not normalized sampling
            # probabilities. BBF's batch-normalized beta=.5 correction consumes
            # these leaves directly; the common tree normalizer cancels.
            ReplayElement('priorities', (batch_size,), np.float32),
        ]

    def set_priority(self, indices, priorities):
        """Updates one raw sum-tree leaf priority per sampled root."""
        indices = np.asarray(indices)
        if not np.issubdtype(indices.dtype, np.integer):
            raise TypeError('Indices must be integers, got {}.'.format(
                indices.dtype))
        indices = indices.astype(np.int32, copy=False).reshape(-1)
        priorities = np.asarray(priorities, dtype=np.float32).reshape(-1)
        if priorities.size != indices.size:
            raise ValueError(
                'Priority count must match index count: got {} for {}.'
                .format(priorities.size, indices.size))
        if not np.all(np.isfinite(priorities)) or np.any(priorities < 0.0):
            raise ValueError('Priorities must be finite and nonnegative.')
        for index, priority in zip(indices, priorities):
            self.sum_tree.set(index, priority)

    def get_priority(self, indices):
        indices = np.asarray(indices)
        if not indices.shape:
            raise ValueError('Indices must be an array.')
        if not np.issubdtype(indices.dtype, np.integer):
            raise TypeError('Indices must be integers, got {}.'.format(
                indices.dtype))
        return np.asarray(
            self.sum_tree.get(indices.astype(np.int32, copy=False)),
            dtype=np.float32,
        )

    def mean_priority(self, update_horizon=None):
        """Returns the mean priority over anchors sampling can actually accept.

        Sampling rejects roots whose frame history/future crosses the circular
        cursor, as well as roots whose future crosses a nonterminal episode end.
        This diagnostic uses that same conditional support rather than including
        leaves that the sampler will reject.
        """
        populated = self.sum_tree.highest_set + 1
        if populated <= 0:
            return np.float32(1.0)

        update_horizon = (self._update_horizon if update_horizon is None else
                          int(update_horizon))
        if update_horizon < 1 or update_horizon > self._update_horizon:
            raise ValueError(
                'update_horizon must be in [1, {}], got {}.'.format(
                    self._update_horizon, update_horizon))
        required_future = self._required_future(update_horizon)

        invalid_indices = set()

        def mark_invalid(row, env_index):
            flat_index = ((int(row) % self._replay_length) * self._n_envs +
                          int(env_index))
            if flat_index < populated:
                invalid_indices.add(flat_index)

        # These roots have a frame stack or required future row on the other
        # side of the circular write cursor.
        cursor_rows = invalid_range(
            self.cursor(), self._replay_length, self._stack_size - 1,
            required_future)
        for row in cursor_rows:
            for env_index in range(self._n_envs):
                mark_invalid(row, env_index)

        if not self.is_full():
            # Before the first wrap there is no frame history preceding row 0.
            for row in range(self._stack_size - 1):
                for env_index in range(self._n_envs):
                    mark_invalid(row, env_index)

        # A timeout at transition e invalidates roots e-F+1 through e. True
        # terminals remain sampleable and are handled by return/model masks.
        for episode_row, env_index in self._timeout_indices:
            for offset in range(required_future):
                mark_invalid(episode_row - offset, env_index)

        valid_count = populated - len(invalid_indices)
        if valid_count <= 0:
            raise RuntimeError('Replay contains no valid anchors.')
        invalid_priority = 0.0
        if invalid_indices:
            invalid_priority = np.sum(
                self.sum_tree.get(
                    np.fromiter(invalid_indices, dtype=np.int64)),
                dtype=np.float64,
            )
        valid_priority = self.sum_tree._total_priority() - invalid_priority
        return np.float32(valid_priority / valid_count)

    def reset_priorities(self):
        self.sum_tree.reset_priorities()
