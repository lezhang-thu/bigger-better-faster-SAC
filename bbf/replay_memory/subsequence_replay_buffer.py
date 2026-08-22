# coding=utf-8

import collections
import gzip
import math
import os
import pickle

from absl import logging
import gin
import jax
import numpy as np

from bbf.replay_memory import deterministic_sum_tree as sum_tree
from bbf.replay_memory.circular_replay_buffer import modulo_range, invalid_range, ReplayElement


@gin.configurable
class JaxSubsequenceParallelEnvReplayBuffer(object):
    """A simple out-of-graph Replay Buffer.

  Stores transitions, state, action, reward, next_state, terminal (and any
  extra contents specified) in a circular buffer and provides a uniform
  transition sampling function.
  When the states consist of stacks of observations storing the states is
  inefficient. This class writes observations and constructs the stacked
  states at sample time.
  This class supports multiple parallel environments and returns
  subsequences by default.
  Attributes:
    add_count: int, counter of how many transitions have been added (including
      the blank ones at the beginning of an episode).
    invalid_range: np.array, an array with the indices of cursor-related invalid
      transitions
    total_steps: int, total number of transitions added across all environments.
  """

    def __init__(
            self,
            observation_shape,
            stack_size,
            replay_capacity,
            batch_size,
            subseq_len,
            n_envs=1,
            update_horizon=1,
            gamma=0.99,
            max_sample_attempts=1000,
            use_next_state=True,
            extra_storage_types=None,
            observation_dtype=np.uint8,
            terminal_dtype=np.uint8,
            action_shape=(),
            action_dtype=np.int32,
            reward_shape=(),
            reward_dtype=np.float32,
    ):
        """Initializes OutOfGraphReplayBuffer.

    Args:
      observation_shape: tuple of ints.
      stack_size: int, number of frames to use in state stack.
      replay_capacity: int, number of transitions to keep in memory.
      batch_size: int.
      subseq_len: int, length of subsequences to return.
      n_envs: int, how many parallel environments will be writing data.
      update_horizon: int, length of update ('n' in n-step update).
      gamma: int, the discount factor.
      max_sample_attempts: int, the maximum number of attempts allowed to get a
        sample.
      use_next_state: bool, whether to return separate "next_observation",
        "next_reward" and "next_action" entries. Disable to reduce sampling time
        in pure sequence modeling tasks.
      extra_storage_types: list of ReplayElements defining the type of the extra
        contents that will be stored and returned by sample_transition_batch.
      observation_dtype: np.dtype, type of the observations. Defaults to
        np.uint8 for Atari 2600.
      terminal_dtype: np.dtype, type of the terminals. Defaults to np.uint8 for
        Atari 2600.
      action_shape: tuple of ints, the shape for the action vector. Empty tuple
        means the action is a scalar.
      action_dtype: np.dtype, type of elements in the action.
      reward_shape: tuple of ints, the shape of the reward vector. Empty tuple
        means the reward is a scalar.
      reward_dtype: np.dtype, type of elements in the reward.

    Raises:
      ValueError: If replay_capacity is too small to hold at least one
          transition.
    """
        assert isinstance(observation_shape, tuple)
        if n_envs < 1:
            raise ValueError('n_envs must be positive, got {}'.format(n_envs))
        replay_length = int(replay_capacity // n_envs)
        minimum_replay_length = stack_size + subseq_len + update_horizon - 1
        if replay_length < minimum_replay_length:
            raise ValueError('There is not enough capacity to cover '
                             'stack_size, subseq_len, and update_horizon per '
                             'environment.')

        logging.info(
            'Creating a %s replay memory with the following parameters:',
            self.__class__.__name__)
        logging.info('\t observation_shape: %s', str(observation_shape))
        logging.info('\t observation_dtype: %s', str(observation_dtype))
        logging.info('\t terminal_dtype: %s', str(terminal_dtype))
        logging.info('\t stack_size: %d', stack_size)
        logging.info('\t use_next_state: %d', use_next_state)
        logging.info('\t replay_capacity: %d', replay_capacity)
        logging.info('\t batch_size: %d', batch_size)
        logging.info('\t update_horizon: %d', update_horizon)
        logging.info('\t gamma: %f', gamma)

        self._action_shape = action_shape
        self._action_dtype = action_dtype
        self._reward_shape = reward_shape
        self._reward_dtype = reward_dtype
        self._observation_shape = observation_shape
        self._stack_size = stack_size
        self._state_shape = self._observation_shape + (self._stack_size,)
        self._batch_size = batch_size
        self._update_horizon = update_horizon
        self._gamma = gamma
        self._observation_dtype = observation_dtype
        self._terminal_dtype = terminal_dtype
        self._max_sample_attempts = max_sample_attempts
        self._subseq_len = subseq_len
        self._use_next_state = use_next_state

        self._n_envs = n_envs
        self._replay_length = replay_length

        # Gotta round this down, since the matrix is rectangular.
        self._replay_capacity = self._replay_length * self._n_envs

        self.total_steps = 0

        if extra_storage_types:
            self._extra_storage_types = extra_storage_types
        else:
            self._extra_storage_types = []
        self._create_storage()
        self.add_count = np.array(0)
        self.invalid_range = np.array([], dtype=np.int64)
        # When the horizon is > 1, we compute the sum of discounted rewards as a dot
        # product using the precomputed vector <gamma^0, gamma^1, ..., gamma^{n-1}>.
        self._cumulative_discount_vector = np.array(
            [math.pow(self._gamma, n) for n in range(update_horizon + 1)],
            dtype=np.float32)
        self._next_experience_is_episode_start = True
        self._episode_end_indices = set()

    def _create_storage(self):
        """Creates the numpy arrays used to store transitions."""
        self._store = {}
        for storage_element in self.get_storage_signature():
            array_shape = [self._replay_length, self._n_envs] + list(
                storage_element.shape)
            self._store[storage_element.name] = np.empty(
                array_shape, dtype=storage_element.type)

    def get_add_args_signature(self):
        """The signature of the add function.

    Note - Derived classes may return a different signature.
    Returns:
      list of ReplayElements defining the type of the argument signature
      needed by the add function.
    """
        return self.get_storage_signature()

    def get_storage_signature(self):
        """Returns a default list of elements to be stored in this replay memory.

        Note - Derived classes may return a different signature.
        Returns:
            list of ReplayElements defining the type of the contents stored.
    """
        storage_elements = [
            ReplayElement('observation', self._observation_shape,
                          self._observation_dtype),
            ReplayElement('action', self._action_shape, self._action_dtype),
            ReplayElement('reward', self._reward_shape, self._reward_dtype),
            ReplayElement('terminal', (), self._terminal_dtype)
        ]

        for extra_replay_element in self._extra_storage_types:
            storage_elements.append(extra_replay_element)
        return storage_elements

    def _add_zero_transition(self):
        """Adds a padding transition filled with zeros (Used in episode beginnings).
    """
        zero_transition = []
        for element_type in self.get_add_args_signature():
            zero_transition.append(
                np.zeros(element_type.shape, dtype=element_type.type))
        self._episode_end_indices.discard(self.cursor())  # If present
        self._add(*zero_transition)

    def add(self,
            observation,
            action,
            reward,
            terminal,
            *args,
            priority=None,
            episode_end=False):
        """Adds a transition to the replay memory.

    This function checks the types and handles the padding at the beginning
    of an episode. Then it calls the _add function.
    Since the next_observation in the transition will be the observation
    added next there is no need to pass it.
    If the replay memory is at capacity the oldest transition will be
    discarded.

    Args:
      observation: np.array with shape observation_shape.
      action: int, the action in the transition.
      reward: float, the reward received in the transition.
      terminal: np.dtype, acts as a boolean indicating whether the transition
        was terminal (1) or not (0).
      *args: extra contents with shapes and dtypes according to
        extra_storage_types.
      priority: float, unused in the circular replay buffer, but may be used in
        child classes like PrioritizedReplayBuffer.
      episode_end: bool, whether this experience is the last experience in the
        episode. This is useful for tasks that terminate due to time-out, but do
        not end on a terminal state. Overloading 'terminal' may not be
        sufficient in this case, since 'terminal' is passed to the agent for
        training. 'episode_end' allows the replay buffer to determine episode
        boundaries without passing that information to the agent.
    """
        if priority is not None:
            args = args + (priority,)

        self.total_steps += self._n_envs

        self._check_add_types(observation, action, reward, terminal, *args)

        resets = episode_end + terminal
        for i in range(resets.shape[0]):
            if resets[i]:
                self._episode_end_indices.add((self.cursor(), i))
            else:
                self._episode_end_indices.discard(
                    (self.cursor(), i))  # If present

        self._add(observation, action, reward, terminal, *args)

    def _add(self, *args):
        """Internal add method to add to the storage arrays.

    Args:
        *args: All the elements in a transition.
    """
        self._check_args_length(*args)
        transition = {
            e.name: args[idx]
            for idx, e in enumerate(self.get_add_args_signature())
        }
        self._add_transition(transition)

    def _add_transition(self, transition):
        """Internal add method to add transition dictionary to storage arrays.

    Args:
        transition: The dictionary of names and values of the transition to add
          to the storage. Each tensor should have leading dim equal to the
          number of environments used by the buffer.
    """
        cursor = self.cursor()
        for arg_name in transition:
            self._store[arg_name][cursor] = transition[arg_name]

        self.add_count += 1
        self.invalid_range = invalid_range(
            self.cursor(), self._replay_length, self._stack_size - 1,
            self._update_horizon + self._subseq_len - 1)

    def _check_args_length(self, *args):
        """Check if args passed to the add method have the same length as storage.

    Args:
        *args: Args for elements used in storage.

    Raises:
        ValueError: If args have wrong length.
    """
        if len(args) != len(self.get_add_args_signature()):
            raise ValueError('Add expects {} elements, received {}'.format(
                len(self.get_add_args_signature()), len(args)))

    def _check_add_types(self, *args):
        """Checks if args passed to the add method match those of the storage.

    Args:
        *args: Args whose types need to be validated.

    Raises:
        ValueError: If args have wrong shape or dtype.
    """
        self._check_args_length(*args)
        for i, (arg_element, store_element) in enumerate(
                zip(args, self.get_add_args_signature())):
            if isinstance(arg_element, np.ndarray):
                arg_shape = arg_element.shape
            elif isinstance(arg_element, tuple) or isinstance(
                    arg_element, list):
                # TODO(b/80536437). This is not efficient when arg_element is a list.
                arg_shape = np.array(arg_element).shape
            else:
                # Assume it is scalar.
                arg_shape = tuple()
            store_element_shape = tuple(store_element.shape)
            assert arg_shape[0] == self._n_envs
            arg_shape = arg_shape[1:]
            if arg_shape != store_element_shape:
                raise ValueError('arg {} has shape {}, expected {}'.format(
                    i, arg_shape, store_element_shape))

    def is_empty(self):
        """Is the Replay Buffer empty?"""
        return self.add_count == 0

    def is_full(self):
        """Is the Replay Buffer full?"""
        return self.add_count >= self._replay_length

    def ravel_indices(self, indices_t, indices_b):
        return np.ravel_multi_index((indices_t, indices_b),
                                    (self._replay_length, self._n_envs),
                                    mode='wrap')

    def unravel_indices(self, indices):
        return np.unravel_index(indices, (self._replay_length, self._n_envs))

    def get_from_store(self, element_name, indices_t, indices_b):
        array = self._store[element_name]
        return array[indices_t, indices_b]

    def cursor(self):
        """Index to the location where the next transition will be written."""
        return self.add_count % self._replay_length

    def parallel_get_stack(self, element_name, indices_t, indices_b,
                           first_valid):
        """Builds stacks in unwrapped time while wrapping storage indices."""
        indices_t = np.arange(-self._stack_size + 1,
                              1)[:, None] + indices_t[None, :]
        indices_b = indices_b[None, :].repeat(self._stack_size, axis=0)
        mask = indices_t >= first_valid
        result = self.get_from_store(element_name,
                                     indices_t % self._replay_length, indices_b)
        mask = mask.reshape(*mask.shape, *([1] * (len(result.shape) - 2)))
        result = result * mask
        result = np.moveaxis(result, 0, -1)
        return result

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

    def get_terminal_stack(self, index_t, index_b):
        index_t = np.asarray(index_t).reshape(-1)
        index_b = np.asarray(index_b).reshape(-1)
        first_valid = np.asarray([
            self._stack_censor_before(t, b)
            for t, b in zip(index_t, index_b)
        ])
        return self.parallel_get_stack('terminal', index_t, index_b,
                                       first_valid)

    def is_valid_transition(self,
                            index_t,
                            index_b,
                            subseq_len=None,
                            update_horizon=None):
        """Checks if the index contains a valid transition.

    Checks for collisions with the end of episodes and the current position
    of the cursor.
    Args:
      index_t: int, index in the time dimension of the state.
      index_b: int, index in the environment dimension of the state.

    Returns:
      Is the index valid: Boolean.
      Start of the current episode (if within our stack size): Integer.
    """
        subseq_len = self._subseq_len if subseq_len is None else subseq_len
        update_horizon = (self._update_horizon if update_horizon is None else
                          update_horizon)
        required_future = update_horizon + subseq_len - 1
        index_t = np.asarray(index_t).reshape(-1)
        index_b = np.asarray(index_b).reshape(-1)
        if index_t.size != 1 or index_b.size != 1:
            raise ValueError('is_valid_transition expects one replay index.')
        start_index = int(index_t[0])
        env_index = int(index_b[0])

        # Check the index is in the valid range.
        if start_index < 0 or start_index >= self._replay_length:
            return False, 0
        if not self.is_full():
            # The final state needed is t + subseq_len - 1 + horizon.
            if start_index > self.cursor() - subseq_len - update_horizon:
                return False, 0
            # The first few indices contain the padding states of the first episode.
            if start_index < self._stack_size - 1:
                return False, 0

        # Skip transitions that straddle the cursor.
        runtime_invalid_range = invalid_range(
            self.cursor(), self._replay_length, self._stack_size - 1,
            required_future)
        if start_index in set(runtime_invalid_range):
            return False, 0

        ep_start = self._stack_censor_before(start_index, env_index)

        # Reject non-terminal episode boundaries anywhere in the complete
        # subsequence plus bootstrap window.
        for i in modulo_range(start_index, required_future,
                              self._replay_length):
            if ((i, env_index) in self._episode_end_indices and
                    not self._store['terminal'][i, env_index]):
                return False, 0

        return True, ep_start

    def _create_batch_arrays(self, batch_size):
        """Create a tuple of arrays with the type of get_transition_elements.

    When using the WrappedReplayBuffer with staging enabled it is important to
    create new arrays every sample because StaginArea keeps a pointer to the
    returned arrays.
    Args:
      batch_size: (int) number of transitions returned. If None the default
        batch_size will be used.

    Returns:
      Tuple of np.arrays with the shape and type of
      get_transition_elements.
    """
        transition_elements = self.get_transition_elements(batch_size)
        batch_arrays = []
        for element in transition_elements:
            batch_arrays.append(np.empty(element.shape, dtype=element.type))
        return tuple(batch_arrays)

    def num_elements(self):
        if self.is_full():
            return self._replay_capacity
        else:
            return self.cursor() * self._n_envs

    def sample_index_batch(self,
                           batch_size,
                           subseq_len=None,
                           update_horizon=None):
        """Returns a batch of valid indices sampled uniformly.

    Args:
      batch_size: int, number of indices returned.

    Returns:
      list of ints, a batch of valid indices sampled uniformly.

    Raises:
      RuntimeError: If the batch was not constructed after maximum number
      of tries.
    """
        subseq_len = self._subseq_len if subseq_len is None else subseq_len
        update_horizon = (self._update_horizon if update_horizon is None else
                          update_horizon)
        self._rng, rng = jax.random.split(self._rng)
        if self.is_full():
            # add_count >= self._replay_capacity > self._stack_size
            min_id = self.cursor() - self._replay_length + self._stack_size - 1
            max_id = self.cursor() - update_horizon - subseq_len + 1
        else:
            # add_count < self._replay_capacity
            min_id = self._stack_size - 1
            max_id = self.cursor() - update_horizon - subseq_len + 1
        if max_id <= min_id:
            raise RuntimeError(
                'Cannot sample a batch with fewer than stack size ({}) + '
                'update_horizon ({}) + subseq_len ({}) transitions.'.format(
                    self._stack_size, update_horizon, subseq_len))
        time_rng, env_rng = jax.random.split(rng)
        t_indices = jax.random.randint(time_rng, (batch_size,), min_id,
                                       max_id) % self._replay_length
        b_indices = jax.random.randint(env_rng, (batch_size,), 0, self._n_envs)
        allowed_attempts = self._max_sample_attempts
        t_indices = np.array(t_indices)
        b_indices = np.array(b_indices)
        censor_before = np.zeros_like(t_indices)
        for i in range(len(t_indices)):
            is_valid, ep_start = self.is_valid_transition(
                t_indices[i:i + 1],
                b_indices[i:i + 1],
                subseq_len=subseq_len,
                update_horizon=update_horizon)
            censor_before[i] = ep_start
            if not is_valid:
                if allowed_attempts == 0:
                    raise RuntimeError(
                        'Max sample attempts: Tried {} times but only sampled {}'
                        ' valid indices. Batch size is {}'.format(
                            self._max_sample_attempts, i, batch_size))
                while not is_valid and allowed_attempts > 0:
                    # If index i is not valid keep sampling others. Note that this
                    # is not stratified.
                    self._rng, rng = jax.random.split(self._rng)
                    time_rng, env_rng = jax.random.split(rng)
                    t_index = jax.random.randint(time_rng, (), min_id,
                                                 max_id).item()
                    t_index %= self._replay_length
                    b_index = jax.random.randint(env_rng, (), 0,
                                                 self._n_envs).item()
                    allowed_attempts -= 1
                    t_indices[i] = t_index
                    b_indices[i] = b_index
                    is_valid, first_valid = self.is_valid_transition(
                        t_indices[i:i + 1],
                        b_indices[i:i + 1],
                        subseq_len=subseq_len,
                        update_horizon=update_horizon)
                    censor_before[i] = first_valid
                if not is_valid:
                    raise RuntimeError(
                        'Max sample attempts: Tried {} times but only sampled '
                        '{} valid indices. Batch size is {}'.format(
                            self._max_sample_attempts, i, batch_size))
        return t_indices, b_indices, censor_before

    def restore_leading_dims(self, batch_size, subseq_len, tensor):
        return tensor.reshape(batch_size, subseq_len, *tensor.shape[1:])

    def sample(self, *args, **kwargs):
        return self.sample_transition_batch(*args, **kwargs)

    def sample_transition_batch(
        self,
        rng=None,
        batch_size=None,
        indices=None,
        subseq_len=None,
        update_horizon=None,
        gamma=None,
    ):
        """Returns a batch of transitions (including any extra contents).

    If get_transition_elements has been overridden and defines elements not
    stored in self._store, an empty array will be returned and it will be
    left to the child class to fill it. For example, for the child class
    OutOfGraphPrioritizedReplayBuffer, the contents of the
    sampling_probabilities are stored separately in a sum tree.
    When the transition is terminal next_state_batch has undefined contents.
    NOTE: This transition contains the indices of the sampled elements.
    These
    are only valid during the call to sample_transition_batch, i.e. they may
    be used by subclasses of this replay buffer but may point to different
    data
    as soon as sampling is done.
    Args:
      rng: Jax PRNG key, if overriding the default buffer state.
      batch_size: int, number of transitions returned. If None, the default
        batch_size will be used.
      indices: None or list of ints, the indices of every transition in the
        batch. If None, sample the indices uniformly.
      subseq_len: The length of subsequence to sample. Can override the replay
        buffer default.
      update_horizon: Update horizon to use, if overriding the original setting.
      gamma: Discount factor to use, if overriding the original setting.

    Returns:
      transition_batch: tuple of np.arrays with the shape and type as in
          get_transition_elements().
    Raises:
      ValueError: If an element to be sampled is missing from the replay
      buffer.
    """
        self._rng = rng if rng is not None else self._rng
        if batch_size is None:
            batch_size = self._batch_size
        if subseq_len is None:
            subseq_len = self._subseq_len
        if update_horizon is None:
            update_horizon = self._update_horizon
        if subseq_len < 1:
            raise ValueError('subseq_len must be positive, got {}'.format(
                subseq_len))
        if update_horizon < 1:
            raise ValueError('update_horizon must be positive, got {}'.format(
                update_horizon))
        if indices is None:
            t_indices, b_indices, censor_before = self.sample_index_batch(
                batch_size,
                subseq_len=subseq_len,
                update_horizon=update_horizon)
        else:
            flat_indices = np.asarray(indices).reshape(-1)
            if flat_indices.size != batch_size:
                raise ValueError('Expected {} indices, got {}'.format(
                    batch_size, flat_indices.size))
            if not np.issubdtype(flat_indices.dtype, np.integer):
                raise TypeError('Replay indices must be integers, got {}'.format(
                    flat_indices.dtype))
            try:
                t_indices, b_indices = self.unravel_indices(flat_indices)
            except ValueError as error:
                raise ValueError('Invalid replay anchor in {}'.format(
                    flat_indices.tolist())) from error
            t_indices = np.asarray(t_indices)
            b_indices = np.asarray(b_indices)
            censor_before = np.zeros_like(t_indices)
            for i in range(batch_size):
                is_valid, ep_start = self.is_valid_transition(
                    t_indices[i:i + 1],
                    b_indices[i:i + 1],
                    subseq_len=subseq_len,
                    update_horizon=update_horizon)
                if not is_valid:
                    raise ValueError('Invalid replay anchor: {}'.format(
                        flat_indices[i]))
                censor_before[i] = ep_start

        effective_gamma = self._gamma if gamma is None else gamma
        cumulative_discount_vector = np.array(
            [math.pow(effective_gamma, n)
             for n in range(update_horizon + 1)],
            dtype=np.float32,
        )
        assert len(t_indices) == batch_size
        assert len(b_indices) == batch_size
        transition_elements = self.get_transition_elements(
            batch_size, subseq_len=subseq_len)
        state_indices = t_indices[:, None] + np.arange(subseq_len)[None, :]
        state_indices = state_indices.reshape(batch_size * subseq_len)
        b_indices = b_indices[:, None].repeat(subseq_len, axis=1).reshape(
            batch_size * subseq_len)
        # A terminal inside the subsequence starts a new frame stack for later
        # states, so each state needs its own censor point.
        censor_before = np.asarray([
            self._stack_censor_before(t, b)
            for t, b in zip(state_indices, b_indices)
        ])

        # Rows store (s_t, a_t, r_{t+1}, done_{t+1}). An N-step target rooted
        # at t consumes rows t..t+N-1 and bootstraps from s_{t+N}.
        trajectory_indices = (np.arange(update_horizon)[:, None] +
                              state_indices[None, :]) % self._replay_length
        trajectory_b_indices = b_indices[None,].repeat(update_horizon, axis=0)
        trajectory_terminals = self._store['terminal'][trajectory_indices,
                                                       trajectory_b_indices]
        is_terminal_transition = trajectory_terminals.any(0).astype(
            self._terminal_dtype)
        # Include the terminal transition's reward and mask only later rewards.
        valid_reward_mask = np.concatenate(
            [
                np.ones_like(trajectory_terminals[:1], dtype=bool),
                np.logical_and.accumulate(
                    np.logical_not(trajectory_terminals[:-1].astype(bool)),
                    axis=0),
            ],
            axis=0,
        ).astype(np.float32)
        trajectory_discount_vector = valid_reward_mask * (
            cumulative_discount_vector[:update_horizon, None])
        trajectory_rewards = self._store['reward'][trajectory_indices,
                                                   trajectory_b_indices]
        discount_shape = trajectory_discount_vector.shape + (
            (1,) * len(self._reward_shape))
        returns = np.sum(
            trajectory_discount_vector.reshape(discount_shape) *
            trajectory_rewards,
            axis=0).astype(self._reward_dtype, copy=False)

        next_indices = state_indices + update_horizon
        next_censor_before = np.asarray([
            self._stack_censor_before(t, b)
            for t, b in zip(next_indices, b_indices)
        ])
        discounts = np.full(batch_size * subseq_len,
                            cumulative_discount_vector[update_horizon],
                            dtype=self._reward_dtype)
        stored_state_indices = state_indices % self._replay_length
        stored_next_indices = next_indices % self._replay_length
        outputs = []
        for element in transition_elements:
            name = element.name
            if name == 'state':
                output = self.parallel_get_stack(
                    'observation',
                    state_indices,
                    b_indices,
                    censor_before,
                )
                output = self.restore_leading_dims(batch_size, subseq_len,
                                                   output)
            elif name == 'return':
                # compute the discounted sum of rewards in the trajectory.
                output = returns
                output = self.restore_leading_dims(batch_size, subseq_len,
                                                   output)
            elif name == 'discount':
                output = discounts
                output = self.restore_leading_dims(batch_size, subseq_len,
                                                   output)
            elif name == 'next_state':
                output = self.parallel_get_stack(
                    'observation',
                    next_indices,
                    b_indices,
                    next_censor_before,
                )
                output = self.restore_leading_dims(batch_size, subseq_len,
                                                   output)
            elif name == 'same_trajectory':
                dones = self._store['terminal'][stored_state_indices, b_indices]
                dones = self.restore_leading_dims(batch_size, subseq_len, dones)
                previous_dones = np.zeros_like(dones)
                previous_dones[:, 1:] = dones[:, :-1]
                output = np.logical_and.accumulate(
                    np.logical_not(previous_dones.astype(bool)),
                    axis=1).astype(self._terminal_dtype)
            elif name in ('next_action', 'next_reward'):
                output = self._store[name.lstrip('next_')][stored_next_indices,
                                                           b_indices]
                output = self.restore_leading_dims(batch_size, subseq_len,
                                                   output)
            elif element.name == 'terminal':
                output = is_terminal_transition
                output = self.restore_leading_dims(batch_size, subseq_len,
                                                   output)
            elif name == 'indices':
                output = self.ravel_indices(stored_state_indices,
                                            b_indices).astype('int32')
                output = self.restore_leading_dims(batch_size, subseq_len,
                                                   output)[:, 0]
            elif name in self._store.keys():
                output = self._store[name][stored_state_indices, b_indices]
                output = self.restore_leading_dims(batch_size, subseq_len,
                                                   output)
            else:
                continue
            outputs.append(output)
        return outputs

    def get_transition_elements(self, batch_size=None, subseq_len=None):
        """Returns a 'type signature' for sample_transition_batch.

    Args:
      batch_size: int, number of transitions returned. If None, the default
        batch_size will be used.
      subseq_len: int, length of subsequences to return.

    Returns:
      signature: A namedtuple describing the method's return type signature.
    """
        subseq_len = self._subseq_len if subseq_len is None else subseq_len
        batch_size = self._batch_size if batch_size is None else batch_size

        transition_elements = [
            ReplayElement('state', (batch_size, subseq_len) + self._state_shape,
                          self._observation_dtype),
            ReplayElement('action',
                          (batch_size, subseq_len) + self._action_shape,
                          self._action_dtype),
            ReplayElement('reward',
                          (batch_size, subseq_len) + self._reward_shape,
                          self._reward_dtype),
            ReplayElement('return',
                          (batch_size, subseq_len) + self._reward_shape,
                          self._reward_dtype),
            ReplayElement('discount', (batch_size, subseq_len),
                          self._reward_dtype),
        ]
        if self._use_next_state:
            transition_elements += [
                ReplayElement('next_state',
                              (batch_size, subseq_len) + self._state_shape,
                              self._observation_dtype),
                ReplayElement('next_action',
                              (batch_size, subseq_len) + self._action_shape,
                              self._action_dtype),
                ReplayElement('next_reward',
                              (batch_size, subseq_len) + self._reward_shape,
                              self._reward_dtype),
            ]
        transition_elements += [
            ReplayElement('terminal', (batch_size, subseq_len),
                          self._terminal_dtype),
            ReplayElement('same_trajectory', (batch_size, subseq_len),
                          self._terminal_dtype),
            ReplayElement('indices', (batch_size,), np.int32)
        ]
        for element in self._extra_storage_types:
            transition_elements.append(
                ReplayElement(element.name,
                              (batch_size, subseq_len) + tuple(element.shape),
                              element.type))
        return transition_elements

    def reset_priorities(self):
        pass


@gin.configurable
class PrioritizedJaxSubsequenceParallelEnvReplayBuffer(
        JaxSubsequenceParallelEnvReplayBuffer):
    """Deterministic version of prioritized replay buffer."""

    def __init__(self,
                 observation_shape,
                 stack_size,
                 replay_capacity,
                 batch_size,
                 update_horizon=1,
                 subseq_len=0,
                 n_envs=1,
                 gamma=0.99,
                 max_sample_attempts=1000,
                 extra_storage_types=None,
                 observation_dtype=np.uint8,
                 terminal_dtype=np.uint8,
                 action_shape=(),
                 action_dtype=np.int32,
                 reward_shape=(),
                 reward_dtype=np.float32):
        super().__init__(observation_shape=observation_shape,
                         stack_size=stack_size,
                         replay_capacity=int(replay_capacity),
                         batch_size=batch_size,
                         update_horizon=update_horizon,
                         gamma=gamma,
                         max_sample_attempts=max_sample_attempts,
                         extra_storage_types=extra_storage_types,
                         observation_dtype=observation_dtype,
                         terminal_dtype=terminal_dtype,
                         subseq_len=subseq_len,
                         n_envs=n_envs,
                         action_shape=action_shape,
                         action_dtype=action_dtype,
                         reward_shape=reward_shape,
                         reward_dtype=reward_dtype)

        self.sum_tree = sum_tree.DeterministicSumTree(self._replay_capacity)

    def get_add_args_signature(self):
        """The signature of the add function."""
        parent_add_signature = super().get_add_args_signature()
        add_signature = parent_add_signature + [
            ReplayElement('priority', (), np.float32)
        ]
        return add_signature

    def _add(self, *args):
        """Internal add method to add to the underlying memory arrays."""
        self._check_args_length(*args)

        # Use Schaul et al.'s (2015) scheme of setting the priority of new elements
        # to the maximum priority so far.
        # Picks out 'priority' from arguments and adds it to the sum_tree.
        transition = {}
        for i, element in enumerate(self.get_add_args_signature()):
            if element.name == 'priority':
                priority = np.asarray(args[i], dtype=np.float32)
            else:
                transition[element.name] = args[i]
        if (not np.all(np.isfinite(priority)) or
                np.any(priority < 0.0)):
            raise ValueError('Priorities must be finite and nonnegative.')

        indices = np.ravel_multi_index(
            (np.ones(
                (1,), dtype='int32') * self.cursor(), np.arange(self._n_envs)),
            (self._replay_length, self._n_envs),
        )

        for i in range(len(indices)):
            self.sum_tree.set(indices[i], priority[i])
        super()._add_transition(transition)

    def sample_index_batch(self,
                           batch_size,
                           subseq_len=None,
                           update_horizon=None):
        """Returns a batch of valid indices sampled as in Schaul et al. (2015)."""
        subseq_len = self._subseq_len if subseq_len is None else subseq_len
        update_horizon = (self._update_horizon if update_horizon is None else
                          update_horizon)
        # Sample stratified indices. Some of them might be invalid.
        # start = time.time()
        indices = self.sum_tree.stratified_sample(batch_size, self._rng)
        indices = np.array(indices)
        # print("Sampling from sum tree took {}".format(time.time() - start))
        allowed_attempts = self._max_sample_attempts

        t_indices, b_indices = self.unravel_indices(indices)  # pylint: disable=unbalanced-tuple-unpacking
        t_indices = np.array(t_indices, copy=True)
        b_indices = np.array(b_indices, copy=True)
        censor_before = np.zeros_like(t_indices)
        for i in range(len(indices)):
            is_valid, ep_start = self.is_valid_transition(
                t_indices[i:i + 1],
                b_indices[i:i + 1],
                subseq_len=subseq_len,
                update_horizon=update_horizon)
            censor_before[i] = ep_start
            if not is_valid:
                if allowed_attempts == 0:
                    raise RuntimeError(
                        'Max sample attempts: Tried {} times but only sampled {}'
                        ' valid indices. Batch size is {}'.format(
                            self._max_sample_attempts, i, batch_size))
                while (not is_valid) and allowed_attempts > 0:
                    # If index i is not valid keep sampling others. Note that this
                    # is not stratified.
                    self._rng, rng = jax.random.split(self._rng)
                    index = self.sum_tree.stratified_sample(1, rng=rng).item()
                    t_index, b_index = self.unravel_indices(index)  # pylint: disable=unbalanced-tuple-unpacking

                    allowed_attempts -= 1
                    t_indices[i] = t_index
                    b_indices[i] = b_index
                    is_valid, ep_start = self.is_valid_transition(
                        t_indices[i:i + 1],
                        b_indices[i:i + 1],
                        subseq_len=subseq_len,
                        update_horizon=update_horizon)
                    censor_before[i] = ep_start
                if not is_valid:
                    raise RuntimeError(
                        'Max sample attempts: Tried {} times but only sampled '
                        '{} valid indices. Batch size is {}'.format(
                            self._max_sample_attempts, i, batch_size))
        return t_indices, b_indices, censor_before

    def sample_transition_batch(
        self,
        rng,
        batch_size=None,
        indices=None,
        subseq_len=None,
        update_horizon=None,
        gamma=None,
    ):
        """Returns a batch of transitions with extra storage and the priorities."""
        transition = super().sample_transition_batch(
            rng,
            batch_size,
            indices,
            subseq_len=subseq_len,
            update_horizon=update_horizon,
            gamma=gamma,
        )
        base_elements = super().get_transition_elements(
            batch_size, subseq_len=subseq_len)
        indices_position = next(
            i for i, element in enumerate(base_elements)
            if element.name == 'indices')
        transition.append(self.get_priority(transition[indices_position]))
        return transition

    def set_priority(self, indices, priorities):
        """Sets the priority of the given elements according to Schaul et al."""
        indices = np.asarray(indices)
        if not np.issubdtype(indices.dtype, np.integer):
            raise TypeError('Indices must be integers, given: {}'.format(
                indices.dtype))
        indices = indices.reshape(-1)
        priorities = np.asarray(priorities, dtype=np.float32).reshape(-1)
        if priorities.size != indices.size:
            raise ValueError(
                'Priority count must match index count: got {} priorities '
                'for {} indices.'.format(priorities.size, indices.size))
        if (not np.all(np.isfinite(priorities)) or
                np.any(priorities < 0.0)):
            raise ValueError('Priorities must be finite and nonnegative.')
        if np.any(indices < 0) or np.any(indices >= self._replay_capacity):
            raise ValueError('Replay indices are out of bounds.')
        indices = indices.astype(np.int32, copy=False)
        for index, priority in zip(indices, priorities):
            self.sum_tree.set(index, priority)

    def get_priority(self, indices):
        """Fetches the priorities correspond to a batch of memory indices."""
        indices = np.asarray(indices)
        if not indices.shape:
            raise ValueError('Indices must be an array.')
        if not np.issubdtype(indices.dtype, np.integer):
            raise TypeError('Indices must be integers, given: {}'.format(
                indices.dtype))
        if np.any(indices < 0) or np.any(indices >= self._replay_capacity):
            raise ValueError('Replay indices are out of bounds.')
        indices = indices.astype(np.int32, copy=False)
        return np.asarray(self.sum_tree.get(indices), dtype=np.float32)

    def get_transition_elements(self, batch_size=None, subseq_len=None):
        """Returns a 'type signature' for sample_transition_batch."""
        batch_size = self._batch_size if batch_size is None else batch_size
        parent_transition_type = (super().get_transition_elements(
            batch_size, subseq_len=subseq_len))
        probablilities_type = [
            ReplayElement('sampling_probabilities', (batch_size,), np.float32)
        ]
        return parent_transition_type + probablilities_type

    def reset_priorities(self):
        self.sum_tree.reset_priorities()
