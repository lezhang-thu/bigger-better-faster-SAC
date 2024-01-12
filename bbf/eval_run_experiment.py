# coding=utf-8

#greedy_frac = 0.5
greedy_frac = 0
import random

import functools
import os
import sys
import time
import gym
import cv2

from absl import logging
import gin
import jax
import numpy as np

atari_human_scores = {
    'Alien': 7127.7,
    'Amidar': 1719.5,
    'Assault': 742.0,
    'Asterix': 8503.3,
    'Asteroids': 47388.7,
    'Atlantis': 29028.1,
    'BankHeist': 753.1,
    'BattleZone': 37187.5,
    'BeamRider': 16926.5,
    'Berzerk': 2630.4,
    'Bowling': 160.7,
    'Boxing': 12.1,
    'Breakout': 30.5,
    'Centipede': 12017.0,
    'ChopperCommand': 7387.8,
    'CrazyClimber': 35829.4,
    'DemonAttack': 1971.0,
    'DoubleDunk': -16.4,
    'Enduro': 860.5,
    'FishingDerby': -38.7,
    'Freeway': 29.6,
    'Frostbite': 4334.7,
    'Gopher': 2412.5,
    'Gravitar': 3351.4,
    'Hero': 30826.4,
    'IceHockey': 0.9,
    'Jamesbond': 302.8,
    'Kangaroo': 3035.0,
    'Krull': 2665.5,
    'KungFuMaster': 22736.3,
    'MontezumaRevenge': 4753.3,
    'MsPacman': 6951.6,
    'NameThisGame': 8049.0,
    'Phoenix': 7242.6,
    'Pitfall': 6463.7,
    'Pong': 14.6,
    'PrivateEye': 69571.3,
    'Qbert': 13455.0,
    'Riverraid': 17118.0,
    'RoadRunner': 7845.0,
    'Robotank': 11.9,
    'Seaquest': 42054.7,
    'Skiing': -4336.9,
    'Solaris': 12326.7,
    'SpaceInvaders': 1668.7,
    'StarGunner': 10250.0,
    'Tennis': -8.3,
    'TimePilot': 5229.2,
    'Tutankham': 167.6,
    'UpNDown': 11693.2,
    'Venture': 1187.5,
    'VideoPinball': 17667.9,
    'WizardOfWor': 4756.5,
    'YarsRevenge': 54576.9,
    'Zaxxon': 9173.3,
}

atari_random_scores = {
    'Alien': 227.8,
    'Amidar': 5.8,
    'Assault': 222.4,
    'Asterix': 210.0,
    'Asteroids': 719.1,
    'Atlantis': 12850.0,
    'BankHeist': 14.2,
    'BattleZone': 2360.0,
    'BeamRider': 363.9,
    'Berzerk': 123.7,
    'Bowling': 23.1,
    'Boxing': 0.1,
    'Breakout': 1.7,
    'Centipede': 2090.9,
    'ChopperCommand': 811.0,
    'CrazyClimber': 10780.5,
    'Defender': 2874.5,
    'DemonAttack': 152.1,
    'DoubleDunk': -18.6,
    'Enduro': 0.0,
    'FishingDerby': -91.7,
    'Freeway': 0.0,
    'Frostbite': 65.2,
    'Gopher': 257.6,
    'Gravitar': 173.0,
    'Hero': 1027.0,
    'IceHockey': -11.2,
    'Jamesbond': 29.0,
    'Kangaroo': 52.0,
    'Krull': 1598.0,
    'KungFuMaster': 258.5,
    'MontezumaRevenge': 0.0,
    'MsPacman': 307.3,
    'NameThisGame': 2292.3,
    'Phoenix': 761.4,
    'Pitfall': -229.4,
    'Pong': -20.7,
    'PrivateEye': 24.9,
    'Qbert': 163.9,
    'Riverraid': 1338.5,
    'RoadRunner': 11.5,
    'Robotank': 2.2,
    'Seaquest': 68.4,
    'Skiing': -17098.1,
    'Solaris': 1236.3,
    'SpaceInvaders': 148.0,
    'StarGunner': 664.0,
    'Surround': -10.0,
    'Tennis': -23.8,
    'TimePilot': 3568.0,
    'Tutankham': 11.4,
    'UpNDown': 533.4,
    'Venture': 0.0,
    'VideoPinball': 0.0,
    'WizardOfWor': 563.5,
    'YarsRevenge': 3092.9,
    'Zaxxon': 32.5,
}
atari_random_scores = {k.lower(): v for k, v in atari_random_scores.items()}
atari_human_scores = {k.lower(): v for k, v in atari_human_scores.items()}


def normalize_score(ret, game):
    return (ret - atari_random_scores[game]) / (atari_human_scores[game] -
                                                atari_random_scores[game])


def create_env_wrapper(create_env_fn):

    def inner_create(*args, **kwargs):
        env = create_env_fn(*args, **kwargs)
        env.cum_length = 0
        env.cum_reward = 0
        return env

    return inner_create


@gin.configurable
def create_atari_environment(game_name=None, sticky_actions=True):
    assert game_name is not None
    game_version = 'v0' if sticky_actions else 'v4'
    full_game_name = '{}NoFrameskip-{}'.format(game_name, game_version)
    env = gym.make(full_game_name)
    # Strip out the TimeLimit wrapper from Gym, which caps us at 100k frames. We
    # handle this time limit internally instead, which lets us cap at 108k frames
    # (30 minutes). The TimeLimit wrapper also plays poorly with saving and
    # restoring states.
    env = env.env
    env = AtariPreprocessing(env)
    return env


@gin.configurable
class AtariPreprocessing(object):
    """A class implementing image preprocessing for Atari 2600 agents.

  Specifically, this provides the following subset from the JAIR paper
  (Bellemare et al., 2013) and Nature DQN paper (Mnih et al., 2015):

    * Frame skipping (defaults to 4).
    * Terminal signal when a life is lost (off by default).
    * Grayscale and max-pooling of the last two frames.
    * Downsample the screen to a square image (defaults to 84x84).

  More generally, this class follows the preprocessing guidelines set down in
  Machado et al. (2018), "Revisiting the Arcade Learning Environment:
  Evaluation Protocols and Open Problems for General Agents".
  """

    def __init__(self,
                 environment,
                 frame_skip=4,
                 terminal_on_life_loss=False,
                 screen_size=84):
        """Constructor for an Atari 2600 preprocessor.

    Args:
      environment: Gym environment whose observations are preprocessed.
      frame_skip: int, the frequency at which the agent experiences the game.
      terminal_on_life_loss: bool, If True, the step() method returns
        is_terminal=True whenever a life is lost. See Mnih et al. 2015.
      screen_size: int, size of a resized Atari 2600 frame.

    Raises:
      ValueError: if frame_skip or screen_size are not strictly positive.
    """
        if frame_skip <= 0:
            raise ValueError(
                'Frame skip should be strictly positive, got {}'.format(
                    frame_skip))
        if screen_size <= 0:
            raise ValueError(
                'Target screen size should be strictly positive, got {}'.format(
                    screen_size))

        self.environment = environment
        self.terminal_on_life_loss = terminal_on_life_loss
        self.frame_skip = frame_skip
        self.screen_size = screen_size

        obs_dims = self.environment.observation_space
        # Stores temporary observations used for pooling over two successive
        # frames.
        self.screen_buffer = [
            np.empty((obs_dims.shape[0], obs_dims.shape[1]), dtype=np.uint8),
            np.empty((obs_dims.shape[0], obs_dims.shape[1]), dtype=np.uint8)
        ]

        self.game_over = False
        self.lives = 0  # Will need to be set by reset().

    @property
    def observation_space(self):
        # Return the observation space adjusted to match the shape of the processed
        # observations.
        return Box(low=0,
                   high=255,
                   shape=(self.screen_size, self.screen_size, 1),
                   dtype=np.uint8)

    @property
    def action_space(self):
        return self.environment.action_space

    @property
    def reward_range(self):
        return self.environment.reward_range

    @property
    def metadata(self):
        return self.environment.metadata

    def close(self):
        return self.environment.close()

    def reset(self):
        """Resets the environment.

    Returns:
      observation: numpy array, the initial observation emitted by the
        environment.
    """
        self.environment.reset()
        self.lives = self.environment.ale.lives()
        self._fetch_grayscale_observation(self.screen_buffer[0])
        self.screen_buffer[1].fill(0)
        return self._pool_and_resize()

    def render(self, mode):
        """Renders the current screen, before preprocessing.

    This calls the Gym API's render() method.

    Args:
      mode: Mode argument for the environment's render() method.
        Valid values (str) are:
          'rgb_array': returns the raw ALE image.
          'human': renders to display via the Gym renderer.

    Returns:
      if mode='rgb_array': numpy array, the most recent screen.
      if mode='human': bool, whether the rendering was successful.
    """
        return self.environment.render(mode)

    def step(self, action):
        """Applies the given action in the environment.

    Remarks:

      * If a terminal state (from life loss or episode end) is reached, this may
        execute fewer than self.frame_skip steps in the environment.
      * Furthermore, in this case the returned observation may not contain valid
        image data and should be ignored.

    Args:
      action: The action to be executed.

    Returns:
      observation: numpy array, the observation following the action.
      reward: float, the reward following the action.
      is_terminal: bool, whether the environment has reached a terminal state.
        This is true when a life is lost and terminal_on_life_loss, or when the
        episode is over.
      info: Gym API's info data structure.
    """
        accumulated_reward = 0.

        for time_step in range(self.frame_skip):
            # We bypass the Gym observation altogether and directly fetch the
            # grayscale image from the ALE. This is a little faster.
            _, reward, game_over, info = self.environment.step(action)
            accumulated_reward += reward

            if self.terminal_on_life_loss:
                new_lives = self.environment.ale.lives()
                is_terminal = game_over or new_lives < self.lives
                self.lives = new_lives
            else:
                is_terminal = game_over

            # We max-pool over the last two frames, in grayscale.
            if time_step >= self.frame_skip - 2:
                t = time_step - (self.frame_skip - 2)
                self._fetch_grayscale_observation(self.screen_buffer[t])

            if is_terminal:
                break

        # Pool the last two observations.
        observation = self._pool_and_resize()

        self.game_over = game_over
        return observation, accumulated_reward, is_terminal, info

    def _fetch_grayscale_observation(self, output):
        """Returns the current observation in grayscale.

    The returned observation is stored in 'output'.

    Args:
      output: numpy array, screen buffer to hold the returned observation.

    Returns:
      observation: numpy array, the current observation in grayscale.
    """
        self.environment.ale.getScreenGrayscale(output)
        return output

    def _pool_and_resize(self):
        """Transforms two frames into a Nature DQN observation.

    For efficiency, the transformation is done in-place in self.screen_buffer.

    Returns:
      transformed_screen: numpy array, pooled, resized screen.
    """
        # Pool if there are enough screens to do so.
        if self.frame_skip > 1:
            np.maximum(self.screen_buffer[0],
                       self.screen_buffer[1],
                       out=self.screen_buffer[0])

        transformed_image = cv2.resize(self.screen_buffer[0],
                                       (self.screen_size, self.screen_size),
                                       interpolation=cv2.INTER_AREA)
        int_image = np.asarray(transformed_image, dtype=np.uint8)
        return np.expand_dims(int_image, axis=2)


@gin.configurable
class Runner(object):

    def __init__(
        self,
        create_agent_fn,
        create_environment_fn=create_atari_environment,
        checkpoint_file_prefix='ckpt',
        logging_file_prefix='log',
        log_every_n=1,
        num_iterations=200,
        training_steps=250000,
        evaluation_steps=125000,
        max_steps_per_episode=27000,
        clip_rewards=True,
    ):

        self._logging_file_prefix = logging_file_prefix
        self._log_every_n = log_every_n
        self._num_iterations = num_iterations
        self._training_steps = training_steps
        print('self._training_steps: {}'.format(self._training_steps))

        self._evaluation_steps = evaluation_steps
        self._max_steps_per_episode = max_steps_per_episode
        self._clip_rewards = clip_rewards

        self._environment = create_environment_fn()
        self._agent = create_agent_fn(self._environment,
                                      explore_end_steps=training_steps -
                                      int(10e3))
        self._start_iteration = 0


@gin.configurable
class DataEfficientAtariRunner(Runner):
    """Runner for evaluating using a fixed number of episodes rather than steps.

  Also restricts data collection to a strict cap,
  following conventions in data-efficient RL research.
  """

    def __init__(
        self,
        create_agent_fn,
        game_name=None,
        create_environment_fn=create_atari_environment,
        num_eval_episodes=100,
        max_noops=30,
        parallel_eval=True,
        num_eval_envs=100,
        num_train_envs=4,
        eval_one_to_one=True,
    ):
        logging.info("game_name: {}".format(game_name))
        """Specify the number of evaluation episodes."""
        create_environment_fn = functools.partial(create_environment_fn,
                                                  game_name=game_name)
        super().__init__(create_agent_fn,
                         create_environment_fn=create_environment_fn)

        self._num_iterations = int(self._num_iterations)
        self._start_iteration = int(self._start_iteration)

        self._num_eval_episodes = num_eval_episodes
        logging.info('Num evaluation episodes: %d', num_eval_episodes)
        self._evaluation_steps = None
        self.num_steps = 0
        self.total_steps = self._training_steps * self._num_iterations
        self.create_environment_fn = create_env_wrapper(create_environment_fn)

        self.max_noops = max_noops
        self.parallel_eval = parallel_eval
        self.num_eval_envs = num_eval_envs
        self.num_train_envs = num_train_envs
        self.eval_one_to_one = eval_one_to_one

        self.train_envs = [
            self.create_environment_fn() for i in range(num_train_envs)
        ]
        self.train_state = None
        self._agent.reset_all(self._initialize_episode(self.train_envs))
        self._agent.cache_train_state()
        self.game_name = game_name.lower().replace('_', '').replace(' ', '')

    def _run_one_phase(self,
                       envs,
                       steps,
                       max_episodes,
                       run_mode_str,
                       needs_reset=False,
                       one_to_one=False,
                       resume_state=None):
        """Runs the agent/environment loop until a desired number of steps.

    We terminate precisely when the desired number of steps has been reached,
    unlike some other implementations.

    Args:
      envs: environments to use in this phase.
      steps: int, how many steps to run in this phase (or None).
      max_episodes: int, maximum number of episodes to generate in this phase.
      run_mode_str: str, describes the run mode for this agent.
      needs_reset: bool, whether to reset all environments before starting.
      one_to_one: bool, whether to precisely match each episode in
        `max_episodes` to an environment in `envs`. True is faster but only
        works in some situations (e.g., evaluation).
      resume_state: bool, whether to have the agent resume its prior state for
        the current mode.

    Returns:
      Tuple containing the number of steps taken in this phase (int), the
      sum of
        returns (float), and the number of episodes performed (int).
    """
        step_count = 0
        num_episodes = 0
        sum_returns = 0.

        (episode_lengths, episode_returns, state, envs) = self._run_parallel(
            episodes=max_episodes,
            envs=envs,
            one_to_one=one_to_one,
            needs_reset=needs_reset,
            resume_state=resume_state,
            max_steps=steps,
        )

        for episode_length, episode_return in zip(episode_lengths,
                                                  episode_returns):
            if run_mode_str == 'train':
                # we use one extra frame at the starting
                self.num_steps += episode_length
            step_count += episode_length
            sum_returns += episode_return
            num_episodes += 1
            sys.stdout.flush()
        return step_count, sum_returns, num_episodes, state, envs

    def _initialize_episode(self, envs):
        """Initialization for a new episode.

    Args:
      envs: Environments to initialize episodes for.

    Returns:
      action: int, the initial action chosen by the agent.
    """
        observations = []
        for env in envs:
            initial_observation = env.reset()
            if self.max_noops > 0:
                self._agent._rng, rng = jax.random.split(self._agent._rng  # pylint: disable=protected-access
                                                        )
                num_noops = jax.random.randint(rng, (), 0, self.max_noops)
                for _ in range(num_noops):
                    initial_observation, _, terminal, _ = env.step(0)
                    if terminal:
                        initial_observation = env.reset()
            observations.append(initial_observation)
        initial_observation = np.stack(observations, 0)

        return initial_observation

    def _run_parallel(self,
                      envs,
                      episodes=None,
                      max_steps=None,
                      one_to_one=False,
                      needs_reset=True,
                      resume_state=None):
        """Executes a full trajectory of the agent interacting with the environment.

    Args:
      envs: Environments to step in.
      episodes: Optional int, how many episodes to run. Unbounded if None.
      max_steps: Optional int, how many steps to run. Unbounded if None.
      one_to_one: Bool, whether to couple each episode to an environment.
      needs_reset: Bool, whether to reset environments before beginning.
      resume_state: State tuple to resume.

    Returns:
      The number of steps taken and the total reward.
    """
        # You can't ask for 200 episodes run one-to-one on 100 envs
        if one_to_one:
            assert episodes is None or episodes == len(envs)

        # Create envs
        live_envs = list(range(len(envs)))

        if needs_reset:
            new_obs = self._initialize_episode(envs)
            new_obses = np.zeros(
                (2, len(envs), *self._agent.observation_shape, 1))
            self._agent.reset_all(new_obs)

            rewards = np.zeros((len(envs),))
            terminals = np.zeros((len(envs),))
            episode_end = np.zeros((len(envs),))

            cum_rewards = []
            cum_lengths = []
        else:
            assert resume_state is not None
            (new_obses, rewards, terminals, episode_end, cum_rewards,
             cum_lengths) = (resume_state)

        total_steps = 0
        total_episodes = 0
        max_steps = np.inf if max_steps is None else max_steps
        step = 0

        # Keep interacting until we reach a terminal state.
        while True:
            b = 0
            step += 1
            episode_end.fill(0)
            total_steps += len(live_envs)
            actions = self._agent.step()

            # The agent may be hanging on to the previous new_obs, so we don't
            # want to change it yet.
            # By alternating, we can make sure we don't end up logging
            # with an offset.
            new_obs = new_obses[step % 2]

            # don't want to do a for-loop since live envs may change
            while b < len(live_envs):
                env_id = live_envs[b]
                obs, reward, d, _ = envs[env_id].step(actions[b])
                envs[env_id].cum_length += 1
                envs[env_id].cum_reward += reward
                new_obs[b] = obs
                rewards[b] = reward
                terminals[b] = d

                if (envs[env_id].game_over or
                        envs[env_id].cum_length == self._max_steps_per_episode):
                    total_episodes += 1
                    cum_rewards.append(envs[env_id].cum_reward)
                    cum_lengths.append(envs[env_id].cum_length)
                    envs[env_id].cum_length = 0
                    envs[env_id].cum_reward = 0

                    human_norm_ret = normalize_score(cum_rewards[-1],
                                                     self.game_name)

                    logging.info(
                        'steps executed: {:>8}, '.format(total_steps) +
                        'num episodes: {:>8}, '.format(len(cum_rewards)) +
                        'episode length: {:>8}, '.format(cum_lengths[-1]) +
                        'return: {:>8}, '.format(cum_rewards[-1]) +
                        'normalized return: {:>8}'.format(
                            np.round(human_norm_ret, 3)))

                    if one_to_one:
                        new_obses = delete_ind_from_array(new_obses, b, axis=1)
                        new_obs = new_obses[step % 2]
                        actions = delete_ind_from_array(actions, b)
                        rewards = delete_ind_from_array(rewards, b)
                        terminals = delete_ind_from_array(terminals, b)
                        self._agent.delete_one(b)
                        del live_envs[b]
                        b -= 1  # live_envs[b] is now the next env, so go back one.
                    else:
                        episode_end[b] = 1
                        new_obs[b] = self._initialize_episode([envs[env_id]])
                        self._agent.reset_one(env_id=b)
                    # debug - start
                    if not self._agent.eval_mode:
                        self._agent.greedy_action = random.random(
                        ) < greedy_frac  #not self._agent.greedy_action
                        #logging.info("self._agent.greedy_action: {}".format(
                        #    self._agent.greedy_action))
                    # debug - end
                elif d:
                    self._agent.reset_one(env_id=b)
                    # debug - start
                    if not self._agent.eval_mode:
                        self._agent.greedy_action = random.random(
                        ) < greedy_frac  #not self._agent.greedy_action
                        #logging.info("self._agent.greedy_action: {}".format(
                        #    self._agent.greedy_action))
                    # debug - end

                b += 1

            if self._clip_rewards:
                # Perform reward clipping.
                rewards = np.clip(rewards, -1, 1)

            self._agent.log_transition(new_obs, actions, rewards, terminals,
                                       episode_end)

            if (not live_envs or
                (max_steps is not None and total_steps > max_steps) or
                (episodes is not None and total_episodes > episodes)):
                break

        state = (new_obses, rewards, terminals, episode_end, cum_rewards,
                 cum_lengths)
        return cum_lengths, cum_rewards, state, envs

    def _run_train_phase(self,):
        """Run training phase.

    Returns:
      num_episodes: int, The number of episodes run in this phase.
      average_reward: float, The average reward generated in this phase.
      average_steps_per_second: float, The average number of steps per
      second.
    """
        # Perform the training phase, during which the agent learns.
        self._agent.eval_mode = False
        # debug - start
        self._agent.greedy_action = random.random() < greedy_frac  #False
        # debug - end
        self._agent.restore_train_state()
        start_time = time.time()
        (
            number_steps,
            sum_returns,
            num_episodes,
            self.train_state,
            self.train_envs,
        ) = self._run_one_phase(
            self.train_envs,
            self._training_steps,
            max_episodes=None,
            run_mode_str='train',
            needs_reset=self.train_state is None,
            resume_state=self.train_state,
        )
        average_return = sum_returns / num_episodes if num_episodes > 0 else 0.0
        human_norm_ret = normalize_score(average_return, self.game_name)
        time_delta = time.time() - start_time
        average_steps_per_second = number_steps / time_delta
        logging.info('Average undiscounted return per training episode: %.2f',
                     average_return)
        logging.info('Average normalized return per training episode: %.2f',
                     human_norm_ret)
        logging.info('Average training steps per second: %.2f',
                     average_steps_per_second)
        self._agent.cache_train_state()
        return (
            num_episodes,
            average_return,
            average_steps_per_second,
            human_norm_ret,
        )

    def _run_eval_phase(self,):
        """Run evaluation phase.

    Returns:
        num_episodes: int, The number of episodes run in this phase.
        average_reward: float, The average reward generated in this phase.
    """
        # Perform the evaluation phase -- no learning.
        self._agent.eval_mode = True
        # debug - start
        self._agent.greedy_action = True
        #self._agent.greedy_action = False
        # debug - end
        eval_envs = [
            self.create_environment_fn() for i in range(self.num_eval_envs)
        ]
        _, sum_returns, num_episodes, _, _ = self._run_one_phase(
            eval_envs,
            steps=None,
            max_episodes=self._num_eval_episodes,
            needs_reset=True,
            resume_state=None,
            one_to_one=self.eval_one_to_one,
            run_mode_str='eval',
        )
        average_return = sum_returns / num_episodes if num_episodes > 0 else 0.0
        logging.info(
            'Average undiscounted return per evaluation episode: %.2f',
            average_return,
        )
        human_norm_return = normalize_score(average_return, self.game_name)
        logging.info(
            'Average normalized return per evaluation episode: %.2f',
            human_norm_return,
        )
        return num_episodes, average_return, human_norm_return

    def _run_one_iteration(self,):
        """Runs one iteration of agent/environment interaction."""
        logging.info('Starting iteration %d', 0)
        (
            num_episodes_train,
            average_reward_train,
            average_steps_per_second,
            norm_score_train,
        ) = self._run_train_phase()
        #if True:
        if False:
            num_episodes_eval, average_reward_eval, human_norm_eval = (
                self._run_eval_phase())

    def run_experiment(self, eval_only=False, seed=None):
        """Runs a full experiment, spread over multiple iterations."""

        import orbax.checkpoint
        from flax.training import orbax_utils
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        directory = './single_save'
        import os
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_name = '{}/{}-{}.pth'.format(directory, self.game_name, seed)
        if not eval_only:
            logging.info('Beginning training...')
            self._run_one_iteration()
            # save jax model(s)
            ckpt = {
                'online_params': self._agent.online_params,
                'target_network_params': self._agent.target_network_params
            }
            save_args = orbax_utils.save_args_from_target(ckpt)
            orbax_checkpointer.save(file_name, ckpt, save_args=save_args)
        else:
            raw_restored = orbax_checkpointer.restore(file_name)
            self._agent.online_params = raw_restored['online_params']
            self._agent.target_network_params = raw_restored[
                'target_network_params']
            num_episodes_eval, average_reward_eval, human_norm_eval = (
                self._run_eval_phase())


def delete_ind_from_array(array, ind, axis=0):
    start = tuple(([slice(None)] * axis) + [slice(0, ind)])
    end = tuple(([slice(None)] * axis) +
                [slice(ind + 1, array.shape[axis] + 1)])
    tensor = np.concatenate([array[start], array[end]], axis)
    return tensor
