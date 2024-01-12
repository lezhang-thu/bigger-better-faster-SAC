import sys
import random

sys.path.insert(0, "/home/ubuntu/lezhang.thu/weird/")
import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1."
# coding=utf-8
r"""Entry point for Atari 100k experiments.

To train a BBF agent locally, run:

python -m bbf.train \
    --agent=BBF \
    --gin_files=bbf/configs/BBF.gin \
    --run_number=1

"""
import functools
import json
import os
import time

from absl import app
from absl import flags
from absl import logging
import gin
import numpy as np

import jax

from bigger_better_faster.bbf import eval_run_experiment
from bigger_better_faster.bbf.agents import spr_agent

FLAGS = flags.FLAGS
flags.DEFINE_multi_string(
    'gin_files', [], 'List of paths to gin configuration files (e.g.'
    '"dopamine/agents/dqn/dqn.gin").')
flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override the values set in the config files '
    '(e.g. "DQNAgent.epsilon_train=0.1",'
    '      "create_environment.game_name="Pong"").')
CONFIGS_DIR = './configs'
AGENTS = [
    'rainbow',
    'der',
    'dopamine_der',
    'DrQ',
    'OTRainbow',
    'SPR',
    'SR-SPR',
    'BBF',
]

# flags are defined when importing run_xm_preprocessing
flags.DEFINE_enum('agent', 'SPR', AGENTS, 'Name of the agent.')
flags.DEFINE_integer('run_number', 1, 'Run number.')
flags.DEFINE_integer('agent_seed', None, 'If None, use the run_number.')
flags.DEFINE_boolean('no_seeding', True, 'If True, choose a seed at random.')
flags.DEFINE_boolean('max_episode_eval', True,
                     'Whether to use `MaxEpisodeEvalRunner` or not.')
flags.DEFINE_boolean('eval_only', False, 'Only evaluation.')


def load_gin_configs(gin_files, gin_bindings):
    """Loads gin configuration files.

  Args:
    gin_files: list, of paths to the gin configuration files for this
      experiment.
    gin_bindings: list, of gin parameter bindings to override the values in the
      config files.
  """
    gin.parse_config_files_and_bindings(gin_files,
                                        bindings=gin_bindings,
                                        skip_unknown=False)


def create_agent(
    environment,
    seed,
    explore_end_steps,
):
    return spr_agent.BBFAgent(
        num_actions=environment.action_space.n,
        seed=seed,
        explore_end_steps=explore_end_steps,
    )


def set_random_seed(seed):
    """Set random seed for reproducibility."""
    logging.info('Setting random seed: %d', seed)
    random.seed(seed)
    np.random.seed(seed)


def main(unused_argv):
    """Main method.

    Args:
        unused_argv: Arguments (unused).
  """
    import logging as x_logging
    fmt = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
    formatter = x_logging.Formatter(fmt)
    logging.get_absl_handler().setFormatter(formatter)
    logging.set_verbosity(logging.INFO)

    gin_files = FLAGS.gin_files
    gin_bindings = FLAGS.gin_bindings
    print('Got gin bindings:')
    print(gin_bindings)
    gin_bindings = [b.replace("'", '') for b in gin_bindings]
    print('Sanitized gin bindings to:')
    print(gin_bindings)

    # Add code for setting random seed using the run_number
    if FLAGS.no_seeding:
        seed = int(time.time() * 10000000) % 2**31
    else:
        seed = FLAGS.run_number if not FLAGS.agent_seed else FLAGS.agent_seed
    set_random_seed(seed)
    load_gin_configs(gin_files, gin_bindings)

    # Set the Jax agent seed
    create_agent_fn = functools.partial(
        create_agent,
        seed=seed,
    )
    print("FLAGS.max_episode_eval: {}".format(FLAGS.max_episode_eval))
    runner_fn = eval_run_experiment.DataEfficientAtariRunner
    logging.info('Using MaxEpisodeEvalRunner for evaluation.')
    runner = runner_fn(create_agent_fn,)

    print(f'Found devices {jax.local_devices()}')

    runner.run_experiment(FLAGS.eval_only, seed)


if __name__ == '__main__':
    app.run(main)
