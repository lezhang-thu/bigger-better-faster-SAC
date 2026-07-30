import types
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np

from bbf.agents import spr_agent
from bbf.replay_memory import subsequence_replay_buffer


class OneStepBackupTest(unittest.TestCase):

    def test_configuration_guard_requires_both_horizons_to_be_one(self):
        spr_agent.validate_expected_one_step_backup(True, 1, 1)
        spr_agent.validate_expected_one_step_backup(True, 1, None)
        spr_agent.validate_expected_one_step_backup(False, 3, 10)

        with self.assertRaisesRegex(ValueError,
                                    "requires update_horizon=1"):
            spr_agent.validate_expected_one_step_backup(True, 3, 10)
        with self.assertRaisesRegex(ValueError,
                                    "requires update_horizon=1"):
            spr_agent.validate_expected_one_step_backup(True, 1, 10)

    def test_expected_c51_mixes_actions_and_ignores_sample(self):
        support = jnp.asarray([-1.0, 0.0, 1.0], dtype=jnp.float32)
        target_dist = types.SimpleNamespace(
            probabilities=jnp.asarray([[1.0, 0.0, 0.0],
                                       [0.0, 0.0, 1.0]], dtype=jnp.float32))
        policy_logits = jnp.log(
            jnp.asarray([0.25, 0.75], dtype=jnp.float32))

        def policy_with_sample(sample):
            return lambda state, rng: (policy_logits,
                                       jnp.asarray(sample, dtype=jnp.int32))

        common = dict(
            target_network=lambda _: target_dist,
            next_states=jnp.zeros((1,), dtype=jnp.float32),
            rewards=jnp.asarray(0.0, dtype=jnp.float32),
            terminals=jnp.asarray(False),
            support=support,
            cumulative_gamma=jnp.asarray(1.0, dtype=jnp.float32),
            rng=jax.random.PRNGKey(0),
        )
        expected_zero = spr_agent.target_output(
            policy_info=policy_with_sample(0),
            expected_one_step_backup=True,
            **common)
        expected_one = spr_agent.target_output(
            policy_info=policy_with_sample(1),
            expected_one_step_backup=True,
            **common)

        expected = np.asarray([0.25, 0.0, 0.75], dtype=np.float32)
        np.testing.assert_allclose(expected_zero, expected, atol=1e-6)
        np.testing.assert_allclose(expected_one, expected, atol=1e-6)

        sampled_zero = spr_agent.target_output(
            policy_info=policy_with_sample(0),
            expected_one_step_backup=False,
            **common)
        sampled_one = spr_agent.target_output(
            policy_info=policy_with_sample(1),
            expected_one_step_backup=False,
            **common)
        self.assertFalse(np.allclose(sampled_zero, sampled_one))

    def test_replay_one_step_uses_root_reward_terminal_and_next_state(self):
        replay = subsequence_replay_buffer.JaxSubsequenceParallelEnvReplayBuffer(
            observation_shape=(1,),
            stack_size=1,
            replay_capacity=16,
            batch_size=2,
            subseq_len=1,
            n_envs=1,
            update_horizon=1,
            gamma=0.9,
        )
        for row in range(8):
            replay.add(
                np.asarray([[row]], dtype=np.uint8),
                np.asarray([row], dtype=np.int32),
                np.asarray([row + 0.5], dtype=np.float32),
                np.asarray([row == 4], dtype=np.uint8),
            )

        replay.sample_index_batch = mock.Mock(return_value=(
            np.asarray([2, 4], dtype=np.int32),
            np.asarray([0, 0], dtype=np.int32),
            np.asarray([0, 0], dtype=np.int32),
        ))
        values = replay.sample_transition_batch(
            rng=jax.random.PRNGKey(0),
            batch_size=2,
            subseq_len=1,
            update_horizon=1,
            gamma=0.9,
        )
        sample = {
            element.name: value
            for element, value in zip(replay.get_transition_elements(2),
                                      values)
        }

        np.testing.assert_allclose(sample['return'][:, 0], [2.5, 4.5])
        np.testing.assert_array_equal(sample['terminal'][:, 0], [0, 1])
        np.testing.assert_allclose(sample['discount'][:, 0], [0.9, 0.9])
        np.testing.assert_array_equal(sample['state'][:, 0, 0, 0], [2, 4])
        np.testing.assert_array_equal(sample['next_state'][:, 0, 0, 0],
                                      [3, 5])


if __name__ == '__main__':
    unittest.main()
