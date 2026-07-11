# coding=utf-8

import unittest

import jax
import numpy as np

from bbf.replay_memory import subsequence_replay_buffer


class SubsequenceReplayBufferTest(unittest.TestCase):

    def _make_buffer(self,
                     prioritized=False,
                     subseq_len=1,
                     update_horizon=1,
                     batch_size=4):
        buffer_cls = (
            subsequence_replay_buffer.
            PrioritizedJaxSubsequenceParallelEnvReplayBuffer
            if prioritized else
            subsequence_replay_buffer.JaxSubsequenceParallelEnvReplayBuffer)
        return buffer_cls(
            observation_shape=(1,),
            stack_size=1,
            replay_capacity=64,
            batch_size=batch_size,
            subseq_len=subseq_len,
            n_envs=1,
            update_horizon=update_horizon,
            gamma=0.5,
        )

    def _add_rows(self, replay, rewards, terminals=None):
        if terminals is None:
            terminals = [False] * len(rewards)
        for time_index, (reward, terminal) in enumerate(
                zip(rewards, terminals)):
            kwargs = {}
            if isinstance(
                    replay,
                    subsequence_replay_buffer.
                    PrioritizedJaxSubsequenceParallelEnvReplayBuffer):
                kwargs['priority'] = np.array([1.0], dtype=np.float32)
            replay.add(
                np.array([[10 + time_index]], dtype=np.uint8),
                np.array([time_index], dtype=np.int32),
                np.array([reward], dtype=np.float32),
                np.array([terminal], dtype=np.uint8),
                **kwargs,
            )

    def _sample(self,
                replay,
                indices,
                subseq_len=1,
                update_horizon=1,
                gamma=0.5):
        indices = np.asarray(indices, dtype=np.int32)
        batch_size = indices.size
        signature = replay.get_transition_elements(
            batch_size=batch_size, subseq_len=subseq_len)
        values = replay.sample_transition_batch(
            jax.random.PRNGKey(0),
            batch_size=batch_size,
            indices=indices,
            subseq_len=subseq_len,
            update_horizon=update_horizon,
            gamma=gamma,
        )
        return {element.name: value
                for element, value in zip(signature, values)}

    def test_one_step_uses_current_row_and_next_state(self):
        replay = self._make_buffer()
        self._add_rows(replay, [1.0, 2.0, 3.0])

        sample = self._sample(replay, [0], update_horizon=1)

        self.assertEqual(sample['state'][0, 0, 0, 0], 10)
        self.assertEqual(sample['next_state'][0, 0, 0, 0], 11)
        self.assertAlmostEqual(sample['return'][0, 0], 1.0)
        self.assertAlmostEqual(sample['discount'][0, 0], 0.5)
        self.assertFalse(sample['terminal'][0, 0])

    def test_multi_step_return_and_bootstrap_state(self):
        replay = self._make_buffer(update_horizon=3)
        self._add_rows(replay, [1.0, 2.0, 3.0, 4.0, 5.0])

        sample = self._sample(replay, [0], update_horizon=3)

        self.assertAlmostEqual(sample['return'][0, 0], 2.75)
        self.assertEqual(sample['next_state'][0, 0, 0, 0], 13)
        self.assertAlmostEqual(sample['discount'][0, 0], 0.125)
        self.assertFalse(sample['terminal'][0, 0])

    def test_terminal_reward_is_included_and_later_rewards_are_masked(self):
        replay = self._make_buffer(update_horizon=4)
        self._add_rows(
            replay,
            [1.0, 2.0, 9.0, 100.0, 5.0],
            [False, False, True, False, False],
        )

        sample = self._sample(replay, [0], update_horizon=4)

        # 1 + .5 * 2 + .25 * 9; the reward after done is excluded.
        self.assertAlmostEqual(sample['return'][0, 0], 4.25)
        self.assertTrue(sample['terminal'][0, 0])
        self.assertEqual(sample['next_state'][0, 0, 0, 0], 14)
        self.assertAlmostEqual(sample['discount'][0, 0], 0.5**4)
        bootstrap = (sample['return'][0, 0] + sample['discount'][0, 0] *
                     (1 - sample['terminal'][0, 0]) * 123.0)
        self.assertAlmostEqual(bootstrap, sample['return'][0, 0])

    def test_terminal_on_final_step_disables_bootstrap(self):
        replay = self._make_buffer(update_horizon=3)
        self._add_rows(
            replay,
            [1.0, 2.0, 9.0, 4.0],
            [False, False, True, False],
        )

        sample = self._sample(replay, [0], update_horizon=3)

        self.assertAlmostEqual(sample['return'][0, 0], 4.25)
        self.assertTrue(sample['terminal'][0, 0])
        self.assertEqual(sample['next_state'][0, 0, 0, 0], 13)
        self.assertAlmostEqual(sample['discount'][0, 0], 0.5**3)
        bootstrap = (sample['return'][0, 0] + sample['discount'][0, 0] *
                     (1 - sample['terminal'][0, 0]) * 123.0)
        self.assertAlmostEqual(bootstrap, sample['return'][0, 0])


    def test_same_trajectory_begins_valid_and_shifts_done_forward(self):
        replay = self._make_buffer(subseq_len=4)
        self._add_rows(
            replay,
            [0.0] * 6,
            [False, True, False, False, False, False],
        )

        sample = self._sample(
            replay, [0, 1], subseq_len=4, update_horizon=1)

        np.testing.assert_array_equal(
            sample['same_trajectory'][0], [1, 1, 0, 0])
        np.testing.assert_array_equal(
            sample['same_trajectory'][1], [1, 0, 0, 0])

    def test_runtime_subsequence_length_reaches_prioritized_sampler(self):
        replay = self._make_buffer(
            prioritized=True,
            subseq_len=1,
            update_horizon=1,
            batch_size=8,
        )
        self._add_rows(replay, [0.0] * 12)

        subseq_len = 4
        update_horizon = 2
        signature = replay.get_transition_elements(
            batch_size=8, subseq_len=subseq_len)
        signature_by_name = {element.name: element for element in signature}
        self.assertEqual(signature_by_name['state'].shape,
                         (8, subseq_len, 1, 1))
        self.assertEqual(signature_by_name['discount'].shape,
                         (8, subseq_len))
        self.assertEqual(signature_by_name['sampling_probabilities'].shape,
                         (8,))

        values = replay.sample_transition_batch(
            jax.random.PRNGKey(1),
            batch_size=8,
            subseq_len=subseq_len,
            update_horizon=update_horizon,
            gamma=0.5,
        )
        sample = {element.name: value
                  for element, value in zip(signature, values)}
        self.assertEqual(sample['state'].shape, (8, subseq_len, 1, 1))
        # With 12 written rows, the final valid anchor is 12 - L - H = 6.
        self.assertTrue(np.all(sample['indices'] <= 6))

    def test_priority_updates_flatten_and_validate_counts(self):
        replay = self._make_buffer(prioritized=True)
        indices = np.array([[1, 2]], dtype=np.int32)

        replay.set_priority(indices, np.array(0.25, dtype=np.float32))
        np.testing.assert_allclose(
            replay.get_priority(indices.reshape(-1)), [0.25, 0.25])

        with self.assertRaisesRegex(ValueError, 'Priority count'):
            replay.set_priority(
                indices,
                np.array([0.1, 0.2, 0.3], dtype=np.float32),
            )


if __name__ == '__main__':
    unittest.main()
