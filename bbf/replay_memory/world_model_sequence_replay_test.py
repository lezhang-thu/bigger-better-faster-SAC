# coding=utf-8
"""Focused tests for the NumPy world-model sequence replay."""

from __future__ import absolute_import, division, print_function

import unittest

import numpy as np

from bbf.replay_memory import world_model_sequence_replay


class _FixedStarts(object):
    """Small Generator stand-in that makes sampled windows deterministic."""

    def __init__(self, starts):
        self._starts = np.asarray(starts, dtype=np.int64)

    def integers(self, low, high, size):
        if self._starts.shape != (size,):
            raise AssertionError((self._starts.shape, size))
        if np.any(self._starts < low) or np.any(self._starts >= high):
            raise AssertionError((self._starts, low, high))
        return self._starts.copy()


class WorldModelSequenceReplayTest(unittest.TestCase):

    def _buffer(self, capacity=12, batch_size=1, batch_length=3):
        return world_model_sequence_replay.WorldModelSequenceReplayBuffer(
            capacity=capacity,
            state_shape=(2, 2, 2),
            num_actions=3,
            stoch_shape=(2, 2),
            deter_shape=(3,),
            batch_size=batch_size,
            batch_length=batch_length,
            seed=7)

    def _state(self, value):
        return np.full((2, 2, 2), value, dtype=np.uint8)

    def _stoch(self, value):
        return np.full((2, 2), value, dtype=np.float32)

    def _deter(self, value):
        return np.full((3,), value, dtype=np.float32)

    def _add_row(self, replay, value, reward=None, is_first=False):
        if reward is None:
            reward = float(value)
        return replay.add(
            state=self._state(value),
            action=value % 3,
            reward=reward,
            is_terminal=False,
            is_first=is_first,
            stoch=self._stoch(value),
            deter=self._deter(value))

    def test_sample_uses_context_action_shift_and_arrival_reward(self):
        replay = self._buffer(batch_size=2, batch_length=3)
        self._add_row(replay, 0, reward=0.0, is_first=True)
        for value in range(1, 6):
            self._add_row(replay, value)

        batch = replay.sample(rng=_FixedStarts([0, 1]))

        np.testing.assert_array_equal(batch['state'][:, :, 0, 0, 0],
                                      [[1, 2, 3], [2, 3, 4]])
        # Actions come from context..T-1, not from the returned state rows.
        np.testing.assert_array_equal(np.argmax(batch['action'], axis=-1),
                                      [[0, 1, 2], [1, 2, 0]])
        np.testing.assert_array_equal(batch['reward'][Ellipsis, 0],
                                      [[1, 2, 3], [2, 3, 4]])
        np.testing.assert_array_equal(batch['initial_stoch'][:, 0, 0], [0, 1])
        np.testing.assert_array_equal(batch['initial_deter'][:, 0], [0, 1])
        np.testing.assert_array_equal(batch['context_row_ids'], [0, 1])
        np.testing.assert_array_equal(batch['row_ids'],
                                      [[1, 2, 3], [2, 3, 4]])
        self.assertNotIn('stoch', batch)
        self.assertNotIn('deter', batch)
        self.assertTrue(all(isinstance(value, np.ndarray)
                            for value in batch.values()))

    def test_life_or_game_boundary_is_two_rows_and_sequences_cross_it(self):
        replay = self._buffer(batch_length=3)
        replay.add_reset_transition(self._state(0), 0, self._stoch(0),
                                    self._deter(0))
        self._add_row(replay, 1, reward=1.0)
        replay.add_boundary_transition(
            state=self._state(2),
            reward=7.0,
            is_terminal=True,
            stoch=self._stoch(2),
            deter=self._deter(2))

        with self.assertRaisesRegex(ValueError, 'must be a separate is_first'):
            self._add_row(replay, 99)

        replay.add_reset_transition(self._state(3), 2, self._stoch(3),
                                    self._deter(3))
        self._add_row(replay, 4, reward=4.0)
        batch = replay.sample(rng=_FixedStarts([1]))

        np.testing.assert_array_equal(batch['state'][0, :, 0, 0, 0],
                                      [2, 3, 4])
        # a_1 enters terminal state 2; boundary a_2 is the zero sentinel and is
        # ignored by the reset at state 3; reset-row a_3 enters state 4.
        np.testing.assert_array_equal(batch['action'][0, 0], [0, 1, 0])
        np.testing.assert_array_equal(batch['action'][0, 1], [0, 0, 0])
        np.testing.assert_array_equal(batch['action'][0, 2], [0, 0, 1])
        np.testing.assert_array_equal(batch['reward'][0, :, 0], [7, 0, 4])
        np.testing.assert_array_equal(batch['is_terminal'][0, :, 0],
                                      [True, False, False])
        np.testing.assert_array_equal(batch['is_first'][0, :, 0],
                                      [False, True, False])

        other = self._buffer()
        with self.assertRaisesRegex(ValueError, 'separate rows'):
            other.add(
                self._state(0), 0, 0.0, True, True, self._stoch(0),
                self._deter(0))
        with self.assertRaisesRegex(ValueError, 'zero arrival reward'):
            other.add(
                self._state(0), 0, 1.0, False, True, self._stoch(0),
                self._deter(0))

    def test_write_back_refreshes_only_recomputed_rows(self):
        replay = self._buffer(batch_length=2)
        self._add_row(replay, 0, reward=0.0, is_first=True)
        for value in range(1, 4):
            self._add_row(replay, value)

        batch = replay.sample(rng=_FixedStarts([0]))
        new_stoch = np.stack([self._stoch(11), self._stoch(12)])[None]
        new_deter = np.stack([self._deter(11), self._deter(12)])[None]
        self.assertEqual(
            replay.write_back(batch['row_ids'], new_stoch, new_deter), 2)

        # Row 1 can later be used as context and now exposes its refreshed
        # posterior.  The original context row 0 was intentionally untouched.
        refreshed = replay.sample(batch_length=1, rng=_FixedStarts([1]))
        np.testing.assert_array_equal(refreshed['initial_stoch'][0],
                                      self._stoch(11))
        np.testing.assert_array_equal(refreshed['initial_deter'][0],
                                      self._deter(11))
        original = replay.sample(batch_length=1, rng=_FixedStarts([0]))
        np.testing.assert_array_equal(original['initial_stoch'][0],
                                      self._stoch(0))

    def test_wraparound_is_chronological_and_stale_write_back_is_safe(self):
        replay = self._buffer(capacity=5, batch_size=2, batch_length=2)
        self._add_row(replay, 0, reward=0.0, is_first=True)
        for value in range(1, 8):
            self._add_row(replay, value)

        # Retained IDs are 3..7, so valid T+1 start IDs are uniformly selected
        # from 3..5 even though both windows wrap physical storage.
        batch = replay.sample(rng=_FixedStarts([3, 5]))
        np.testing.assert_array_equal(batch['context_row_ids'], [3, 5])
        np.testing.assert_array_equal(batch['row_ids'], [[4, 5], [6, 7]])
        np.testing.assert_array_equal(batch['state'][:, :, 0, 0, 0],
                                      [[4, 5], [6, 7]])

        stale_ids = batch['row_ids'][0:1].copy()
        self._add_row(replay, 8)  # overwrites logical row 3
        self._add_row(replay, 9)  # overwrites logical row 4
        post_stoch = np.zeros((1, 2, 2, 2), dtype=np.float32)
        post_deter = np.zeros((1, 2, 3), dtype=np.float32)
        with self.assertRaises(world_model_sequence_replay.StaleReplayIndexError):
            replay.write_back(stale_ids, post_stoch, post_deter)
        # Row 4 is stale, row 5 is retained.
        self.assertEqual(
            replay.write_back(stale_ids,
                              post_stoch,
                              post_deter,
                              ignore_stale=True), 1)

    def test_invalidation_blocks_old_context_and_pre_reset_write_back(self):
        replay = self._buffer(batch_length=2)
        self._add_row(replay, 0, reward=0.0, is_first=True)
        for value in range(1, 4):
            self._add_row(replay, value)
        old_batch = replay.sample(rng=_FixedStarts([0]))
        old_generation = old_batch['generation']

        new_generation = replay.invalidate_latents()
        self.assertEqual(new_generation, 1)
        self.assertFalse(replay.can_sample())
        with self.assertRaisesRegex(RuntimeError, 'current-generation'):
            replay.sample(rng=_FixedStarts([0]))

        post_stoch = np.zeros((1, 2, 2, 2), dtype=np.float32)
        post_deter = np.zeros((1, 2, 3), dtype=np.float32)
        with self.assertRaises(
                world_model_sequence_replay.StaleLatentGenerationError):
            replay.write_back(old_batch['row_ids'],
                              post_stoch,
                              post_deter,
                              generation=old_generation)

        # Fresh online posteriors become valid contexts in the new generation.
        for value in range(4, 7):
            self._add_row(replay, value)
        self.assertTrue(replay.can_sample())
        fresh_batch = replay.sample(rng=_FixedStarts([0]))
        self.assertEqual(fresh_batch['context_row_ids'][0], 4)
        self.assertEqual(fresh_batch['generation'], new_generation)


if __name__ == '__main__':
    unittest.main()
