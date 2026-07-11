# coding=utf-8
"""Focused tests for Dreamer-style training helpers."""

import unittest

import jax.numpy as jnp
import numpy as np

from bbf.agents import spr_agent


class LambdaReturnTest(unittest.TestCase):

    def test_action_aligned_rewards_include_final_transition(self):
        rewards = jnp.asarray([[1.0, 2.0]])
        continues = jnp.asarray([[1.0, 1.0]])
        values = jnp.zeros((1, 3))
        result = spr_agent.lambda_return(
            rewards, continues, values, discount=0.9, lambd=1.0)
        np.testing.assert_allclose(result, [[2.8, 2.0]], rtol=1e-6)

    def test_lambda_zero_uses_post_action_value(self):
        rewards = jnp.asarray([[1.0, 2.0]])
        continues = jnp.asarray([[1.0, 0.0]])
        values = jnp.asarray([[10.0, 20.0, 30.0]])
        result = spr_agent.lambda_return(
            rewards, continues, values, discount=0.9, lambd=0.0)
        np.testing.assert_allclose(result, [[19.0, 2.0]], rtol=1e-6)


class InterpolateWeightsTest(unittest.TestCase):

    def test_preserves_parameter_container_type(self):
        old = {"params": {"layer": {"weight": jnp.asarray([1.0])}}}
        new = {"params": {"layer": {"weight": jnp.asarray([3.0])}}}
        dict_result = spr_agent.interpolate_weights(
            old, new, keys=None, old_weight=0.5, new_weight=0.5)
        self.assertIsInstance(dict_result, dict)

        frozen_old = spr_agent.FrozenDict(old)
        frozen_new = spr_agent.FrozenDict(new)
        frozen_result = spr_agent.interpolate_weights(
            frozen_old,
            frozen_new,
            keys=None,
            old_weight=0.5,
            new_weight=0.5,
        )
        self.assertIsInstance(frozen_result, spr_agent.FrozenDict)
        np.testing.assert_allclose(
            frozen_result["params"]["layer"]["weight"], [2.0])

if __name__ == "__main__":
    unittest.main()
