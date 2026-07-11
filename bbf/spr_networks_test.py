# coding=utf-8
"""Regression tests for the true-baseline SPR/RSSM network split."""

import unittest

import jax
import jax.numpy as jnp

from bbf import spr_networks


class BaselineSprStructureTest(unittest.TestCase):

    def test_two_branch_spr_has_separate_rssm_control_adapters(self):
        hidden_dim = 8
        network = spr_networks.RainbowDQNNetwork(
            num_actions=4,
            num_atoms=11,
            noisy=False,
            distributional=True,
            hidden_dim=hidden_dim,
            width_scale=1,
            rssm_stoch_size=2,
            rssm_discrete_size=2,
            rssm_deter_size=8,
            rssm_hidden_size=8,
            rssm_embed_dim=16,
        )
        observation = jnp.zeros((84, 84, 4))
        actions = jnp.zeros((2,), dtype=jnp.int32)
        support = jnp.linspace(-1.0, 1.0, 11)
        variables = network.init(
            jax.random.PRNGKey(0),
            method=network.init_fn,
            x=observation,
            support=support,
            actions=actions,
            do_rollout=True,
        )

        parameter_keys = set(variables["params"])
        self.assertNotIn("representation_projection", parameter_keys)
        self.assertTrue({
            "projection",
            "predictor",
            "policy_projection",
            "predict_policy",
            "q_projection",
            "actor_projection",
        }.issubset(parameter_keys))

        predictions = network.apply(
            variables,
            observation,
            actions,
            True,
            method=network.spr_from_observation,
        )
        target = network.apply(
            variables,
            observation,
            True,
            method=network.encode_project,
        )
        self.assertEqual(predictions.shape, (2, 2 * hidden_dim))
        self.assertEqual(target.shape, (2 * hidden_dim,))


if __name__ == "__main__":
    unittest.main()
