# coding=utf-8
"""Focused tests for the discrete RSSM."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from bbf import rssm


class RSSMTest(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.model = rssm.RSSM(
            stoch_size=4,
            discrete_size=5,
            deter_size=7,
            hidden_size=11,
        )
        self.embed = jnp.zeros((2, 3, 6), jnp.float32)
        action_ids = jnp.array([[0, 1, 2], [2, 1, 0]])
        self.actions = jax.nn.one_hot(action_ids, 3)
        self.is_first = jnp.array(
            [[True, False, False], [True, False, True]])
        self.variables = self.model.init(
            {
                'params': jax.random.PRNGKey(0),
                'sample': jax.random.PRNGKey(1),
            }, self.embed, self.actions, self.is_first)

    def test_observe_shapes_one_hot_and_jit(self):
        apply = jax.jit(lambda key: self.model.apply(
            self.variables,
            self.embed,
            self.actions,
            self.is_first,
            rngs={'sample': key},
        ))
        post, prior = apply(jax.random.PRNGKey(2))
        self.assertEqual(post.deter.shape, (2, 3, 7))
        self.assertEqual(post.stoch.shape, (2, 3, 4, 5))
        self.assertEqual(prior.logits.shape, (2, 3, 4, 5))
        np.testing.assert_array_equal(np.asarray(post.stoch.sum(-1)), 1.0)
        feat = self.model.apply(self.variables, post, method=self.model.get_feat)
        self.assertEqual(feat.shape, (2, 3, 27))

    def test_deterministic_apply_does_not_require_sample_rng(self):
        first, _ = self.model.apply(
            self.variables,
            self.embed,
            self.actions,
            self.is_first,
            deterministic=True,
        )
        second, _ = self.model.apply(
            self.variables,
            self.embed,
            self.actions,
            self.is_first,
            deterministic=True,
        )
        np.testing.assert_array_equal(first.stoch, second.stoch)

    def test_is_first_clears_state_and_previous_action(self):
        arbitrary = rssm.RSSMState(
            deter=jnp.ones((2, 7)),
            stoch=jax.nn.one_hot(jnp.ones((2, 4), jnp.int32), 5),
            logits=jnp.ones((2, 4, 5)),
        )
        zeros = self.model.apply(
            self.variables, 2, deterministic=True, method=self.model.initial)
        embed = jnp.ones((2, 6))
        action = jax.nn.one_hot(jnp.array([1, 2]), 3)
        reset_post, reset_prior = self.model.apply(
            self.variables,
            arbitrary,
            action,
            embed,
            jnp.ones((2,), jnp.bool_),
            deterministic=True,
            method=self.model.obs_step,
        )
        zero_post, zero_prior = self.model.apply(
            self.variables,
            zeros,
            jnp.zeros_like(action),
            embed,
            jnp.zeros((2,), jnp.bool_),
            deterministic=True,
            method=self.model.obs_step,
        )
        np.testing.assert_allclose(reset_post.deter, zero_post.deter)
        np.testing.assert_allclose(reset_prior.logits, zero_prior.logits)

    def test_unimix_and_balanced_kl_gradient_boundaries(self):
        extreme = jnp.array([[[1000.0, -1000.0, -1000.0]]])
        mixed = rssm.unimix_logits(extreme, 0.01)
        self.assertGreaterEqual(
            float(jnp.min(jnp.exp(mixed))) + 1e-8, 0.01 / 3.0)

        post = jnp.array([[[1.0, -1.0], [-0.5, 0.5]]])
        prior = jnp.zeros_like(post)
        loss, dynamics, representation = rssm.balanced_kl_loss(
            post, prior, free_nats=0.0, balance=0.75)
        self.assertEqual(loss.shape, (1,))
        np.testing.assert_allclose(loss, 0.75 * dynamics + 0.25 * representation)

        dynamics_post_grad = jax.grad(lambda value: rssm.balanced_kl_loss(
            value, prior, free_nats=0.0, balance=1.0)[0].sum())(post)
        representation_prior_grad = jax.grad(lambda value:
            rssm.balanced_kl_loss(
                post, value, free_nats=0.0, balance=0.0)[0].sum())(prior)
        np.testing.assert_array_equal(dynamics_post_grad, 0.0)
        np.testing.assert_array_equal(representation_prior_grad, 0.0)

    def test_learned_initial_state(self):
        model = rssm.RSSM(
            stoch_size=2,
            discrete_size=3,
            deter_size=5,
            hidden_size=7,
            initial_mode='learned',
        )
        variables = model.init(
            {
                'params': jax.random.PRNGKey(4),
                'sample': jax.random.PRNGKey(5),
            },
            jnp.zeros((2, 1, 4)),
            jnp.zeros((2, 1, 3)),
            jnp.ones((2, 1), jnp.bool_),
        )
        initial = model.apply(
            variables, 2, deterministic=True, method=model.initial)
        self.assertEqual(initial.deter.shape, (2, 5))
        self.assertEqual(initial.stoch.shape, (2, 2, 3))
        self.assertIn('initial_deter', variables['params'])


if __name__ == '__main__':
    unittest.main()
