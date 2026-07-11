# coding=utf-8
"""Small-shape tests for the isolated R2 auxiliary training path."""

import functools
import unittest

import flax
import gin
import jax
import jax.numpy as jnp
import numpy as np
import optax

from bbf import spr_networks
from bbf.agents import spr_agent


class R2IntegrationTest(unittest.TestCase):

    def _network_and_params(self, enabled=True, do_rollout=False):
        network = spr_networks.RainbowDQNNetwork(
            num_actions=4,
            num_atoms=11,
            noisy=False,
            distributional=True,
            hidden_dim=64,
            width_scale=0.25,
            r2_world_model_enabled=enabled,
            r2_world_model_stoch=4,
            r2_world_model_deter=32,
            r2_world_model_hidden=16,
            r2_world_model_discrete=4,
            r2_world_model_units=16,
            r2_world_model_blocks=4,
        )
        params = flax.core.FrozenDict(network.init(
            jax.random.PRNGKey(0),
            method=network.init_fn,
            x=jnp.zeros((16, 16, 4), dtype=jnp.float32),
            actions=jnp.zeros((2,), dtype=jnp.int32),
            do_rollout=do_rollout,
            support=jnp.linspace(-5.0, 5.0, 11),
        ))
        return network, params

    @staticmethod
    def _tree_norm(tree):
        leaves = jax.tree_util.tree_leaves(tree)
        return float(sum(jnp.sum(jnp.square(x)) for x in leaves) ** 0.5)

    def test_imagined_actor_gradient_is_policy_only(self):
        network, params = self._network_and_params()
        stoch, deter = network.apply(params, 3, method=network.r2_initial)

        def actor_loss(candidate):
            imagined = network.apply(
                candidate,
                stoch,
                deter,
                2,
                jax.random.PRNGKey(1),
                eval_mode=True,
                method=network.r2_imagine,
            )
            return -jnp.mean(imagined["log_probs"])

        grads = jax.grad(actor_loss)(params)["params"]
        self.assertGreater(self._tree_norm(grads["policy"]), 0.0)
        self.assertEqual(self._tree_norm(grads["policy_projection"]), 0.0)
        self.assertEqual(self._tree_norm(grads["r2_world_model"]), 0.0)
        self.assertEqual(self._tree_norm(grads["r2_value"]), 0.0)
        self.assertEqual(self._tree_norm(grads["encoder"]), 0.0)

    def test_enabling_r2_does_not_change_bbf_initial_parameters(self):
        _, baseline = self._network_and_params(enabled=False)
        _, integrated = self._network_and_params(enabled=True)
        for key, baseline_subtree in baseline["params"].items():
            integrated_subtree = integrated["params"][key]
            for baseline_leaf, integrated_leaf in zip(
                    jax.tree_util.tree_leaves(baseline_subtree),
                    jax.tree_util.tree_leaves(integrated_subtree)):
                np.testing.assert_array_equal(baseline_leaf, integrated_leaf)
        self.assertNotIn("observation_adapter", integrated["params"])
        self.assertNotIn("actor_bridge", integrated["params"])
        self.assertIn("observation_adapter",
                      integrated["params"]["r2_world_model"])
        self.assertIn("actor_bridge",
                      integrated["params"]["r2_world_model"])

    def test_stage0_has_true_pre_abf_network_contract(self):
        network, params = self._network_and_params(
            enabled=False, do_rollout=True)
        self.assertEqual(set(params["params"]), {
            "_log_alpha",
            "encoder",
            "transition_model",
            "projection",
            "predictor",
            "head",
            "policy_projection",
            "predict_policy",
            "policy",
        })
        self.assertNotIn("representation_projection", params["params"])

        state = jnp.arange(16 * 16 * 4, dtype=jnp.float32).reshape(16, 16, 4)
        support = jnp.linspace(-5.0, 5.0, 11)
        actions = jnp.asarray([0, 1], dtype=jnp.int32)
        output, policy_logits = network.apply(
            params,
            state,
            support,
            actions,
            True,
            True,
            method=network.init_fn,
        )
        encoded = network.apply(params, state, True, method=network.encode)
        raw_representation = encoded.reshape(-1)
        np.testing.assert_array_equal(output.representation,
                                      raw_representation)

        q_feature = network.apply(
            params, raw_representation, True, method=network.project)
        policy_feature = network.apply(
            params,
            raw_representation,
            True,
            method=lambda module, x, eval_mode: module.policy_projection(
                x, eval_mode),
        )
        spr_target = network.apply(
            params, state, True, method=network.encode_project)
        np.testing.assert_array_equal(
            spr_target, jnp.concatenate([q_feature, policy_feature], axis=-1))
        self.assertEqual(spr_target.shape, (2 * network.hidden_dim,))

        expected_policy_logits = network.apply(
            params,
            raw_representation,
            True,
            method=network.policy_logits_from_feature,
        )
        np.testing.assert_array_equal(policy_logits, expected_policy_logits)
        self.assertEqual(output.latent.shape, (actions.shape[0],
                                               2 * network.hidden_dim))

    def test_spr_loss_normalizes_two_branches_independently(self):
        predictions = jnp.asarray([[[3.0, 4.0, 0.0, 5.0]]])
        targets = jnp.asarray([[[0.0, 5.0, 12.0, 0.0]]])
        pred_branches = predictions.reshape(1, 1, 2, 2)
        target_branches = targets.reshape(1, 1, 2, 2)
        pred_branches /= jnp.linalg.norm(
            pred_branches, axis=-1, keepdims=True)
        target_branches /= jnp.linalg.norm(
            target_branches, axis=-1, keepdims=True)
        expected = jnp.square(pred_branches - target_branches).sum((-1, -2))
        actual = spr_agent._two_branch_spr_loss(predictions, targets)
        np.testing.assert_allclose(actual, expected)

    def test_post_adam_actor_step_scaling_distinguishes_ladder_stages(self):
        params = flax.core.FrozenDict({
            "policy": {"kernel": jnp.ones((3, 2))},
            "r2_value": {"kernel": jnp.ones((3, 1))},
        })
        optimizer = optax.adam(0.1)

        def update(actor_scale):
            grads = jax.tree_util.tree_map(jnp.ones_like, params)
            grads = grads.copy(add_or_replace={
                key: jax.tree_util.tree_map(
                    lambda grad: actor_scale * grad, grads[key])
                for key in ("policy",)
            })
            updated, _, _ = spr_agent._r2_control_optimizer_step(
                optimizer,
                optimizer.init(params),
                params,
                grads,
                actor_scale,
                1.0,
            )
            return updated

        stage5 = update(1e-4)
        stage6 = update(1e-3)
        stage5_delta = self._tree_norm(jax.tree_util.tree_map(
            lambda old, new: new - old,
            params["policy"], stage5["policy"]))
        stage6_delta = self._tree_norm(jax.tree_util.tree_map(
            lambda old, new: new - old,
            params["policy"], stage6["policy"]))
        self.assertAlmostEqual(stage6_delta / stage5_delta, 10.0, places=1)
        for stage5_leaf, stage6_leaf in zip(
                jax.tree_util.tree_leaves(stage5["r2_value"]),
                jax.tree_util.tree_leaves(stage6["r2_value"])):
            np.testing.assert_allclose(stage5_leaf, stage6_leaf)

    def test_world_model_only_aux_update_is_finite(self):
        network, params = self._network_and_params()
        encoder_keys = ("encoder",)
        control_keys = ("policy", "r2_value")
        encoder_params = spr_agent._param_subset(params, encoder_keys)
        control_params = spr_agent._param_subset(params, control_keys)
        encoder_optimizer = spr_agent.create_scaling_optimizer(1e-4)
        control_optimizer = optax.adam(1e-4)

        batch_size, batch_length = 2, 3
        raw_states = np.arange(
            batch_size * batch_length * 16 * 16 * 4,
            dtype=np.uint8).reshape(batch_size, batch_length, 16, 16, 4)
        action_index = np.arange(batch_size * batch_length).reshape(
            batch_size, batch_length) % 4
        actions = np.eye(4, dtype=np.float32)[action_index]
        rewards = np.zeros((batch_size, batch_length, 1), dtype=np.float32)
        terminals = np.zeros_like(rewards, dtype=np.bool_)
        is_first = np.zeros_like(rewards, dtype=np.bool_)
        is_first[:, 0] = True
        initial_stoch = np.zeros((batch_size, 4, 4), dtype=np.float32)
        initial_deter = np.zeros((batch_size, 32), dtype=np.float32)

        train_fn = jax.jit(
            spr_agent.r2_aux_train,
            static_argnames=spr_agent.r2_aux_static_argnames,
        )
        result = train_fn(
            network,
            params,
            params,
            spr_agent.r2_laprop_init(params["params"]["r2_world_model"]),
            encoder_optimizer,
            encoder_optimizer.init(encoder_params),
            control_optimizer,
            control_optimizer.init(control_params),
            params["params"]["r2_value"],
            raw_states,
            actions,
            rewards,
            terminals,
            is_first,
            initial_stoch,
            initial_deter,
            jnp.linspace(-5.0, 5.0, 11),
            jax.random.PRNGKey(2),
            jnp.float32,
            True,
            False,
            False,
            False,
            0,
            4e-5,
            0.9,
            0.999,
            1e-20,
            10,
            0.3,
            1e-3,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1e-5,
            0.997,
            0.95,
            3e-4,
            0.02,
            jnp.asarray(0.0),
            jnp.asarray(0.0),
        )
        metrics = result[6]
        self.assertEqual(result[7].shape, (batch_size, batch_length, 4, 4))
        self.assertEqual(result[8].shape, (batch_size, batch_length, 32))
        for value in metrics.values():
            self.assertTrue(np.isfinite(np.asarray(value)).all())
        self.assertEqual(float(metrics["R2ImagActorGradNorm"]), 0.0)
        self.assertEqual(float(metrics["R2ValueGradNorm"]), 0.0)
        self.assertEqual(float(metrics["R2EncoderUpdateEnabled"]), 0.0)
        self.assertEqual(float(metrics["R2EncoderGradNorm"]), 0.0)
        updated_params = result[0]
        for key, original_subtree in params["params"].items():
            if key == "r2_world_model":
                continue
            for original_leaf, updated_leaf in zip(
                    jax.tree_util.tree_leaves(original_subtree),
                    jax.tree_util.tree_leaves(updated_params["params"][key])):
                np.testing.assert_array_equal(original_leaf, updated_leaf)

    def test_agent_collection_reaches_jitted_aux_update(self):
        gin.clear_config()
        replay_selector = (
            "bbf.replay_memory.subsequence_replay_buffer."
            "PrioritizedJaxSubsequenceParallelEnvReplayBuffer")
        gin.bind_parameter(replay_selector + ".replay_capacity", 128)
        gin.bind_parameter(replay_selector + ".n_envs", 1)
        network = functools.partial(
            spr_networks.RainbowDQNNetwork,
            hidden_dim=64,
            width_scale=0.25,
        )
        agent = spr_agent.BBFAgent(
            num_actions=4,
            network=network,
            num_atoms=11,
            jumps=1,
            spr_weight=1.0,
            data_augmentation=True,
            batch_size=2,
            replay_ratio=2,
            batches_to_group=1,
            update_horizon=1,
            max_update_horizon=1,
            reset_every=-1,
            r2_world_model_enabled=True,
            r2_control_bridge_enabled=True,
            r2_latent_value_enabled=True,
            r2_world_model_batch_size=2,
            r2_world_model_batch_length=3,
            r2_world_model_replay_capacity=128,
            r2_world_model_update_period=1,
            r2_world_model_stoch=4,
            r2_world_model_deter=32,
            r2_world_model_hidden=16,
            r2_world_model_discrete=4,
            r2_world_model_units=16,
            r2_world_model_blocks=4,
            r2_value_q_anchor_weight=1.0,
            r2_bridge_policy_consistency_weight=1.0,
            r2_imag_horizon=2,
            r2_imag_value_weight=1.0,
            r2_imag_actor_weight=0.01,
            r2_imag_actor_grad_scale=0.1,
            r2_imag_start_step=0,
            r2_actor_start_step=0,
            seed=5,
        )
        observation = np.zeros((1, 84, 84, 1), dtype=np.uint8)
        agent.reset_all(observation)
        action = agent.step()
        for index in range(3):
            next_observation = np.full_like(observation, index + 1)
            agent.log_transition(
                next_observation,
                action,
                np.asarray([float(index % 2)], dtype=np.float32),
                np.asarray([False]),
                np.asarray([False]),
            )
            action = agent.step()

        self.assertEqual(agent._r2_replay.count(), 4)
        primary_rng = np.asarray(agent._rng).copy()
        params_before_aux = agent.online_params
        agent._r2_aux_update()
        np.testing.assert_array_equal(agent._rng, primary_rng)
        self.assertEqual(agent.r2_update_count, 1)
        self.assertEqual(agent.r2_metrics["R2UpdateRan"], 1.0)
        self.assertEqual(agent.r2_metrics["R2ImagHorizon"], 2.0)
        self.assertEqual(agent.r2_metrics["R2EncoderUpdateEnabled"], 0.0)
        self.assertEqual(agent.r2_metrics["R2EncoderGradNorm"], 0.0)
        self.assertAlmostEqual(
            agent.r2_metrics["R2ImagActorStepScale"], 0.001)
        self.assertLessEqual(
            agent.r2_metrics["R2ImagActorUpdateRatio"],
            agent.r2_actor_max_update_ratio * 1.0001)
        self.assertGreaterEqual(
            agent.r2_metrics["R2RealPolicyUpdateKL"], -1e-7)
        for key, old_subtree in params_before_aux["params"].items():
            if key in ("r2_world_model", "r2_value", "policy"):
                continue
            for old_leaf, new_leaf in zip(
                    jax.tree_util.tree_leaves(old_subtree),
                    jax.tree_util.tree_leaves(
                        agent.online_params["params"][key])):
                np.testing.assert_array_equal(old_leaf, new_leaf)
        policy_delta = self._tree_norm(jax.tree_util.tree_map(
            lambda old, new: new - old,
            params_before_aux["params"]["policy"],
            agent.online_params["params"]["policy"]))
        self.assertGreater(policy_delta, 0.0)
        self.assertTrue(all(np.isfinite(value)
                            for value in agent.r2_metrics.values()))

        # Stage 4 uses the same bridge/value/imagination path with actor weight
        # zero. Every BBF-owned subtree must then remain byte-identical.
        agent.r2_imag_actor_weight = 0.0
        params_before_value_only = agent.online_params
        agent._r2_aux_update()
        for key, old_subtree in params_before_value_only["params"].items():
            if key in ("r2_world_model", "r2_value"):
                continue
            for old_leaf, new_leaf in zip(
                    jax.tree_util.tree_leaves(old_subtree),
                    jax.tree_util.tree_leaves(
                        agent.online_params["params"][key])):
                np.testing.assert_array_equal(old_leaf, new_leaf)
        self.assertEqual(agent.r2_metrics["R2ImagActorGradNorm"], 0.0)
        self.assertEqual(agent.r2_metrics["R2RealPolicyUpdateKL"], 0.0)

        # Match the runner ordering: preserve the final row, reset recurrent
        # state, then log/flush a distinct first row.
        final_observation = np.full((84, 84, 1), 9, dtype=np.uint8)
        primary_rng = np.asarray(agent._rng).copy()
        agent.add_r2_episode_boundary_transition(
            0, final_observation, reward=3.0, terminal=True)
        np.testing.assert_array_equal(agent._rng, primary_rng)
        agent.reset_one(0)
        reset_observation = np.zeros((1, 84, 84, 1), dtype=np.uint8)
        agent.log_transition(
            reset_observation,
            action,
            np.asarray([1.0], dtype=np.float32),
            np.asarray([True]),
            np.asarray([True]),
        )
        agent.step()
        self.assertEqual(agent._r2_replay.count(), 6)
        boundary_slot, reset_slot = 4, 5
        self.assertEqual(agent._r2_replay._reward[boundary_slot, 0], 1.0)
        self.assertTrue(agent._r2_replay._is_terminal[boundary_slot, 0])
        self.assertFalse(agent._r2_replay._is_first[boundary_slot, 0])
        self.assertEqual(agent._r2_replay._reward[reset_slot, 0], 0.0)
        self.assertFalse(agent._r2_replay._is_terminal[reset_slot, 0])
        self.assertTrue(agent._r2_replay._is_first[reset_slot, 0])

        # The unchanged BBF optimizer/target update must also accept a Flax
        # tree containing the isolated R2 parameter subtrees.
        for index in range(8):
            next_observation = np.full_like(
                reset_observation, index + 10, dtype=np.uint8)
            agent.log_transition(
                next_observation,
                action,
                np.asarray([0.0], dtype=np.float32),
                np.asarray([False]),
                np.asarray([False]),
            )
            action = agent.step()
        agent.r2_world_model_update_period = 1000
        agent.initialize_prefetcher()
        agent._sample_from_replay_buffer()
        previous_grad_steps = agent.grad_steps
        agent._training_step_update(0)
        self.assertEqual(agent.grad_steps, previous_grad_steps + 1)
        gin.clear_config()


if __name__ == "__main__":
    unittest.main()
