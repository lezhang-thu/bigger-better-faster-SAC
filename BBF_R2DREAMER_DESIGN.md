# BBF + R2-Dreamer Integration Design

## Goal

Combine BBF's strong off-policy Atari100k replay learning with R2-Dreamer's decoder-free world-model imagination. BBF remains the anchor learner; imagination is added as an auxiliary policy-improvement path rather than as synthetic replay.

## Core Ideas

1. Keep a single acting policy.

   The environment policy, replay policy loss, and imagined-rollout policy loss all update the same BBF policy head. This avoids two competing actors.

2. Keep BBF's real-data critic as the stable anchor.

   Distributional Q-learning, target networks, SPR, and prioritized replay stay driven by real replay-buffer transitions. Replay priorities remain based on real DQN TD error only.

3. Train a decoder-free latent world model from replay.

   The existing BBF encoder and transition model are extended with reward, continuation, and value heads. The model is trained with masked reward prediction, continuation prediction, and a Barlow Twins latent alignment loss.

4. Use imagination only as an auxiliary loss.

   Short imagined rollouts start from replay observations and follow the shared BBF policy in latent space. Lambda returns from predicted rewards, continuations, and values train the policy/value heads with small weights.

5. Do not put imagined transitions into replay.

   Synthetic transitions are not inserted into PER. This avoids contaminating BBF's distributional critic and replay priority distribution with model errors.

6. Match reward and terminal semantics.

   The first integration stage uses BBF-style Atari semantics: clipped rewards and life-loss continuation/terminal breaks. The world model, replay losses, and imagined returns should all agree on these semantics.

## Current Conservative Settings

- `world_model_weight = 1.0`
- `imag_horizon = 3`
- `imag_actor_weight = 0.02`
- `imag_value_weight = 0.02`
- `barlow_weight = 0.01`
- `barlow_lambd = 0.0005`

These settings intentionally make imagination weak at first. The expected workflow is to verify world-model metrics, then gradually increase horizon or auxiliary weights if training remains stable.

## Main Failure Modes To Watch

- Imagined actor loss overpowering real replay learning.
- Reward/continuation prediction learning the wrong semantics.
- Barlow loss dominating SPR or Q-learning gradients.
- Long imagined horizons amplifying model bias.
- PER priorities becoming misleading if synthetic losses are mixed into TD-error priority updates.

## Recommended Ablations

1. BBF baseline.
2. BBF plus world-model losses, no imagined actor/value.
3. Add horizon-3 imagination with small actor/value weights.
4. Sweep imagination weight before increasing horizon.
5. Only then test longer horizons or value-consistency losses.
