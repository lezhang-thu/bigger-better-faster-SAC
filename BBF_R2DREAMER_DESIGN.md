# BBF + R2-Dreamer Integration Design

> **Update (branch `stage0123`).** This branch merges the RSSM world model
> (commits `d0196c7`/`284d8e7`/`581e345`) into the pilot and implements the
> staged integration below. All stages are code-complete and selected purely
> via gin. The original pilot design (conv-TM imagination on the CNN
> representation) remains available and unchanged underneath.
>
> ## Architecture: two representations, one policy, bridged
>
> - **Anchor (unchanged):** BBF-SAC on the CNN representation — C51 n-step
>   PER Q-learning, SAC policy head, SPR, resets, replay ratio 64. Acting
>   stays stateless from the CNN policy.
>   **The anchor is `abf7870`, not the pilot.** Per-game evals (2026-07-11)
>   showed the pilot's conv-TM additions are game-dependent: ~tie on
>   Gopher/Krull/Frostbite but Hero −60%, Jamesbond −72%, Kangaroo 0.0 vs
>   5288.6. The pilot's `world_model_weight`/`imag_horizon` are therefore 0
>   by default; the RSSM below is the sole world-model/imagination path.
> - **World model (from `284d8e7`, proven):** the ported block-GRU RSSM
>   consumes `encode_project(s)` as its embedding, trains with KL + Barlow +
>   two-hot reward + continue losses on its own LaProp/AGC optimizer
>   (update period 16 gradient steps), with act-time posterior maintenance
>   used only to store sequence initials and the WM-batch refresh keeping
>   them fresh. Latents are never inputs to any RL head.
> - **Bridge (new):** `g: sg(feat) -> sg(representation)` (in
>   `R2DreamerWorldModel.bridge`), trained by MSE on real posterior states.
>   It is the decoder-free analog of a decoder: it decodes latents into the
>   RL feature space so the shared policy can run in imagination.
>   **Readout rule:** the bridge and value heads are strict readouts — their
>   losses never backprop through the RSSM into the encoder. (The bridge's
>   summed MSE is ~1000x the other losses early on; letting it reach the
>   encoder stalls the BBF anchor at -21 on Pong.) `r2_value_through_wm=True`
>   restores r2dreamer's repval-through-WM behavior for ablation.
>   **Isolation rule (default `r2_stop_encoder_grads=True`):** the WM
>   consumes `sg(representation)` — no WM loss shapes the encoder at all,
>   so the anchor's gradient stream is exactly `abf7870`. The 284d8e7-style
>   co-training (flag False) was Pong-validated only and left Kangaroo flat
>   (~50 mean through 61k vs baseline cluster 1682-3065).
> - **Feat value head (new):** 255-bin symexp-twohot critic on RSSM
>   features, trained on real-sequence lambda-returns bootstrapped per-step
>   from the BBF critic (`boot = E_pi[Q]`), plus a slow-value regularizer
>   from the target network — r2dreamer's `repval` grounded in BBF's value
>   scale.
> - **Imagination (Stage 2/3):** RSSM prior rollouts from the WM batch's
>   posterior states under the shared policy via the bridge; REINFORCE with
>   (optionally ReturnEMA-normalized) advantages + entropy on the shared
>   policy head; two-hot value loss on imagined lambda-returns. Rollouts are
>   fully stop-gradient; only the policy head and value head learn from
>   imagination. Imagination is frozen for `r2_imag_reset_freeze_steps` env
>   steps after each shrink-and-perturb reset.
> - Rewards entering the WM buffer are clipped to [-1, 1]
>   (`r2_world_model_clip_reward = True`) so reward/value/imagined-return
>   semantics match BBF's C51 critic.
>
> ## Stages and gates (all via gin)
>
> - **Stage 0** — `BBF-100K.gin` with `r2_imag_horizon = 0` and
>   `r2_bridge_weight = r2_value_weight = 0`: pilot + WM side-training only.
>   Gate: Pong matches the pilot baseline.
> - **Stage 1** — `BBF-100K.gin` as checked in (bridge + feat value on).
>   Gates (from `R2WM` log lines): `R2WMBridgeCos` high (>0.9),
>   `R2PolicyBridgeKL` small (the shared policy agrees through the bridge),
>   `R2WMRewardMAENonzero` low, `R2WMContAccTerminal` high,
>   `R2ValueBootMAE` small.
> - **Stage 2** — add `--gin_files=bbf/configs/stage2.gin` (H=5, actor/value
>   weight 0.02). Gate: no regression vs Stage 1 on 3 seeds.
> - **Stage 3** — add `--gin_files=bbf/configs/stage3.gin` (H=15, weight 0.3,
>   ReturnEMA advantage normalization). Watch `R2ImagReturn` calibration vs
>   real returns, `R2ImagEntropy`, `R2ReturnScale`.
>
> Original pilot design notes follow.

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
