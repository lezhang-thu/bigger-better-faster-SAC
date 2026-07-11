# BBF + R2-Dreamer experiment ladder

This ladder changes one control boundary at a time. BBF remains the real-data
anchor throughout: its observation policy, distributional Q-learning, n-step
targets, SPR, prioritized replay, replay ratio, and reset schedule are not
replaced by RSSM features.

## Stages and promotion gates

| Stage | Added capability | Gate before promotion |
| --- | --- | --- |
| 0 | True pre-`abf7870` BBF/SAC algorithm (`2634d962`) | Reproduce the supplied ten-run Atari100k reference distribution before interpreting any hybrid stage. |
| 1 | Shadow decoder-free RSSM; BBF encoder is read-only | Training reward/continuation losses improve, prior/posterior entropies avoid collapse, and BBF score does not regress materially. Add a held-out/open-loop evaluator before claiming predictive quality. |
| 2 | Stopped-latent R2 actor bridge only | `R2BridgeKL` decreases while BBF parameters and RSSM predictive dynamics remain owned by their existing objectives. |
| 3 | Q-anchored latent value; no imagination | Value-to-target-Q error decreases without changing BBF's acting or TD representation. |
| 4 | Horizon-3 imagined value at weight `0.1`; actor is exactly zero | Imagined returns/value targets remain on the BBF Q scale and finite. Confirm no imagined gradient reaches the actor or world model. |
| 5 | First final-layer actor intervention; effective scale `1e-4` | The update-ratio cap is respected, real-policy update KL remains small, and paired-seed score is no worse than stage 4. |
| 6 (optional) | Raise only effective actor scale to `1e-3` | Run only after stage 5 passes the five-seed, multi-game gate; otherwise retain stage 5. |

## Baseline correction and evidence calibration

The true original algorithm is commit
`2634d962aed0c5102b8220062f5d31970b8fd3ff`, the parent of `abf7870`.
Stage 0 restores its raw flattened CNN representation, separate Q and policy
projections, policy-side SPR predictor, two-branch SPR loss, optimizer masks,
and reset ownership. The portable `bbf...` import fixes from `abf7870` remain;
they do not change the algorithm.

The supplied ten-run scores were produced by a version that is algorithmically
equivalent to `2634d962`. It was runnable in the authors' setup despite its
problematic import lines, and it has no `representation_projection` or other
architecture change. The scores are therefore the valid reference distribution
for stage 0.

Commit `1393080` is useful as a cautionary result, not as evidence that
Dreamer-style control broadly improves BBF. It used BBF's deterministic SPR
transition model rather than an RSSM, trained all auxiliary losses through the
main BBF update, had no Q anchor or slow value, and imagined with an
action/reward ordering different from the implementation in this ladder.

| Game | `2634d962`-equivalent BBF mean / median (10 runs) | `1393080` evaluation |
| --- | ---: | ---: |
| Gopher | 1179.5 / 1250.2 | 1259.0, 589.0 |
| Frostbite | 1600.1 / 1622.6 | 3026.3 |
| Hero | 7423.3 / 7692.0 | 2767.8 |
| Jamesbond | 1014.4 / 928.3 | 336.0 |
| Kangaroo | 2898.7 / 2064.0 | 0.0 |
| Krull | 7651.4 / 7932.0 | 8832.1 |

Hero, Jamesbond, and Kangaroo are below every one of the ten original-BBF
runs. Frostbite is high but within the baseline's broad range; Krull is only 16
points above its best run; and two Gopher runs are inconclusive. Across games,
the median ratio of `1393080` to the original-BBF mean is about 0.58. This is a
direct comparison against the corrected stage-0 architecture.

Accordingly, use Hero, Jamesbond, and Kangaroo as regression sentinels and
Frostbite and Krull as possible upside cases; Gopher remains useful for
variance. Use at least three matched seeds for screening and five for a
promotion decision. Compare paired per-game changes and an aggregate such as
IQM or median normalized improvement with bootstrap intervals. A single Pong
or favorable-game run is not a promotion signal.

## Commands

Run the original default workload (stage 0, Pong, GPU 0, run 11):

```bash
bash run-cuda0.sh
```

Select a stage positionally or with `STAGE`:

```bash
bash run-cuda0.sh 3
STAGE=4 bash run-cuda0.sh
# Optional only after stage 5 promotion:
bash run-cuda0.sh 6
```

Run paired deterministic seeds 11 through 15 on two games:

```bash
STAGE=4 DETERMINISTIC=1 RUN_START=11 RUN_END=15 \
  GAMES="Pong Breakout" GPU_ID=0 bash run-cuda0.sh
```

The default ladder treats the BBF encoder as read-only from the R2 side. To
run the explicit joint-encoder diagnostic, add `R2_JOINT_ENCODER=1`:

```bash
STAGE=1 R2_JOINT_ENCODER=1 DETERMINISTIC=1 \
  RUN_START=11 RUN_END=15 GAMES="Hero Jamesbond Kangaroo" bash run-cuda0.sh
```

For a single explicit seed, set `AGENT_SEED`. It takes precedence over using
the run number as the seed:

```bash
STAGE=2 RUN_START=21 RUN_END=21 AGENT_SEED=1807987954 bash run-cuda0.sh
```

For stage 0, the launcher loads only the Atari100k base and the all-off stage
override. Hybrid stages load, in order, the base, common R2 settings, and one
stage override. The equivalent direct structure for stage 5 is:

```bash
python -m bbf.train --agent=BBF \
  --gin_files=bbf/configs/BBF-100K.gin \
  --gin_files=bbf/configs/r2_ladder/common.gin \
  --gin_files=bbf/configs/r2_ladder/stage5_imag_actor_001.gin \
  --gin_bindings='DataEfficientAtariRunner.game_name="Pong"' \
  --run_number=11 --no_seeding=False
```

## Fixed semantics

- BBF PER contains real transitions only. Imagined transitions never enter PER,
  and priorities remain based only on the real distributional TD loss.
- R2 uses its own replay, PRNG stream, LaProp model state, optional encoder
  AdamW state, and control Adam state. Its observation adapter is inside the
  conditional R2 subtree; there is no extra projection in the BBF path. By
  default, auxiliary gradients do not update BBF's encoder. `R2_JOINT_ENCODER=1`
  is a separate diagnostic and must earn promotion independently.
- In numbered stages 1--4, no R2 update owns any parameter used by BBF acting
  or TD learning. With deterministic seeds, their BBF-side parameter trajectory
  should match stage 0; a mismatch is an ownership/RNG leak, not an expected
  experimental effect.
- The imagined control return uses BBF-clipped reward and BBF life-loss
  continuation semantics. If raw reward is also modeled, it is diagnostic and
  must not silently replace the clipped control reward.
- Discount remains `0.997`; imagined lambda is `0.95`; stages 4--6 use horizon
  3. Imagined-value learning begins at 10,000 environment actions and pauses
  for 1,000 actions after a BBF reset. Actor intervention starts only at 40,000
  actions and pauses for 2,000 actions after a reset.
- An R2-owned actor bridge maps RSSM features to the original actor's hidden
  size and then shares only BBF's final categorical `policy` layer. Real
  observations still use the original `policy_projection -> policy` path.
  Raw cached `(stoch, deter)` is never an input to the real Q-function or
  acting policy. Bridge inputs are stopped, so policy consistency updates the
  bridge rather than shaping RSSM posterior/dynamics.
- Imagined starts, dynamics, reward, continuation, return targets, and
  advantages are stopped. Stage 4 updates latent value only; stages 5--6 may
  update only the final categorical `policy` layer. Because Adam otherwise
  cancels loss scaling, the policy parameter step is explicitly multiplied by
  actor weight times the configured scale: `1e-4` in stage 5 and `1e-3` in
  optional stage 6. A second hard bound limits each auxiliary actor step to
  `1e-5` of the current final-layer parameter norm.
- The slow latent-value target uses EMA fraction `0.02`, and posterior value is
  anchored to the stopped BBF target-Q expectation with weight `1.0`.
- The 125k Atari-only sequence ring reserves about 7.4 GB for stacked states
  and cached full-size RSSM latents, and physically fills that storage over the
  run. Plan host memory accordingly.

## Metrics to retain for every stage

Always compare evaluation score, `DQNLoss`, `TD Error`, `SPRLoss`, policy
entropy, replay priorities, and encoder/actor gradient norms against stage 0.
For stages 1--6, log total world-model loss and its dynamics, representation,
Barlow, reward, continuation, prior-entropy, and posterior-entropy components,
plus the fraction of scheduled model updates that ran.

For stages 2--6, additionally retain `R2BridgeKL`,
`R2ValueAnchorLoss`, `R2WMGradNorm`, `R2EncoderGradNorm`,
`R2ValueGradNorm`, and `R2ImagActorGradNorm`. The last metric must be exactly
zero through stage 4. `R2EncoderUpdateEnabled` and `R2EncoderGradNorm` must
both be zero in the default ladder. For stages 4--6, retain `R2ImagRewardMean`,
`R2ImagContinueMean`, `R2ImagReturnMean`, `R2ImagReturnScale`,
`R2ImagWeightMean`, `R2ImagValueLoss`, `R2ImagActorLoss`,
`R2ImagActorUpdateRatio`, `R2ImagActorClipScale`, and
`R2RealPolicyUpdateKL`. Non-finite
values, representation collapse, rapidly increasing policy KL, an imagined
return scale inconsistent with target Q, or a stage-1 score regression are stop
conditions rather than reasons to advance the ladder.

The implementation keeps the latest values in `agent.r2_metrics` and emits
them with `logging.info` every 100 successful auxiliary updates. It does not
introduce a new experiment-tracking dependency, so preserve the job's text log
or connect this dictionary to your existing metric sink.

Before launching a full Atari100k run, the true-baseline contract,
two-branch-SPR, reduced-shape replay, gradient ownership, boundary, and JIT
integration checks can be run with:

```bash
python -m unittest \
  bbf.replay_memory.world_model_sequence_replay_test \
  bbf.agents.r2_integration_test
```

## `BBFAgent` configuration interface

The stage files expect these integration controls:

```text
r2_world_model_enabled
r2_control_bridge_enabled
r2_latent_value_enabled
r2_value_q_anchor_weight
r2_bridge_policy_consistency_weight
r2_imag_horizon
r2_imag_value_weight
r2_imag_actor_weight
r2_imag_actor_grad_scale
r2_actor_max_update_ratio
r2_imag_start_step
r2_imag_pause_after_reset
r2_actor_start_step
r2_actor_pause_after_reset
r2_imag_lambda
r2_imag_entropy_weight
r2_slow_value_fraction
```

`common.gin` also binds the `r2_world_model_*` optimizer, replay, cadence, and
RSSM-size arguments carried forward from the working periodic world-model
implementation.
