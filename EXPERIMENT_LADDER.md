# BBF + R2-Dreamer experiment ladder

This ladder changes one control boundary at a time. BBF remains the real-data
anchor throughout: its observation policy, distributional Q-learning, n-step
targets, SPR, prioritized replay, replay ratio, and reset schedule are not
replaced by RSSM features.

## Stages and promotion gates

| Stage | Added capability | Gate before promotion |
| --- | --- | --- |
| 0 | Original BBF/SAC baseline | Reproduce the expected Atari100k score and learning curve over at least three paired seeds. |
| 1 | Periodic decoder-free RSSM training | Training reward/continuation losses improve, prior/posterior entropies avoid collapse, and BBF score does not regress materially. Add a held-out/open-loop evaluator before claiming predictive quality. |
| 2 | R2 projector bridge plus Q-anchored latent value | Posterior policy-consistency KL and value-to-target-Q error decrease without changing the acting or TD representation. |
| 3 | Horizon-3 imagined value learning; actor weight is exactly zero | Imagined returns/value targets remain on the BBF Q scale and finite. Confirm no imagined gradient reaches the actor or world model. |
| 4 | Shared imagined-actor loss at weight `0.01` | Actor-gradient cap is respected, real-policy KL remains small, and paired-seed score is no worse than stage 3. |
| 5 | Raise only imagined-actor weight to `0.02` | Promote only if stage 4 improves or matches stage 3 across seeds; otherwise retain stage 4. |

Do not infer success from one Pong seed. Use the same seeds at adjacent stages,
and add at least one game with different reward density before increasing the
horizon or actor weight.

## Commands

Run the original default workload (stage 0, Pong, GPU 0, run 11):

```bash
bash run-cuda0.sh
```

Select a stage positionally or with `STAGE`:

```bash
bash run-cuda0.sh 3
STAGE=4 bash run-cuda0.sh
```

Run paired deterministic seeds 11 through 15 on two games:

```bash
STAGE=4 DETERMINISTIC=1 RUN_START=11 RUN_END=15 \
  GAMES="Pong Breakout" GPU_ID=0 bash run-cuda0.sh
```

For a single explicit seed, set `AGENT_SEED`. It takes precedence over using
the run number as the seed:

```bash
STAGE=2 RUN_START=21 RUN_END=21 AGENT_SEED=1807987954 bash run-cuda0.sh
```

The launcher loads, in order, the unchanged Atari100k base, the common R2
settings, and one stage override. The equivalent direct structure is:

```bash
python -m bbf.train --agent=BBF \
  --gin_files=bbf/configs/BBF-100K.gin \
  --gin_files=bbf/configs/r2_ladder/common.gin \
  --gin_files=bbf/configs/r2_ladder/stage4_imag_actor_001.gin \
  --gin_bindings='DataEfficientAtariRunner.game_name="Pong"' \
  --run_number=11 --no_seeding=False
```

## Fixed semantics

- BBF PER contains real transitions only. Imagined transitions never enter PER,
  and priorities remain based only on the real distributional TD loss.
- R2 uses its own replay, PRNG stream, LaProp model state, encoder AdamW state,
  and control Adam state. The periodic encoder step is intentionally separate
  from BBF's optimizer; for the strictest isolation diagnostic, override
  `BBFAgent.r2_world_model_train_encoder = False` and compare stage 1 again.
- The imagined control return uses BBF-clipped reward and BBF life-loss
  continuation semantics. If raw reward is also modeled, it is diagnostic and
  must not silently replace the clipped control reward.
- Discount remains `0.997`; imagined lambda is `0.95`; stages 3--5 use horizon
  3. Actor/value imagination begins at 10,000 environment actions and pauses
  for 1,000 actions after a BBF reset.
- The RSSM projector is the bridge into the shared actor. Raw cached
  `(stoch, deter)` is never an input to the real Q-function or acting policy.
- Imagined starts, dynamics, reward, continuation, return targets, and
  advantages are stopped. Stage 3 updates latent value only; stages 4--5 may
  update the shared policy. Because Adam otherwise cancels loss scaling, the
  policy parameter step is explicitly multiplied by actor weight times `0.1`:
  `0.001` in stage 4 and `0.002` in stage 5.
- The slow latent-value target uses EMA fraction `0.02`, and posterior value is
  anchored to the stopped BBF target-Q expectation with weight `1.0`.
- The 125k Atari-only sequence ring reserves about 7.4 GB for stacked states
  and cached full-size RSSM latents, and physically fills that storage over the
  run. Plan host memory accordingly.

## Metrics to retain for every stage

Always compare evaluation score, `DQNLoss`, `TD Error`, `SPRLoss`, policy
entropy, replay priorities, and encoder/actor gradient norms against stage 0.
For stages 1--5, log total world-model loss and its dynamics, representation,
Barlow, reward, continuation, prior-entropy, and posterior-entropy components,
plus the fraction of scheduled model updates that ran.

For stages 2--5, additionally retain `R2BridgeKL`,
`R2ValueAnchorLoss`, `R2WMGradNorm`, `R2EncoderGradNorm`,
`R2ValueGradNorm`, and `R2ImagActorGradNorm`. The last metric must be exactly
zero through stage 3. For stages 3--5, retain `R2ImagRewardMean`,
`R2ImagContinueMean`, `R2ImagReturnMean`, `R2ImagReturnScale`,
`R2ImagWeightMean`, `R2ImagValueLoss`, and `R2ImagActorLoss`. Non-finite
values, representation collapse, rapidly increasing policy KL, an imagined
return scale inconsistent with target Q, or a stage-1 score regression are stop
conditions rather than reasons to advance the ladder.

The implementation keeps the latest values in `agent.r2_metrics` and emits
them with `logging.info` every 100 successful auxiliary updates. It does not
introduce a new experiment-tracking dependency, so preserve the job's text log
or connect this dictionary to your existing metric sink.

Before launching a full Atari100k run, the reduced-shape replay, gradient
ownership, boundary, and JIT integration checks can be run with:

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
r2_imag_start_step
r2_imag_pause_after_reset
r2_imag_lambda
r2_imag_entropy_weight
r2_slow_value_fraction
```

`common.gin` also binds the `r2_world_model_*` optimizer, replay, cadence, and
RSSM-size arguments carried forward from the working periodic world-model
implementation.
