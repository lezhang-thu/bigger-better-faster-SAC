# BBF + R2-Dreamer Integration Design

## Implemented Principle

There is no teacher, student, or distillation loss. One categorical actor is
optimized from two sources:

- real replay states, using the real-data C51 critic;
- RSSM prior states, using a separate Dreamer value function and imagined
  lambda returns.

Neither source is assumed to be better. The real and imagined objectives are
independent evidence for the same actor, with explicit gradient and data
boundaries.

## State and Loss Paths

The common control state is a discrete RSSM feature
`[deterministic, flattened stochastic]`. The online environment policy keeps
the recurrent posterior state and previous action across steps. Episode and
life-loss resets clear both.

Each sampled replay sequence grounds C51 and the real actor at two positions:
the zero-context first posterior and the longest still-valid recurrent
posterior. This keeps bootstrap compatibility while training on the recurrent
feature distribution used by behavior.

| Path | Data | Updates | Does not update |
| --- | --- | --- | --- |
| C51 | real PER transitions | C51 head | encoder/RSSM, Dreamer value |
| Real actor | posterior replay features | shared actor | C51, encoder/RSSM |
| World model | replay sequences | encoder, RSSM, Barlow projector, reward/continue heads | actor, C51, value |
| Imagined actor | RSSM prior rollouts | shared actor | RSSM, reward/continue, C51 |
| Imagined value | RSSM prior rollouts | scalar value head | RSSM, C51 |

The old convolutional transition remains only for BBF's SPR objective; it is
not the imagination model. SPR follows the true pre-`abf7870` BBF structure:
the `projection/predictor` and `policy_projection/predict_policy` branches are
normalized independently and concatenated. The later
`representation_projection` module is intentionally absent. RSSM control uses
separate `q_projection` and `actor_projection` adapters because its feature
dimension differs from the flattened image encoder.

## World Model

The RSSM has a recurrent deterministic state and 16 independent 16-way
categorical variables, with straight-through sampling and 1% uniform mixing.
Training uses:

- separately weighted dynamics and representation KL terms with stopped
  opposite sides and free nats;
- R2-Dreamer Barlow alignment between projected posterior features and stopped
  encoder embeddings;
- one-step clipped-reward prediction;
- continuation prediction under the same life-loss semantics as BBF.

If a full-game boundary replaces the endpoint with a reset observation, the
terminal reward and continuation heads use the action-aligned prior feature

Replay rows are explicitly interpreted as
`(s_t, a_t, r_{t+1}, done_{t+1})`. An n-step target consumes rows
`t ... t+n-1`, includes the terminal transition reward, and bootstraps from
`s_{t+n}`.

## Imagination

Imagination starts from stopped replay posterior states. Each step performs:

1. sample an action from the shared actor;
2. advance the RSSM prior with that action;
3. predict reward and continuation from the post-action feature;
4. bootstrap with the slowly updated target value head.

Action-aligned lambda returns have shape `H` from rewards/continuations of
shape `H` and values of shape `H+1`. Advantages are normalized by a
weighted batch return scale. Actor and value gradients stop at imagined
features, so model error cannot directly train the RSSM through policy loss.

Imagined transitions are never added to replay, never train C51, and never
affect PER priorities.

For an n-step C51 bootstrap, replay currently supplies the endpoint image but
not every intermediate action beyond the sampled subsequence. The target RSSM
feature is therefore an observation-conditioned zero-context posterior. The
source critic is trained on both zero-context and recurrent posteriors to keep
that target meaningful. Returning full n-step action paths is a possible later
ablation, not part of the first comparison.

## Conservative Atari100k Settings

- RSSM sequence length: the existing six-state BBF subsequence
- posterior burn-in before imagination starts: 1 state
- imagination horizon: 5
- imagination warm-up: 10,000 gradient steps
- imagined actor/value weights: 0.02 / 0.02
- Barlow weight: 0.05
- target-network interpolation: 0.005

The encoder and world model survive BBF head resets. The encoder is no longer
shrink-perturbed at resets because that would invalidate the preserved RSSM
and prediction heads.

## Required Ablations

1. True original BBF algorithm at `2634d96` (`abf7870^`), with only the import
   path fixes needed to run it. Do not include `abf7870`'s
   `representation_projection` change in this baseline.
2. Common RSSM plus world-model losses, with both imagination weights zero.
3. Full integration with the configured warm-up and horizon 5.
4. No warm-up, to test whether early model bias caused the pilot collapses.
5. Actor weight and horizon sweeps only after reward, continuation, KL, return
   scale, and imagined-weight metrics are stable.

The current implementation deliberately uses short BBF replay subsequences
instead of R2-Dreamer's much longer batches. If the full integration is stable,
a separate lower-frequency long-sequence world-model sampler is the next
structural experiment; it should not be mixed into the first comparison.
