# Row 40 Priority Lifecycle and Loss Weighting

Let $s_i$ be the raw priority score, $\rho_i$ the stored sum-tree priority,
and $P(i)=\rho_i/\sum_j\rho_j$ the sampling probability.

## Priority lifetime

1. **Insertion**

   A new replay transition receives the largest priority ever observed,
   initially `1.0`. This encourages new transitions to be sampled. When the
   circular buffer overwrites a transition, its old priority is also replaced.
   See [transition insertion](../bbf/agents/spr_agent.py#L3298).

2. **Dynamic n-step phase: `[0, 10,000)`**

   After sampling anchor $i$, the warmup computes

   $$
   \delta_i = \mathbb{E}[T_i^{n\text{-step}}] - Q(s_i,a_i),
   $$

   where the expectation uses the projected C51 target. The raw score is

   $$
   s_i = |\delta_i| + 10^{-6},
   $$

   and the stored priority is

   $$
   \rho_i = \sqrt{s_i + 10^{-10}}.
   $$

   The categorical critic itself is still trained using cross-entropy; delta
   only determines the PER priority.

3. **Ordinary refresh behavior**

   Only sampled anchors receive a newly computed priority. Other leaves retain
   their previous value until sampled, overwritten, or globally reset.
   Consequently, within the dynamic warmup, different leaves can reflect
   earlier values of $n$, $\gamma$, and older network parameters.

   There is also a normal one-update-group prefetch lag: the next batch is
   selected before the current batch's new priorities are written.

4. **Switch at update `10,000`**

   All populated leaves are set to the same historical maximum priority. The
   prefetched warmup batch is discarded, and a fresh, uniformly sampled batch
   becomes the first Retrace batch. See
   [phase-boundary handling](../bbf/agents/spr_agent.py#L3062).

5. **Retrace phase: `>= 10,000`**

   Define

   $$
   TV_i = \frac{1}{2}\sum_z
   \left|T_i^{\mathrm{Retrace}}(z)-p_\theta(z\mid s_i,a_i)\right|.
   $$

   The raw score is

   $$
   s_i = \operatorname{clip}(TV_i,0,1)^2,
   $$

   and the stored priority is

   $$
   \rho_i = \sqrt{s_i+10^{-10}}
   \approx \operatorname{clip}(TV_i,0,1).
   $$

   Thus no warmup delta-based ranking survives into Retrace.

6. **Successful shrink-and-perturb reset**

   The cycle counter returns to zero, all populated priorities are uniformized,
   and any pending Retrace-prefetched batch is discarded. The new dynamic
   n-step phase therefore begins uniformly. Ordinary Atari episode resets do
   **not** reset priorities.

One nuance: uniformization uses the historical maximum priority, not
necessarily `1.0`, and that maximum never decreases. Therefore old relative
rankings are removed, but the common reset magnitude may originate from the
earlier phase. Unsampled leaves retain that common sentinel until refreshed.
See [sum-tree reset](../bbf/replay_memory/deterministic_sum_tree.py#L128).

## How priorities weight losses

Two importance weights are constructed from each sampled stored priority
$\rho_i$ ([weight formulas](../bbf/agents/spr_agent.py#L616)).

The full $\beta=1$ TD correction is

$$
w_i^{\mathrm{TD}}
= \frac{\bar\rho}{\rho_i+10^{-10}}
\approx \frac{1}{N P(i)},
$$

where $\bar\rho$ is the replay-wide mean priority.

The batch-max-normalized $\beta=0.5$ auxiliary correction is

$$
w_i^{\mathrm{aux}}
= \frac{(\rho_i+10^{-10})^{-1/2}}
       {\max_{k\in\text{sampled group}}
        (\rho_k+10^{-10})^{-1/2}}.
$$

In row 40, this normalization is shared across the grouped $2\times32=64$
samples.

| Optimized loss | PER weight | Additional scaling |
|---|---:|---|
| Real C51 critic cross-entropy | $w_i^{\mathrm{TD}}$, $\beta=1$ | Applied in both dynamic n-step and Retrace phases |
| SPR representation loss | $w_i^{\mathrm{aux}}$, $\beta=0.5$ | `spr_weight = 5` |
| Replay actor, including entropy term | $w_i^{\mathrm{aux}}$ | Actor coefficient 1; entropy uses scheduled `x_ent_coef` |
| Reward rollout and real-frame MSE | $w_i^{\mathrm{aux}}$, repeated over trajectory steps | `reward_weight = 1` |
| Continue BCE | $w_i^{\mathrm{aux}}$, repeated over trajectory steps | `continue_weight = 1` |
| Imagined actor and its entropy | $w_i^{\mathrm{aux}}$ times imagined reach weight | `0.1` times imagination ramp |
| Imagined value C51 loss | $w_i^{\mathrm{aux}}$ times imagined reach weight | `0.05` times imagination ramp |
| PER priority score itself | None | Does not contribute gradients; only updates the replay tree |

The loss assembly is visible in
[main/SPR/model losses](../bbf/agents/spr_agent.py#L1549) and
[actor/imagination losses](../bbf/agents/spr_agent.py#L1692).

The practical interpretation is:

- The real TD critic is fully corrected for prioritized sampling.
- All auxiliary, actor, world-model, and imagined-value losses are only
  partially corrected with $\beta=0.5$.
- Immediately after either priority reset, all leaves are equal, so both
  weight families are approximately `1`.
- The imagined-value loss currently uses $\beta=0.5$ despite a nearby stale
  comment suggesting critic-style weighting; the executable code uses
  $\beta=0.5$.
