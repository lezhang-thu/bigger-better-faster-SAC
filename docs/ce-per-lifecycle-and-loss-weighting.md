# CE-PER Lifecycle and Loss Weighting

This document describes the sole supported prioritized-replay path in the
current BBF configuration. There is no delta-priority warmup, Retrace/TV
priority phase, uniform-sampling phase, or hard priority switch. PER is enabled
throughout training and always uses the per-anchor C51 cross-entropy.

Let:

- $c$ be the number of gradient updates completed in the current network-reset
  cycle;
- $H(c)$ and $\gamma(c)$ be the TD horizon and discount used to materialize a
  replay group;
- $\mathcal V_H$ be the replay roots valid for the current horizon;
- $\ell_i$ be the raw C51 cross-entropy for sampled root $i$;
- $\rho_i$ be the value stored in the sum-tree leaf; and
- $\mathcal G$ be one sampled update group. In the current configuration,
  $|\mathcal G|=2\times32=64$.

## Sampling schedule and valid support

The schedule is cycle-local. With

$$
p=\operatorname{clip}\left(\frac{c}{40{,}000},0,1\right),
\qquad d=0.1^p,
$$

the replay group is materialized using

$$
H(c)=\operatorname{clip}(\operatorname{round}(10d),1,10),
\qquad
\gamma(c)=\operatorname{clip}(1-0.03d,0.97,0.997).
$$

Thus rounding makes $H$ decrease stepwise from 10 to 1, while $\gamma$
anneals continuously from $0.97$ to $0.997$. By 40,000 cycle-local gradient
updates both are at their endpoints and remain there. Reaching the schedule
endpoint does not reset priorities or flush the lookahead group. See
[`td_schedule`](../bbf/agents/spr_agent.py#L61-L74).

The learned-model input is always six states (five transitions). A root must
therefore have

$$
F(H)=\max(H,5)
$$

usable future rows. The required stack history and future window may not cross
the circular write cursor. A nonterminal episode boundary such as a time limit
may not occur in the forward $F(H)$ transitions. An episode boundary in the
frame history is allowed; frames before it are zero-censored when the stack is
built. True terminals in the forward window also remain valid: the n-step
return, bootstrap, and model losses mask them explicitly. See
[`_required_future` and `is_valid_transition`](../bbf/replay_memory/subsequence_replay_buffer.py#L276-L315).

The sum-tree draw is rejection-filtered against this dynamic support. Ignoring
the stratification between draws, its marginal distribution is the leaf
distribution conditioned on current validity:

$$
P_H(i)=
\frac{\rho_i}{\sum_{j\in\mathcal V_H}\rho_j},
\qquad i\in\mathcal V_H.
$$

Using the current $H$ is intentional. Once $H$ becomes smaller than 10, this
admits roots that can supply the current TD endpoint and fixed model sequence
instead of rejecting them merely because they cannot supply the maximum
configured horizon. See
[`sample_index_batch`](../bbf/replay_memory/subsequence_replay_buffer.py#L317-L359).

## Priority lifecycle

### 1. Insertion and overwrite

A newly inserted transition receives the maximum recorded leaf value,
initially `1.0`. This gives new data a strong opportunity to be sampled before
its first measured CE priority. When the circular replay overwrites a row, the
new transition replaces that leaf as well.

Insertion priority does not bypass validity checks: a fresh row cannot be
sampled until its required frame history and future rows are available.

The sum tree's `max_recorded_priority` never decreases. It is now only the
insertion sentinel; production reset handling does not fill the tree with this
value. See
[`_store_transition`](../bbf/agents/spr_agent.py#L1719-L1738) and
[`DeterministicSumTree.set`](../bbf/replay_memory/deterministic_sum_tree.py#L130-L166).

### 2. Sampling and target construction

The agent samples all 64 roots in an update group using the leaves currently
in the tree. Replay materializes each root's scheduled $H$-step return,
$\gamma^H$ discount, terminal flag, endpoint state $S_H$, and fixed six-state
model sequence.

At $S_H$, the online policy probabilities mix the target critic's categorical
action distributions. The scheduled return and discount move that mixture,
which is then projected onto the fixed C51 support to form $T_i(z)$. See
[`sample_transition_batch`](../bbf/replay_memory/subsequence_replay_buffer.py#L364-L479)
and [C51 target construction](../bbf/agents/spr_agent.py#L983-L1013).

### 3. CE score and alpha exponent

For the replayed action, the raw priority score is the unweighted per-anchor
C51 cross-entropy

$$
\ell_i
=-\sum_z T_i(z)\log p_\theta(z\mid s_i,a_i).
$$

This is cross-entropy, not expected-value delta, total variation, or KL alone:
$\ell_i=H(T_i)+D_{\mathrm{KL}}(T_i\|p_\theta)$. Consequently, even a perfect
prediction of a nondegenerate target has a positive score equal to the target
entropy.

The score is stopped before replay writeback, so prioritization is not an
additional optimization objective. The stored leaf applies Google's fixed
priority exponent $\alpha=0.5$:

$$
\rho_i\leftarrow\sqrt{\ell_i+10^{-10}}.
$$

See [per-example CE extraction](../bbf/agents/spr_agent.py#L734-L745) and
[priority writeback](../bbf/agents/spr_agent.py#L1690-L1700).

The two $32$-sample minibatches are scanned sequentially by JAX. The second
minibatch is evaluated after the first minibatch's optimizer step, but all 64
CE scores survive the scan. Before writeback, cardinality checks require
exactly one score and one new leaf for every sampled index. This deliberately
avoids the upstream grouped-minibatch reduction bug. If a replay root occurs
more than once in a group, flattened writeback order applies and its last
occurrence determines the final leaf.

### 4. Ordinary persistence and one-group lag

Only sampled roots receive refreshed CE leaves. Every other leaf retains the
score produced by older network parameters and possibly an older $H$ and
$\gamma$ until it is sampled again or overwritten.

For CPU replay/GPU training overlap, the next 64-root group is sampled after
the current training call is dispatched but before the current group's new
leaves are written. The next sample uses the next value of the cycle-local
schedule. This creates one update group (two gradient updates in the current
configuration) of ordinary priority lag in both selection and captured leaf
weights, while avoiding Google's deeper two-group prefetch. See
[lookahead and writeback ordering](../bbf/agents/spr_agent.py#L1678-L1700).

### 5. Network and environment resets

A successful shrink-and-perturb network reset:

1. resets $c$ to zero, so the next fresh group uses $H=10$ and $\gamma=0.97$;
2. retains every sum-tree leaf, its relative ranking, and the historical
   maximum used for future insertions; and
3. discards the already-materialized lookahead group because its return and
   discount were built under the previous cycle's schedule.

The retained leaves describe the pre-reset network until their roots are
sampled and refreshed. This staleness is the requested Google-BBF reset
behavior; it is not priority uniformization. A skipped network reset leaves
both the schedule and pending group untouched. Ordinary Atari episode resets
also do not change replay priorities. See
[`reset_weights`](../bbf/agents/spr_agent.py#L1526-L1580).

The replay classes still expose `mean_priority()` and `reset_priorities()` as
generic utilities, but the production training path calls neither one.

## Importance weights

There is one importance-weight family. For every sampled raw leaf $\rho_i$,
the fixed $\beta=0.5$ weight is

$$
w_i=
\frac{(\rho_i+10^{-10})^{-1/2}}
{\max_{k\in\mathcal G}(\rho_k+10^{-10})^{-1/2}}.
$$

The maximum is taken across the complete 64-root group before it is reshaped
into two minibatches. Batch-max normalization makes $\max_i w_i=1$ and cancels
the replay-size and valid-support normalizers that would otherwise appear in
$(N P_H(i))^{-\beta}$. For a CE-refreshed leaf,
$\rho_i\approx\ell_i^{1/2}$, so the effective correction relative to its raw CE
is approximately $w_i\propto\ell_i^{-1/4}$. Insertion sentinels and stale
retained leaves need not correspond to the root's current CE.

The host code currently returns two arrays for compatibility with the training
signature, but they are identical. There is no $\beta=1$ critic correction and
no replay-wide mean-priority factor. See
[`replay_loss_weights`](../bbf/agents/spr_agent.py#L307-L329).

The same $w_i$ is used by the real critic and every objective anchored at that
sampled replay root:

| Optimized loss | Priority-derived weighting | Current configured scaling |
|---|---|---:|
| Real C51 critic cross-entropy | $w_i$ | 1 |
| SPR representation loss | $w_i$ | `spr_weight = 5` |
| Replay actor, including entropy | $w_i$ | Actor coefficient 1; scheduled `x_ent_coef` for entropy |
| Reward rollout and real-frame MSE | $w_i$ repeated over valid trajectory steps | `reward_weight = 1` |
| Continue BCE | $w_i$ repeated over valid trajectory steps | `continue_weight = 1` |
| Imagined actor and entropy | $w_i$ times imagined reach weight | `0.1` times the imagination ramp |
| Imagined value C51 loss | $w_i$ times imagined reach weight | `0.05` times the imagination ramp |
| Raw CE priority score $\ell_i$ | None | Stopped; replay writeback only |

Reward and continue losses retain their unweighted valid-mask denominator, so
PER changes each sampled trajectory's contribution without self-normalizing
the priority weights away. Imagined losses additionally use their temporal
reach weights. The imagination multiplier is zero for the first 2,000
cycle-local gradient updates, then ramps linearly to the configured coefficient
over the next 2,000; a successful network reset restarts that ramp.

See [critic and SPR weighting](../bbf/agents/spr_agent.py#L768-L773),
[reward/continue weighting](../bbf/agents/spr_agent.py#L812-L833),
[actor/imagination weighting](../bbf/agents/spr_agent.py#L895-L972), and the
[active configuration](../bbf/configs/BBF-100K.gin#L10-L45).

## Current behavior at a glance

- Replay is always prioritized.
- The priority metric is always per-anchor projected-C51 cross-entropy.
- Stored leaves use $\alpha=0.5$.
- Every replay-anchored optimized loss uses the same fixed, group-normalized
  $\beta=0.5$ importance weight.
- All 64 sampled anchors receive priority writeback.
- Valid sampling support follows the current $H$ and the five-transition model
  span.
- Lookahead is one update group.
- Schedule endpoints and episode resets do not alter priorities.
- Successful network resets retain priorities but discard the pending
  old-schedule group.
