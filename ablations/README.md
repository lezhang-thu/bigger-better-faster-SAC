# Ablation plan (state as of 2026-07-16)

Where things stand: the full `c6e22ae`-config table (24 games, 2-3 seeds) puts
the integration at suite parity vs the 10-seed anchor baselines (game-level
HNS IQM 1.09 vs 1.17, mean 2.60 vs 2.62) — big wins on sparse-reward games
(Gopher/Jamesbond/Kangaroo/Frostbite/Pong), big losses on dense-reward games
(ChopperCommand −55%, BankHeist −42%, Asterix −41%, Breakout −25%). RewardCorr
*inverts* across the two columns (winners 0.25-0.73, losers 0.91-0.95), which
killed the RewardCorr-gating idea and points at the reward/continue grounding
gradients (they deliberately flow into encoder+TM; ~spr_agent.py line 640) as
the prime suspect for the dense-game losses.

Every script pins its imagination knobs as explicit `--gin_bindings`, so logs
are self-documenting and **mid-queue gin edits cannot contaminate runs** (the
failure mode that split the run-12 sweep). Scripts 01c/03/04/05 pin
`imag_discount=0.997` (fixed gamma, as at c6e22ae) for comparability with the
existing table; the going-forward default remains the annealed schedule.

All scripts: `[GAMES=...] [GPU=0] [RUN=nn] bash ablations/<script> > log 2>&1`.
Default RUN numbers are distinct per script (21/22/23/13/25/26/31-32).

## Step 1 — run in parallel on two boxes (~1 box-day each)

**Box A: `01a` (arm B) + `01b` (arm C)** — the loss-column decomposition on
Breakout + ChopperCommand (4 runs).

- B = imagination off, heads on. C = pure anchor (also yields the missing
  anchor-only wall-clock reference).
- Decision rule:
  - C not ≈ baseline → setup drifted; stop and investigate before anything else.
  - B ≈ C (recovered) → the imagined actor loss is the interference; skip arm
    D for these games; the lever becomes reward-density-aware actor weighting.
  - B low, C recovered → grounding gradients are the culprit → **Step 2: `01c`**.

**Box B: `02` (value gate)** — Kangaroo/Hero/Jamesbond/ChopperCommand/Pong at
`imag_value_weight=0.05` (5 runs). Red flags: Kangaroo <1000, Hero <5000,
Jamesbond <600, Pong <14; CC ≥7000 would mean the value signal helps there.
Control arm if ambiguous: `VW=0 RUN=14`.

## Step 2 — contingent on Step 1

**`01c` (arm D: readout heads)** — only if B stayed low and C recovered.
`reward_readout=True` stop-gradients the heads' rollout features; SPR becomes
the TM's sole supervisor (this also makes the MODEL_HORIZON comment's premise
true). Cannot hurt the sparse-game winners by construction. If Breakout/CC
recover here with imagination still on → adopt the readout as default, then
re-run the value gate on top of it later.

## Step 3 — cheap validations, fill spare box time anytime

**`03` (entropy replacement check)** — Pong + Gopher with the fixed 3e-4
coefficient as the only delta. One seed each suffices against their tight
reference clusters (Pong 17.41-20.41 ×5, Gopher 1533-1808 ×4).
RESULT 2026-07-17: Pong 11.54 FAIL, Gopher 1554 weak pass — replacement
semantics withdrawn as default (it also strips the early 1e-2 entropy).

**`06` (entropy true floor) — tested and removed.** The salvage variant,
max(x_ent_coef, 3e-4) late-only protection, also failed: Pong 14.11
(refs 17.41-20.41), Gopher 582.8 (below the baseline minimum). Pong's
dose-response (coupled 17.4-20.4 > floor 14.11 > replacement 11.54) says
the anneal-to-zero is load-bearing in imagination, not an oversight. The
imag_entropy_floor flag and this script were removed; the entropy line of
work is closed.

**`05` (seed fill-ins)** — 8 runs to make the 26-game table citable: Seaquest
×2, UpNDown ×2 (zero clean seeds), Frostbite +1 (the +113% is n=1), Freeway,
DemonAttack, RoadRunner +1 each.

## Step 4 — after Steps 1-2 settle the interference story

**`04` (deeper TM)** — `transition_hidden_layers=2`. Perturbs the anchor's own
SPR, so it gates on Pong/Gopher/Hero first; only then try it on loser games.

## Bookkeeping

- The working tree carries several uncommitted changes (PER-on-value fix,
  value 0.05 + entropy 3e-4 gin defaults, reward_readout flag,
  transition_hidden_layers port, this folder). Commit before launching long
  queues — running processes load whatever is on disk at launch time.
- Residual, unbindable delta vs true c6e22ae in all current-code runs: commit
  272a784 PER-weights the imagined actor loss (effective actor weight ~half of
  the unweighted 0.1). Note it when comparing decimal-for-decimal.

## Suite-scale rows added after the 2026-07-16 plan

The scripts numbered 07 and up postdate the plan above and run at full 26-game
suite scale; each carries a self-documenting header, so only the two newest are
catalogued here. Their common reference is the combo `07-suite-combo.sh`
(RUN=50: `reward_readout=True` + `imag_value_weight=0.05` + actor 0.1 + coupled
entropy + annealed discount); both below are single-delta variants of it.

**`15-suite-combo-rr4.sh` (RUN=80)** — the combo at replay ratio 4
(`replay_ratio=128`) instead of 2. `reset_every=10_000` ships with it and is
*not* optional: it is counted in env steps but encodes a gradient-step schedule
(`20_000 x RR2 = 5_000 x RR8 = 40_000` grad steps per reset cycle, per the
BBF-100K.gin note), so RR=4 needs resets twice as often; leaving it at 20_000
would double the cycle. `cycle_steps`/`imag_warmup` key on gradient steps and
adapt on their own, so they are deliberately untouched. Caveat: the
bbf-raw-scores.txt anchors are RR=2, so attributing anything to the combo *at*
RR=4 needs a base@RR4 control arm (not written yet); budget ~2x 07's box-time.

**`20-suite-combo-skip-60k-reset.sh` (RUN=130) — the strongest row so far.**
Scored in HNS against the 10-seed anchors (`bbf-raw-scores.txt`, git object
654cabc), the full 26-game row gives **canonical pooled IQM 1.326 / mean 2.839
/ median 0.975** — top of the ladder: combo 1.271, published SAC-BBF 1.088,
BBF@RR8 1.045, baseline 1.012. Combo's mean deficit is gone (DemonAttack +3.15
and Breakout +0.45 vs baseline, where combo was −6.4 and −3.5). It is n=1 per
game, where pooled and game-level coincide, and **the null at that granularity
is not 1.012**: resampling the baseline at one seed per game gives IQM 1.082,
95% range [0.788, 1.427], putting 1.326 at one-sided p ≈ 0.08. A second full
seed (~56 h) is the obvious next spend.

**OptGap is the one canonical metric it loses on**: 0.345 vs combo 0.327
(baseline 0.372). OptGap is linear in HNS so it carries no n=1 premium (null
mean exactly 0.372, 95% [0.317, 0.435] → p ≈ 0.20, weaker than IQM's 0.08),
and it counts only sub-human games — precisely the near-zero games IQM trims.
The deficit vs combo is two coin-game draws: Frostbite (+0.027 alone; combo
3226.6 vs this row's 257.6) and Hero (+0.011). Worst shortfall regressions vs
baseline: Frostbite +0.314, Kangaroo +0.236, Hero +0.201; best gains
KungFuMaster −0.248 (reaches human), Freeway −0.106, Qbert −0.100.

Two traps in reading its table by raw points: Hero's
1436-vs-7517 is only **−0.20 HNS** (denominator 29799) and lands in IQM's
trimmed bottom tail; DemonAttack's +3.51 HNS is inside both its anchor spread
(10200–62184) and combo's own DA n=5 spread. Neither supports the "restore
the reset, buy DA more updates" reading — though OptGap is where the Hero
concern legitimately lands, worth ~+0.008 there.

**`21-late-updates-gate.sh` (RUN=140 arm A / 141 arm B)** — keep the 60k reset,
double the gradient updates in the tail, via the new `late_update_after` /
`late_update_multiplier` knobs (env-step threshold, ×N update phases per env
step; default off, so every other row is unaffected). Motivated by how the two
rows split once RUN=130 reached 46 runs: skipping the reset buys **mean** in
IQM's trimmed tail (DemonAttack +7.79, UpNDown +6.02, Breakout +3.18 vs combo)
and gives back **mid-mass IQM** (Jamesbond −4.11, Krull −2.10,
ChopperCommand −1.32, RoadRunner −0.90, Asterix −0.46) — so the lever targets
combo's IQM and RUN=130's mean at once. Panel splits the two questions:
DemonAttack/UpNDown/Breakout ask whether extra updates deliver what the skipped
reset delivered, Asterix/RoadRunner whether the restored reset holds them at
combo level; bands in the header. Arm A is the stated design (tail = 160k grad
steps in one un-reset stretch); arm B adds `no_resets_after=110_000` so the 80k
reset stops being skipped, holding the cycle at the control's 80k grad.
~15 h/arm/seed; suite version ~78 h.

**`16-suite-readout-only.sh` (RUN=62)** — the combo minus the imagined value
loss (`imag_value_weight=0.0`, annealed discount kept). Arm R of the
DA-attribution design and the direct mirror of `12-suite-value-only.sh` (arm V,
RUN=60, the "-readout" row). Use `0.0`, never `None`: the weight is `float()`'d
unconditionally (spr_agent.py:1237) so `None` crashes the constructor before
training. The imagined actor (0.1) stays live, so only the model-generated
critic targets are removed. Pre-registered: value-only already convicted the
readout for DemonAttack (value-only DA 38923), so readout-only DA should stay
low (<=~18500, like combo's {13562, 18283}); >=~22000 would mean it takes both
ingredients. Prior readout+value-0 data (Asterix 8213, Gopher 1608, Breakout
324, BankHeist 46.2, CC {11161, 2329}) is all fixed-discount arm-D; this is its
annealed-discount, suite-scale counterpart.
