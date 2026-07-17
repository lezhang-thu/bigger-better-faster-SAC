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

**`06` (entropy true floor)** — the salvage variant: max(x_ent_coef, 3e-4)
via the new imag_entropy_floor flag; late protection only. Same games, pins,
and bands as 03; Pong recovering to ≥17 would also confirm early-stripping
was the replacement variant's harm mechanism.

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
