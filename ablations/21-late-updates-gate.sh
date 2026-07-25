set -ex
# Keep the 60k reset, double the gradient updates in the last phase.
# User's design, launched on their call after the RUN=130 (skip-60k) row grew
# to 46 runs. Matched control is 07-suite-combo.sh (RUN=50/51): the ONLY delta
# between combo and RUN=130 is no_resets_after (100_000 vs 80_000), so combo
# is a true single-delta with-reset control for every band below.
#
# WHY THIS IS THE RIGHT SHAPE OF EXPERIMENT. With all seeds counted, the two
# rows split by metric (game-level HNS, unequal seed counts so never pooled):
#   combo    (n=1-5)  IQM 1.249  mean 2.322  median 0.996
#   RUN=130  (n=1-3)  IQM 1.225  mean 2.671  median 0.960   OptGap 0.350
#   baseline (n=10)   IQM 1.163  mean 2.495  median 0.853   OptGap 0.372
# Skipping the reset buys MEAN (DemonAttack +7.79, UpNDown +6.02, Breakout
# +3.18 vs combo) and gives back a little IQM (Jamesbond -4.11, Krull -2.10,
# ChopperCommand -1.32, RoadRunner -0.90, Asterix -0.46). The gains are all in
# IQM's trimmed tail; the losses are mid-mass. So "restore the reset, buy the
# tail back with updates" is aimed exactly at combo's IQM + RUN=130's mean.
# None of the per-game losses clears its own noise bar (Jamesbond's tail is
# 4.31 HNS, ChopperCommand 0.92, Krull n=1 on both sides), so the bands below
# are the test, not the RUN=130 table.
#
# WHAT THE KNOB DOES. late_update_after is in env steps (like reset_every);
# from it onwards the agent runs late_update_multiplier x the update phases
# per env step. At the gin default (replay_ratio=64 -> 2 grad steps per env
# step) a multiplier of 2 makes the tail RR=4. Schedules keyed on
# cycle_grad_steps (update horizon 10->3, gamma 0.97->0.997, imag_warmup)
# anneal proportionally faster in the tail, exactly as under a global
# replay_ratio=128, per the BBF-100K.gin note and 15-suite-combo-rr4.sh.
#
# ARM C (default, cheapest, and the cleanest delta) = resets at 20k/40k/60k
# unchanged, 60k-80k at the ordinary RR2, only 80k-100k doubled. ARM A doubles
# the whole 60k-100k tail. ARM B is A with the reset cadence held (adds
# no_resets_after=110_000 so the 80k reset stops being skipped by the
# recovery-horizon rule, next_reset 100_001 > 100_000, rather than by design).
#
#                    resets              total grad  final un-reset stretch  h/game
#   combo (RUN=50)   20k/40k/60k            200k      80k grad               2.14
#   RUN=130          20k/40k                200k     120k grad               2.14
#   arm C            20k/40k/60k            240k     120k grad (40k+80k)     2.56
#   arm A            20k/40k/60k            280k     160k grad               2.99
#   arm B            20k/40k/60k/80k        280k     2 x 80k grad            2.99
#
# WHY C IS THE CLEAN ONE, not just the cheap one. cycle_grad_steps drives the
# update-horizon (10->3) and gamma (0.97->0.997) anneals over cycle_steps=10_000
# GRAD steps, i.e. 5_000 env steps at RR2, so after the 60k reset they finish
# at ~65k env; imag_warmup finishes at ~62k. Doubling from 80k therefore leaves
# every schedule identical to the control (traced: anneal completes at env
# 65004 in both) -- the ONLY thing that changes is the number of gradient
# updates. Arm A doubles from 60k and so also compresses those anneals into
# 2_500 env steps (completes at 62504), bundling two deltas.
# Note also that x_ent_coef is EXACTLY 0 from step 80_000 on (the clip in
# linearly_decaying_epsilon pins it), and it is the only entropy regularization
# live -- the SAC learned-alpha form is behind `if False` and _log_alpha is
# inert -- and imagination shares the same coefficient at imag_entropy_weight=
# None. So arm C's doubled window is precisely the entropy-free phase, in both
# actor channels, with no re-randomized head coming either (last reset 60k).
# Watch ImagEntropy (already logged every 500 grad steps). RUN=130 baseline,
# median over 46 runs: 1.452 (40-60k) -> 1.301 (60-80k) -> 1.131 (80-90k) ->
# 1.109 (90-100k) -- a gentle, decelerating drift, no collapse at RR2. Doubling
# should roughly double the per-env-step drift; ~1.1 nats is the headroom.
# If it DOES collapse, the fix is a shorter or later window, NOT an entropy
# floor: max(x_ent_coef, 3e-4) is a measured failure (Pong 14.11, Gopher 582.8
# vs a coupled reference of 17.4-20.4) and that line is closed.
#
# PRE-REGISTERED BANDS. Two questions, one panel. Tail games ask "do the extra
# updates deliver what skipping the reset delivered?"; mid-mass games ask "does
# the restored reset hold them at combo level?".
#   DemonAttack  RUN=130 {34594, 28215} / combo n=5 {11684-24241}.
#                >=~28000 delivers; <=~24000 = combo-like, lever dead.
#   UpNDown      RUN=130 {83061, 66944} / combo 7876 / anchor max 64545.
#                >=~50000 delivers; <=~20000 = the gain needed the skipped
#                reset, not the updates. Biggest single term in the row.
#   Breakout     RUN=130 {374, 330} / combo {333, 188} / anchor 259-401.
#                >=~350 delivers. Weak band by construction -- Breakout's seed
#                tail is 5.04 HNS (~145 raw points); read it as support only.
#   Asterix      combo {7247, 9828} / RUN=130 {5310, 4206} / anchor 2589-12732.
#                >=~7000 = reset restored it (same threshold the surgery gate
#                used); <=~4500 = the reset was never Asterix's lever.
#   RoadRunner   combo 27626 / RUN=130 {23093, 18024} / anchor 17771-34237.
#                >=~26000 restored; ~20000 not.
# DECISION: tail games hold AND mid-mass returns to combo -> this config
# dominates both rows on IQM and mean at once -> 26-game suite (GAMES=... with
# the full list, ~78 h). Tail games fall back -> the tail gains belonged to the
# skipped reset itself, so drop the lever and spend the box-time on RUN=130
# seeds instead. Deliberately NOT in the panel: Jamesbond (jackpot, 4.31 HNS
# tail), BankHeist / Frostbite / Pong (coin games, ~40-50% false alarm under
# any config -- Frostbite's own RUN=130 seeds are {258, 247, 2866}).
#
# WHAT THIS ARM DISCRIMINATES -- two live hypotheses, evidence on both sides,
# and this is the cheap cell that separates them. Do not read the result as
# confirming either one before checking which pattern it matches.
#
#   BUDGET: what the tail games want is productive gradient steps. Support:
#   RUN=80 @RR4 resets EIGHT times (10k..80k), the most aggressive resetting in
#   the project, and its last reset leaves the same 80k grad steps in the final
#   stretch as combo's does -- same tail structure, 2x total updates -- yet DA
#   came out 47432 vs combo's {11684-24241} and Breakout 380.47/380.08 (twin
#   seeds, best in the readout family). Under weight-preservation RR4 should
#   have been the worst arm for those games; it was the best.
#
#   WEIGHTS: what the tail games want is not having the Q/actor heads (and half
#   the encoder/TM) wiped at 60k. Support: RUN=130 beats combo on exactly those
#   games at an IDENTICAL total gradient budget -- both run 200k grad steps,
#   since skipping a reset adds no updates. Its extra tail benefit has to come
#   from the un-reset stretch being 120k grad long instead of 80k.
#
# Both readings fit the RUN=130 table; only BUDGET predicts arm C hits its
# bands, since arm C changes update volume with the 60k reset left in place.
# The project record filed this as "routing-vs-budget unresolved" after the RR4
# probes -- this arm is the resolution cell, at +20% instead of RR4's +100%.
#
# COST: throughput is gradient-bound (RR1 25.1 sps, RR2 13.0, RR4 6.5), so arm
# C runs ~2.56 h/game (+20%) vs ~2.99 h (+40%) for arms A/B. The 4-game panel
# is ~10.3 h per arm per seed; add Breakout for ~12.8 h; 26-game suite ~66.7 h
# (vs ~55.6 h at plain RR2).
#
# Usage: [ARM=C|A|B] [GAMES=...] [GPU=0] [REPS=1] bash ablations/21-late-updates-gate.sh
# RUN is the row id, not the seed (train.py --no_seeding defaults True, so
# every repetition draws a fresh time-based seed): C=142, A=140, B=141.
cd "$(dirname "$0")/.."
GAMES=${GAMES:-"DemonAttack UpNDown Asterix RoadRunner"}
GPU=${GPU:-0}
ARM=${ARM:-C}
REPS=${REPS:-1}

case "$ARM" in
C) RUN=${RUN:-142}; LATE_AFTER=80000; RESET_BINDING="BBFAgent.no_resets_after=100000" ;;
A) RUN=${RUN:-140}; LATE_AFTER=60000; RESET_BINDING="BBFAgent.no_resets_after=100000" ;;
B) RUN=${RUN:-141}; LATE_AFTER=60000; RESET_BINDING="BBFAgent.no_resets_after=110000" ;;
*) echo "ARM must be C, A or B" >&2; exit 1 ;;
esac

for ((rep = 1; rep <= REPS; rep++)); do
	for game_name in $GAMES; do
		CUDA_VISIBLE_DEVICES=$GPU python -m bbf.train \
			--agent=BBF \
			--gin_files=bbf/configs/BBF-100K.gin \
			--gin_bindings="DataEfficientAtariRunner.game_name=\"$game_name\"" \
			--gin_bindings="BBFAgent.reward_readout=True" \
			--gin_bindings="BBFAgent.imag_value_weight=0.05" \
			--gin_bindings="BBFAgent.imag_actor_weight=0.1" \
			--gin_bindings="BBFAgent.imag_entropy_weight=None" \
			--gin_bindings="BBFAgent.imag_discount=None" \
			--gin_bindings="BBFAgent.late_update_after=$LATE_AFTER" \
			--gin_bindings="BBFAgent.late_update_multiplier=2" \
			--gin_bindings="$RESET_BINDING" \
			--run_number=$RUN
	done
done
