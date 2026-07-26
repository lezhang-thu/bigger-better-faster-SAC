set -ex
# The combo (07-suite-combo.sh, RUN=50) with the gradient updates doubled over
# ONE 20k env-step window and nothing else touched -- resets still 20k/40k/60k,
# entropy still annealing to zero at 80k, replay_ratio still 64.
#   ARM=E (RUN=144)  x2 over [20k, 40k)
#   ARM=F (RUN=145)  x2 over [40k, 60k)
#
# These complete a four-window sweep at IDENTICAL cost. Every arm doubles
# exactly 20k env steps, so each is 240k gradient steps against the combo's
# 200k, and each runs ~2.56 h/game against ~2.14 -- the +20% is the same
# whichever window you pick, which is what makes the four directly comparable:
#
#   arm   window        grad per reset cycle          post-reset anneal done
#   combo  --           40k / 40k / 40k / 80k         25002 / 45003 / 65004
#   E     [20k, 40k)    40k / 80k / 40k / 80k         22502 / 45003 / 65004
#   F     [40k, 60k)    40k / 40k / 80k / 80k         25002 / 42503 / 65004
#   D     [60k, 80k)    40k / 40k / 40k / 120k        25002 / 45003 / 62504
#   C     [80k,100k)    40k / 40k / 40k / 120k        25002 / 45003 / 65004
# (D and C differ in where inside the final stretch the doubling sits: D
# front-loads it onto the post-60k-reset recovery, C onto the run-out.)
#
# WHAT EACH WINDOW IS. Every one of E/F/D covers a post-reset recovery phase --
# the 20k, 40k and 60k resets respectively -- while C covers the only stretch
# with no reset in it. Recovery is worth quantifying: measured over the 46
# RUN=130 runs at their 40k reset, the training return drops to a median 49% of
# its pre-reset level (38/46 runs below 80%) and takes a median 5_500 env steps
# to climb back, with the worst cases at 11-14k. So each 20k window here is
# roughly "the recovery, plus 15k of ordinary training".
#
# TWO PREDICTIONS THAT DISAGREE, which is the point of running E and F rather
# than assuming.
#  (a) EARLY COMPOUNDS. Resets do not wipe the representation: encoder and
#      transition_model are shrink-perturbed 0.5/0.5, and reward_head,
#      continue_head and _log_alpha are copied over intact. So representation
#      quality bought early partially survives every later reset and has 60-80k
#      env steps left to pay off, whereas anything C buys has 20k. On this
#      reading E > F > D > C.
#  (b) DATA-STARVATION. The buffer holds only 20-40k transitions during E's
#      window against 60-80k during D's, so E spends 80k gradient steps in the
#      cycle where there is least to learn from -- and this repo's own note is
#      that plasticity loss scales with gradient steps, which is why higher
#      replay ratios need resets MORE often, not less. On this reading the
#      ordering inverts: D > F > E, with E the most likely to overfit its
#      buffer inside a cycle whose reset spacing was not widened to match.
# The sweep separates them cleanly, and the answer also tells us whether the
# RUN=130 tail advantage was ever about update volume at all.
#
# CONFOUND, one per arm and unavoidable: doubling across a reset compresses the
# post-reset update-horizon (10->3) and gamma (0.97->0.997) anneals, which key
# on cycle_grad_steps=10_000 grad steps, from 5_000 env steps into 2_500 (E:
# 22502 vs 25002; F: 42503 vs 45003; D: 62504 vs 65004). C is the only arm
# whose schedules are bit-identical to the control, and it pays for that by
# sitting entirely inside the window where x_ent_coef is exactly 0.
#
# PANEL, shared with arms C/D so all four rows are read off the same games.
# References are the COMBO (these are combo variants, not RUN=130 variants):
#   DemonAttack  combo n=5 {11683.8, 13561.75, 18282.7, 18408.55, 24240.55},
#                mean 17235. Arm C returned 10988.75 -- below all five.
#                >=~25000 beats every combo seed; <=~13000 = harm.
#   UpNDown      combo 7876; anchor 4375-64545; RUN=130 {83061, 66944}.
#                >=~20000 = a real move; <=~7000 = nothing.
#   Asterix      combo {7246.5, 9828}; anchor 2588-12731.
#                >=~9800 beats both combo seeds; <=~5500 = harm.
#   RoadRunner   combo 27626; anchor 17771-34237. >=~28000 gain; <=~22000 harm.
# DECISION: read E, F, D and C against each other, not just against combo --
# equal cost means the winner is simply the best window, and a flat result
# across all four says update volume is not the lever anywhere. Any arm that
# beats combo on DemonAttack AND holds Asterix at combo level earns a 26-game
# suite (~67 h at +20%).
#
# Usage: ARM=E|F [GAMES=...] [GPU=0] [REPS=1] bash ablations/24-mid-phase-updates.sh
# RUN is the row id, not the seed (train.py --no_seeding defaults True).
cd "$(dirname "$0")/.."
GAMES=${GAMES:-"DemonAttack UpNDown Asterix RoadRunner"}
GPU=${GPU:-0}
ARM=${ARM:-E}
REPS=${REPS:-1}

case "$ARM" in
E) RUN=${RUN:-144}; LATE_AFTER=20000; LATE_UNTIL=40000 ;;
F) RUN=${RUN:-145}; LATE_AFTER=40000; LATE_UNTIL=60000 ;;
*) echo "ARM must be E ([20k,40k)) or F ([40k,60k))" >&2; exit 1 ;;
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
			--gin_bindings="BBFAgent.no_resets_after=100000" \
			--gin_bindings="BBFAgent.late_update_after=$LATE_AFTER" \
			--gin_bindings="BBFAgent.late_update_until=$LATE_UNTIL" \
			--gin_bindings="BBFAgent.late_update_multiplier=2" \
			--run_number=$RUN
	done
done
