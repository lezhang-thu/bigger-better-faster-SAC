set -ex
# The combo (07-suite-combo.sh, RUN=50) with gradient updates doubled over
# the cycle AFTER the first actual reset and UNTIL the second -- and nothing
# else touched. Resets still fire three times (then the 80k attempt is
# skipped by no_resets_after=100k), entropy still anneals to zero at 80k,
# replay_ratio still 64.
#
# WHY NOT [20k, 40k). Resets do not land on those round numbers. With
# reset_every=20_000 and reset_offset=1, next_reset starts at 20001, the
# check is training_steps > next_reset, and training_steps increments after
# the reset, so the three resets are at 20002 / 40003 / 60004. A hard
# [20000, 40000) env-step window therefore turns x2 on two steps before the
# first reset and off three steps before the second -- it is not "the
# first-to-second reset cycle". This row keys the multiplier on
# cumulative_resets (after=1, until=2): updates run at 1x on the env step
# that performs reset 1 (reset is after the update), x2 starts on the next
# env step, and x2 stays on through the env step that performs reset 2.
#
# COST. One reset cycle doubled: ~20001 env steps x2 against the combo's
# 200k grad, so ~240k grad and ~2.56 h/game vs ~2.14. Same +20% as the
# other single-window late-update arms.
#
# CONFOUND, unavoidable for a post-reset window: doubling compresses the
# post-reset update-horizon (10->3) and gamma (0.97->0.997) anneals, which
# key on cycle_grad_steps=10_000 grad steps, from 5_000 env steps into
# 2_500 (done ~22503 vs combo 25002).
#
# PANEL, shared with the late-update C/D rows. References are the COMBO:
#   DemonAttack  combo n=5 {11683.8, 13561.75, 18282.7, 18408.55, 24240.55},
#                mean 17235. >=~25000 beats every combo seed; <=~13000 = harm.
#   UpNDown      combo 7876; anchor 4375-64545; RUN=130 {83061, 66944}.
#                >=~20000 = a real move; <=~7000 = nothing.
#   Asterix      combo {7246.5, 9828}; anchor 2588-12731.
#                >=~9800 beats both combo seeds; <=~5500 = harm.
#   RoadRunner   combo 27626; anchor 17771-34237. >=~28000 gain; <=~22000 harm.
#
# Usage: [GAMES=...] [GPU=0] [REPS=1] [RUN=144] bash ablations/24-mid-phase-updates.sh
# RUN is the row id, not the seed (train.py --no_seeding defaults True).
cd "$(dirname "$0")/.."
GAMES=${GAMES:-"DemonAttack UpNDown Asterix RoadRunner"}
GPU=${GPU:-0}
REPS=${REPS:-1}
RUN=${RUN:-144}

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
			--gin_bindings="BBFAgent.late_update_after_resets=1" \
			--gin_bindings="BBFAgent.late_update_until_resets=2" \
			--gin_bindings="BBFAgent.late_update_multiplier=2" \
			--run_number=$RUN
	done
done
