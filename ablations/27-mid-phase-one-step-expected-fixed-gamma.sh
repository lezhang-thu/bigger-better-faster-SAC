set -ex
# Fixed-gamma counterpart of 26-mid-phase-one-step-expected.sh. It retains
# strict one-step expected-C51 replay learning and changes only the discount:
# gamma is fixed at 0.997 instead of restarting at 0.97 and annealing after
# each reset.
#
# BBFAgent.min_gamma=None selects the constant scheduler
# gamma_scheduler(step) = self.gamma. JaxDQNAgent.gamma=0.997 is repeated
# explicitly here so the fixed value does not depend silently on the base gin
# file. imag_discount=None makes the five-step imagined lambda-return use that
# same constant gamma, keeping replay and imagination aligned.
#
# Usage:
#   ARM=E bash ablations/27-mid-phase-one-step-expected-fixed-gamma.sh
#   ARM=F GAMES="DemonAttack Asterix" REPS=2 GPU=1 \
#     bash ablations/27-mid-phase-one-step-expected-fixed-gamma.sh
#
# RUN is a row id, not a seed. E/F use ids distinct from both the historical
# multi-step controls (144/145) and dynamic-gamma one-step rows (180/181).
cd "$(dirname "$0")/.."
GAMES=${GAMES:-"DemonAttack UpNDown Asterix RoadRunner"}
GPU=${GPU:-0}
ARM=${ARM:-E}
REPS=${REPS:-1}

case "$ARM" in
E) RUN=${RUN:-182}; LATE_AFTER=20000; LATE_UNTIL=40000 ;;
F) RUN=${RUN:-183}; LATE_AFTER=40000; LATE_UNTIL=60000 ;;
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
			--gin_bindings="JaxDQNAgent.gamma=0.997" \
			--gin_bindings="BBFAgent.min_gamma=None" \
			--gin_bindings="BBFAgent.imag_discount=None" \
			--gin_bindings="BBFAgent.no_resets_after=100000" \
			--gin_bindings="BBFAgent.late_update_after=$LATE_AFTER" \
			--gin_bindings="BBFAgent.late_update_until=$LATE_UNTIL" \
			--gin_bindings="BBFAgent.late_update_multiplier=2" \
			--gin_bindings="BBFAgent.update_horizon=1" \
			--gin_bindings="BBFAgent.max_update_horizon=1" \
			--gin_bindings="BBFAgent.expected_one_step_backup=True" \
			--run_number=$RUN
	done
done
