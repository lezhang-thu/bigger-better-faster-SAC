set -ex
# Distributional-Retrace counterpart of the fixed-gamma one-step control in
# row 27. The root C51 loss is still applied only to (s_0, a_0), but its signed
# target uses up to ten raw replay rewards with exact clipped pi/mu correction.
#
# The replay buffer must return raw one-step rows; retrace_horizon owns the
# separate fixed ten-step correction horizon. Gamma is fixed at 0.997 rather
# than restarting at 0.97 after each reset. MODEL_HORIZON remains five for SPR,
# reward/continue learning, and imagination.
#
# Usage:
#   ARM=E bash ablations/28-mid-phase-retrace-fixed10-fixed-gamma.sh
#   ARM=F GAMES="Kangaroo Asterix" REPS=2 GPU=1 \
#     bash ablations/28-mid-phase-retrace-fixed10-fixed-gamma.sh
cd "$(dirname "$0")/.."
GAMES=${GAMES:-"Kangaroo Asterix"}
GPU=${GPU:-0}
ARM=${ARM:-E}
REPS=${REPS:-1}

case "$ARM" in
E) RUN=${RUN:-188}; LATE_AFTER=20000; LATE_UNTIL=40000 ;;
F) RUN=${RUN:-189}; LATE_AFTER=40000; LATE_UNTIL=60000 ;;
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
			--gin_bindings="BBFAgent.expected_one_step_backup=False" \
			--gin_bindings="BBFAgent.retrace=True" \
			--gin_bindings="BBFAgent.retrace_horizon=10" \
			--gin_bindings="BBFAgent.retrace_lambda=1.0" \
			--gin_bindings="PrioritizedJaxSubsequenceParallelEnvReplayBuffer.prioritized_sampling=True" \
			--run_number=$RUN
	done
done
