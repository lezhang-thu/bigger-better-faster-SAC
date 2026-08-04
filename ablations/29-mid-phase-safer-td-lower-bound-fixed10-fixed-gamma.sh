set -ex
# One-step expected C51 plus a sampled-action, uncorrected ten-step lower-bound
# signal. The one-step C51 cross-entropy remains the primary critic objective.
# The ten-step projected target is reduced to its support mean and can only
# raise the root value through a weight-0.1 one-sided Huber penalty.
#
# PER keeps the codebase's existing alpha/beta/update plumbing. Its raw score is
# u_t = |Y_1 - Q| + 0.5 * relu(Y_10 - Q) + 1e-6, after which the existing host
# square root implements alpha=.5. Replay returns raw one-step rows; the fixed
# ten-step reward prefix and sampled-action S_10 bootstrap are reconstructed
# from the same sampled anchor. Gamma is fixed at 0.997.
#
# Usage:
#   ARM=E bash ablations/29-mid-phase-safer-td-lower-bound-fixed10-fixed-gamma.sh
#   ARM=F GAMES="Kangaroo Asterix" REPS=2 GPU=1 \
#     bash ablations/29-mid-phase-safer-td-lower-bound-fixed10-fixed-gamma.sh
cd "$(dirname "$0")/.."
GAMES=${GAMES:-"Kangaroo Asterix"}
GPU=${GPU:-0}
ARM=${ARM:-E}
REPS=${REPS:-1}

case "$ARM" in
E) RUN=${RUN:-190}; LATE_AFTER=20000; LATE_UNTIL=40000 ;;
F) RUN=${RUN:-191}; LATE_AFTER=40000; LATE_UNTIL=60000 ;;
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
			--gin_bindings="BBFAgent.retrace=False" \
			--gin_bindings="BBFAgent.td_lower_bound_weight=0.3" \
			--gin_bindings="BBFAgent.td_lower_bound_horizon=10" \
			--gin_bindings="BBFAgent.td_lower_bound_priority_eta=0.5" \
			--gin_bindings="BBFAgent.td_lower_bound_priority_epsilon=1e-6" \
			--gin_bindings="PrioritizedJaxSubsequenceParallelEnvReplayBuffer.prioritized_sampling=True" \
			--run_number=$RUN
	done
done
