set -ex
# Optimistic C51 target max(Y_1, Y_10), with both candidates bootstrapped from
# the same exact current-policy mixture of target-critic action distributions.
# Their projected support means choose the complete candidate distribution;
# the selected C51 distribution replaces the ordinary one-step critic target.
#
# PER keeps the codebase's existing alpha/beta/sampling/update plumbing. Its raw
# score is |max(Y_1, Y_10) - Q(s_t,a_t)| + 1e-6; the existing host square root
# (with its shared 1e-10 numerical floor) implements alpha=.5. Replay returns
# raw one-step rows; the fixed ten-step reward prefix and S_10 value bootstrap
# are reconstructed from the same sampled anchor. Gamma is fixed at 0.997.
# Late-update multiplication is explicitly disabled, so the gradient-update
# rate stays constant throughout training; there are no 20k-40k or 40k-60k
# arms in this experiment.
#
# Usage:
#   bash ablations/30-mid-phase-maximization-target-fixed10-fixed-gamma.sh
#   GAMES="Kangaroo Asterix" REPS=2 GPU=1 RUN=192 \
#     bash ablations/30-mid-phase-maximization-target-fixed10-fixed-gamma.sh
cd "$(dirname "$0")/.."
GAMES=${GAMES:-"Kangaroo Asterix"}
GPU=${GPU:-0}
REPS=${REPS:-1}
RUN=${RUN:-192}

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
			--gin_bindings="BBFAgent.late_update_after=-1" \
			--gin_bindings="BBFAgent.late_update_until=-1" \
			--gin_bindings="BBFAgent.late_update_multiplier=1" \
			--gin_bindings="BBFAgent.update_horizon=1" \
			--gin_bindings="BBFAgent.max_update_horizon=1" \
			--gin_bindings="BBFAgent.expected_one_step_backup=True" \
			--gin_bindings="BBFAgent.retrace=False" \
			--gin_bindings="BBFAgent.td_lower_bound_weight=0.0" \
			--gin_bindings="BBFAgent.td_maximization_target=True" \
			--gin_bindings="BBFAgent.td_maximization_horizon=10" \
			--gin_bindings="BBFAgent.td_maximization_priority_epsilon=1e-6" \
			--gin_bindings="PrioritizedJaxSubsequenceParallelEnvReplayBuffer.prioritized_sampling=True" \
			--run_number=$RUN
	done
done
