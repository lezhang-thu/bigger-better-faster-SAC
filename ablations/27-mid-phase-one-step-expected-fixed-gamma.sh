set -ex
# ERE + strict one-step expected C51 + fixed gamma. This is the bug-fixed
# counterpart of the same-named script on bbf-starting-point-claude.
#
# Relative to the combo configuration:
#   1. ERE replaces TD-prioritized replay. Within each two-minibatch update
#      phase, k=0 samples the full retained replay and k=1 samples uniformly
#      from the most recent c_k=max(5000, 200000*0.995^(k*1000/2)) transitions.
#   2. update_horizon=max_update_horizon=1 gives the real transition
#      (s_t, a_t, r_{t+1}, done_{t+1}, s_{t+1}) throughout training.
#   3. The C51 bootstrap is the exact current-policy mixture of the target
#      critic's per-action distributions instead of one sampled action.
#   4. gamma is fixed at 0.997 for both replay TD and imagined lambda-returns.
#   5. The normal replay ratio is retained throughout training.
#
# SPR, reward/continue supervision, imagination depth (5), reset timing, and
# all remaining combo settings are unchanged. RUN=184 is distinct from the
# dynamic-gamma one-step rows (180/181) and the fixed-gamma rows (182/183) on
# bbf-starting-point-claude.
#
# Usage:
#   bash ablations/27-mid-phase-one-step-expected-fixed-gamma.sh
#   GAMES="DemonAttack Asterix" REPS=2 GPU=1 \
#     bash ablations/27-mid-phase-one-step-expected-fixed-gamma.sh
cd "$(dirname "$0")/.."
GAMES=${GAMES:-"DemonAttack UpNDown Asterix RoadRunner"}
GPU=${GPU:-0}
REPS=${REPS:-1}
RUN=${RUN:-184}

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
			--gin_bindings="PrioritizedJaxSubsequenceParallelEnvReplayBuffer.prioritized_sampling=False" \
			--gin_bindings="PrioritizedJaxSubsequenceParallelEnvReplayBuffer.ere_sampling=True" \
			--gin_bindings="PrioritizedJaxSubsequenceParallelEnvReplayBuffer.ere_eta=0.995" \
			--gin_bindings="PrioritizedJaxSubsequenceParallelEnvReplayBuffer.ere_min_window=5000" \
			--gin_bindings="BBFAgent.update_horizon=1" \
			--gin_bindings="BBFAgent.max_update_horizon=1" \
			--gin_bindings="BBFAgent.expected_one_step_backup=True" \
			--run_number=$RUN
	done
done
