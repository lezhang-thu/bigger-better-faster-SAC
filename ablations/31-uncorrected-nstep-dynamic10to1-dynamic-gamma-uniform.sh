#!/usr/bin/env bash
set -ex

# Recover the original uncorrected replay n-step C51 target, but anneal the
# replay horizon from 10 to 1 instead of from 10 to 3. Both schedules restart
# after every successful agent reset and run for the first 10,000 gradient
# steps of the new reset cycle:
#
#   n:      10 -> 1
#   gamma:  0.97 -> 0.997
#
# update_horizon is the FINAL horizon and max_update_horizon is the initial
# horizon. At each scheduled endpoint S_{t+n}, expected_one_step_backup=False
# samples a' from the current policy and bootstraps from the target critic's
# complete C51 distribution for that action. In particular, once n reaches 1,
# the target is r_{t+1} + gamma Q_target(S_{t+1}, a') with sampled a', not the
# policy-weighted mixture over actions.
#
# Replay sampling is uniform. The prioritized buffer class remains in use for
# compatibility, but its sum-tree sampling, importance weights, and priority
# updates are disabled by prioritized_sampling=False. Late update multiplication
# is explicitly disabled so cycle_steps=10,000 always means 10,000 gradient
# steps rather than a compressed environment-step window.
#
# Usage:
#   bash ablations/31-uncorrected-nstep-dynamic10to1-dynamic-gamma-uniform.sh
#   GAMES="Kangaroo Asterix" REPS=2 GPU=1 RUN=193 \
#     bash ablations/31-uncorrected-nstep-dynamic10to1-dynamic-gamma-uniform.sh
cd "$(dirname "$0")/.."
GAMES=${GAMES:-"Kangaroo Asterix"}
GPU=${GPU:-0}
REPS=${REPS:-1}
RUN=${RUN:-193}

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
			--gin_bindings="BBFAgent.min_gamma=0.97" \
			--gin_bindings="BBFAgent.cycle_steps=10000" \
			--gin_bindings="BBFAgent.imag_discount=None" \
			--gin_bindings="BBFAgent.reset_every=20000" \
			--gin_bindings="BBFAgent.no_resets_after=100000" \
			--gin_bindings="BBFAgent.late_update_after=-1" \
			--gin_bindings="BBFAgent.late_update_until=-1" \
			--gin_bindings="BBFAgent.late_update_multiplier=1" \
			--gin_bindings="BBFAgent.update_horizon=1" \
			--gin_bindings="BBFAgent.max_update_horizon=10" \
			--gin_bindings="BBFAgent.expected_one_step_backup=False" \
			--gin_bindings="BBFAgent.retrace=False" \
			--gin_bindings="BBFAgent.td_lower_bound_weight=0.0" \
			--gin_bindings="BBFAgent.td_maximization_target=False" \
			--gin_bindings="PrioritizedJaxSubsequenceParallelEnvReplayBuffer.prioritized_sampling=False" \
			--run_number=$RUN
	done
done
