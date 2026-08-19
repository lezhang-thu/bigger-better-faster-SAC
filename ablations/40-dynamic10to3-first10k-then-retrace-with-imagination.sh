set -ex
# Dynamic-warmup counterpart of row 38. During cycle-local gradient updates
# [0, 10_000), the replay critic uses an uncorrected n-step C51 target whose
# horizon and discount follow the original BBF schedules:
#
#   n:      10 -> 3
#   gamma:  0.97 -> 0.997
#
# The endpoint uses the same exact online-policy mixture as row 38's fixed
# warmup. Starting at update 10_000, the critic switches to the existing H=10,
# lambda=1 Distributional Retrace target at fixed gamma=0.997. Initialization
# starts a cycle, and every successful shrink-and-perturb reset restarts both
# warmup schedules and the 10k target phase.
#
# As in row 38, the replay buffer supplies raw one-step rows throughout. During
# warmup, PER uses |E[projected n-step target] - Q(s,a)| + 1e-6 as its raw
# delta score; the existing alpha=.5 host transform stores its square root. At
# the 10k switch, every populated priority is made uniform and the pending
# prefetched batch is discarded before Retrace sampling begins. Active Retrace
# then uses its bounded total-variation priorities, so priorities from the two
# TD targets never mix. Beta=1 critic correction, beta=0.5 auxiliary/imagination
# weights, and behavior probabilities remain active throughout.
#
# With imag_discount=None, imagination follows the scheduled critic gamma
# during warmup and the fixed final gamma under Retrace. Mid-phase update
# doubling remains disabled.
#
# This keeps row 38's complete learned-model objectives:
#   * MODEL_HORIZON=imag_horizon=5 (inherited from BBF-100K.gin),
#   * imagined actor weight 0.1 and imagined value weight 0.05,
#   * reward/continue head weights 1.0 (inherited from BBF-100K.gin), and
#   * five-step SPR training.
# The cycle-local imag_warmup=2_000 is retained: imagined losses are off through
# update 2k, ramp linearly to full strength from 2k to 4k, and remain fully
# active thereafter. reward_readout=True keeps the reward/continue losses from
# backpropagating into the encoder/transition model.
#
# Usage:
#   ARM=E bash ablations/40-dynamic10to3-first10k-then-retrace-with-imagination.sh
#   ARM=F GAMES="Kangaroo Asterix" GPU=1 \
#     bash ablations/40-dynamic10to3-first10k-then-retrace-with-imagination.sh
cd "$(dirname "$0")/.."
GAMES=${GAMES:-"Kangaroo Asterix"}
GPU=${GPU:-0}
ARM=${ARM:-E}

case "$ARM" in
E) RUN=${RUN:-209} ;;
F) RUN=${RUN:-210} ;;
*)
	echo "ARM must be E or F (paired run-ID selectors)" >&2
	exit 1
	;;
esac

BBF_PYTHON=${BBF_PYTHON:-"/home/amd-7763/.venvs/bbf-rocm714-jax011/bin/python"}
for game_name in $GAMES; do
	env \
		-u LD_LIBRARY_PATH \
		-u ROCM_PATH \
		-u HIP_PATH \
		-u LLVM_PATH \
		-u HSA_NO_SCRATCH_RECLAIM \
		ROCR_VISIBLE_DEVICES="$GPU" \
		ROCPROFILER_QUEUE_INTERPOSITION=0 \
		DEBUG_HIP_DYNAMIC_QUEUES=0 \
		"$BBF_PYTHON" -m bbf.train \
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
		--gin_bindings="BBFAgent.expected_one_step_backup=False" \
		--gin_bindings="BBFAgent.retrace=True" \
		--gin_bindings="BBFAgent.retrace_horizon=10" \
		--gin_bindings="BBFAgent.retrace_lambda=1.0" \
		--gin_bindings="BBFAgent.retrace_warmup_n_step_updates=10000" \
		--gin_bindings="BBFAgent.retrace_warmup_n_step_horizon=10" \
		--gin_bindings="BBFAgent.retrace_warmup_n_step_final_horizon=3" \
		--gin_bindings="BBFAgent.retrace_warmup_min_gamma=0.97" \
		--gin_bindings="BBFAgent.retrace_warmup_delta_based_priority=True" \
		--gin_bindings="BBFAgent.retrace_reset_priorities_on_target_switch=True" \
		--gin_bindings="BBFAgent.delta_priority_epsilon=1e-6" \
		--gin_bindings="PrioritizedJaxSubsequenceParallelEnvReplayBuffer.prioritized_sampling=True" \
		--run_number=$RUN
done
