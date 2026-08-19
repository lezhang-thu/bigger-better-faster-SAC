set -ex
# Fixed-warmup counterpart of row 40. During cycle-local gradient updates
# [0, 10_000), the replay critic uses an uncorrected 10-step C51 target at a
# fixed gamma of 0.997:
#
#   n:      10
#   gamma:  0.997
#
# The endpoint uses the same exact online-policy mixture as rows 38 and 40.
# Starting at update 10_000, the critic switches to the existing H=10,
# lambda=1 Distributional Retrace target, also at gamma=0.997. Initialization
# starts a cycle, and every successful shrink-and-perturb reset restarts the
# 10k fixed-target warmup phase.
#
# As in row 40, the replay buffer supplies raw one-step rows throughout. During
# warmup, PER uses |E[projected 10-step target] - Q(s,a)| + 1e-6 as its raw
# delta score; the existing alpha=.5 host transform stores its square root. At
# the 10k switch, every populated priority is made uniform and the pending
# prefetched batch is discarded before Retrace sampling begins. Active Retrace
# then uses its bounded total-variation priorities, so priorities from the two
# TD targets never mix. Beta=1 critic correction, beta=0.5 auxiliary/imagination
# weights, and behavior probabilities remain active throughout.
#
# With imag_discount=None, imagination follows the fixed critic gamma=0.997 in
# both phases. Mid-phase update doubling remains disabled.
#
# This keeps row 40's complete learned-model objectives:
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
#   ARM=E bash ablations/41-nstep10-fixedgamma-first10k-then-retrace-with-imagination.sh
#   ARM=F GAMES="Kangaroo Asterix" GPU=1 \
#     bash ablations/41-nstep10-fixedgamma-first10k-then-retrace-with-imagination.sh
cd "$(dirname "$0")/.."
GAMES=${GAMES:-"Kangaroo Asterix"}
GPU=${GPU:-0}
ARM=${ARM:-E}

case "$ARM" in
E) RUN=${RUN:-211} ;;
F) RUN=${RUN:-212} ;;
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
		--gin_bindings="BBFAgent.retrace_warmup_n_step_final_horizon=None" \
		--gin_bindings="BBFAgent.retrace_warmup_min_gamma=None" \
		--gin_bindings="BBFAgent.retrace_warmup_delta_based_priority=True" \
		--gin_bindings="BBFAgent.retrace_reset_priorities_on_target_switch=True" \
		--gin_bindings="BBFAgent.delta_priority_epsilon=1e-6" \
		--gin_bindings="PrioritizedJaxSubsequenceParallelEnvReplayBuffer.prioritized_sampling=True" \
		--run_number=$RUN
done
