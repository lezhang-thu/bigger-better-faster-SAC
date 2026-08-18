set -ex
# Imagination-enabled counterpart of row 39 and target-schedule variant of row
# 28. During cycle-local gradient updates [0, 10_000), the replay critic uses
# an uncorrected fixed ten-step C51 target with terminal truncation and the
# exact online-policy mixture at S10. Starting at update 10_000 it switches to
# the existing H=10, lambda=1 Distributional Retrace target. Initialization
# starts a cycle, and every successful shrink-and-perturb reset restarts the
# 10k fixed-target phase.
#
# Unlike row 39, this keeps row 28's complete learned-model objectives:
#   * MODEL_HORIZON=imag_horizon=5 (inherited from BBF-100K.gin),
#   * imagined actor weight 0.1 and imagined value weight 0.05,
#   * reward/continue head weights 1.0 (inherited from BBF-100K.gin), and
#   * five-step SPR training.
# The existing cycle-local imag_warmup=2_000 is also retained: imagined losses
# are off through update 2k, ramp linearly to full strength from 2k to 4k, and
# remain fully active thereafter.
# reward_readout=True also matches row 28, so reward/continue losses train the
# two readout heads without backpropagating into the encoder/transition model.
#
# This remains a target-only switch: raw one-step replay, TV priorities,
# beta=1 critic correction, beta=0.5 auxiliary/imagination weights, behavior
# probabilities, fixed gamma=0.997, and prefetching stay unchanged across the
# boundary. Mid-phase update doubling remains disabled.
#
# Usage:
#   ARM=E bash ablations/38-nstep10-first10k-then-retrace-with-imagination.sh
#   ARM=F GAMES="Kangaroo Asterix" REPS=2 GPU=1 \
#     bash ablations/38-nstep10-first10k-then-retrace-with-imagination.sh
cd "$(dirname "$0")/.."
GAMES=${GAMES:-"Kangaroo Asterix"}
GPU=${GPU:-0}
ARM=${ARM:-E}
REPS=${REPS:-1}

case "$ARM" in
E) RUN=${RUN:-205} ;;
F) RUN=${RUN:-206} ;;
*) echo "ARM must be E or F (paired run-ID selectors)" >&2; exit 1 ;;
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
			--gin_bindings="PrioritizedJaxSubsequenceParallelEnvReplayBuffer.prioritized_sampling=True" \
			--run_number=$RUN
	done
done
