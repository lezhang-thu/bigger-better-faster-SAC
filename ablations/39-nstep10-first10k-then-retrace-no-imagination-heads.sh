set -ex
# Target-schedule ablation paired with row 37. Replay always supplies ten
# consecutive raw one-step rows under fixed gamma=0.997. During cycle-local
# gradient updates [0, 10_000), the critic uses the uncorrected fixed ten-step
# C51 target
#
#   Pi_Z[R1 + gamma R2 + ... + gamma^9 R10
#        + gamma^10 Z(S10, A), A ~ pi_online(.|S10)],
#
# with terminal truncation. Starting at update 10_000 it uses the existing
# H=10, lambda=1 Distributional Retrace target. Initialization starts a cycle,
# and every successful shrink-and-perturb reset restarts the 10k warmup.
#
# This is deliberately a target-only switch: TV priorities, beta=1 critic
# importance correction, beta=0.5 auxiliary weights, behavior-probability
# logging, raw replay horizon, and prefetching stay unchanged across it.
# Imagined actor/value training and the unused reward/continue heads remain off
# exactly as in row 37. Mid-phase update doubling remains disabled.
#
# Usage:
#   ARM=E bash ablations/39-nstep10-first10k-then-retrace-no-imagination-heads.sh
#   ARM=F GAMES="Kangaroo Asterix" REPS=2 GPU=1 \
#     bash ablations/39-nstep10-first10k-then-retrace-no-imagination-heads.sh
cd "$(dirname "$0")/.."
GAMES=${GAMES:-"Kangaroo Asterix"}
GPU=${GPU:-0}
ARM=${ARM:-E}
REPS=${REPS:-1}

case "$ARM" in
E) RUN=${RUN:-207} ;;
F) RUN=${RUN:-208} ;;
*) echo "ARM must be E or F (paired run-ID selectors)" >&2; exit 1 ;;
esac

for ((rep = 1; rep <= REPS; rep++)); do
	for game_name in $GAMES; do
		CUDA_VISIBLE_DEVICES=$GPU python -m bbf.train \
			--agent=BBF \
			--gin_files=bbf/configs/BBF-100K.gin \
			--gin_bindings="DataEfficientAtariRunner.game_name=\"$game_name\"" \
			--gin_bindings="BBFAgent.reward_readout=True" \
			--gin_bindings="BBFAgent.reward_weight=0.0" \
			--gin_bindings="BBFAgent.continue_weight=0.0" \
			--gin_bindings="BBFAgent.reward_grad_surgery=False" \
			--gin_bindings="BBFAgent.imag_horizon=0" \
			--gin_bindings="BBFAgent.imag_value_weight=0.0" \
			--gin_bindings="BBFAgent.imag_actor_weight=0.0" \
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
