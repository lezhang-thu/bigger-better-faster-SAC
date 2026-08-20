#!/usr/bin/env bash
set -ex

# The only supported TD schedule is cycle-local:
#   * updates [0, 10_000): fixed H=10 C51 target;
#   * update 10_000: reset replay priorities before sampling the first H=1 batch;
#   * updates [10_000, infinity): fixed H=1 C51 target.
# Both phases use gamma=0.997 and exact online-policy mixtures at the target
# endpoint. Every successful shrink-and-perturb reset starts a new cycle.
#
# Usage:
#   ARM=E bash ablations/42-nstep10-first10k-then-nstep1-with-imagination.sh
#   ARM=F GAMES="Kangaroo Asterix" GPU=1 \
#     bash ablations/42-nstep10-first10k-then-nstep1-with-imagination.sh
cd "$(dirname "$0")/.."
GAMES=${GAMES:-"Kangaroo Asterix"}
GPU=${GPU:-0}
ARM=${ARM:-E}

case "$ARM" in
E) RUN=${RUN:-213} ;;
F) RUN=${RUN:-214} ;;
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
		--run_number="$RUN"
done
