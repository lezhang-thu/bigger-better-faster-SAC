#!/usr/bin/env bash
set -ex

# The only supported TD schedule is cycle-local. For
# p=clip(cycle_gradient_updates/20_000, 0, 1):
#   H=round(10 * 0.1^p)
#   gamma=1 - 0.03 * 0.1^p
# Thus the schedule starts at H=10/gamma=.97 and stays at H=1/gamma=.997
# after 20,000 gradient updates. There is no replay-priority reset at 20k.
# Every successful shrink-and-perturb reset uniformizes priorities and restarts
# the schedule. The C51 endpoint remains an exact online-policy mixture.
#
# Usage:
#   ARM=E bash ablations/43-dynamic10to1-dynamicgamma20k-with-imagination.sh
#   ARM=F GAMES="Kangaroo Asterix" GPU=1 \
#     bash ablations/43-dynamic10to1-dynamicgamma20k-with-imagination.sh
cd "$(dirname "$0")/.."
GAMES=${GAMES:-"Kangaroo Asterix"}
GPU=${GPU:-0}
ARM=${ARM:-E}

case "$ARM" in
E) RUN=${RUN:-215} ;;
F) RUN=${RUN:-216} ;;
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
