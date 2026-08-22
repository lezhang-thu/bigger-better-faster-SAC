#!/usr/bin/env bash
set -ex

# The only supported TD schedule is cycle-local. For
# p=clip(cycle_gradient_updates/40_000, 0, 1):
#   H=round(10 * 0.1^p)
#   gamma=1 - 0.03 * 0.1^p
# Thus the schedule starts at H=10/gamma=.97 and stays at H=1/gamma=.997
# after 40,000 gradient updates. There is no replay-priority reset at 40k.
# Every successful shrink-and-perturb reset retains priorities and restarts the
# schedule, discarding any batch materialized under the old H/gamma. The C51
# endpoint remains an exact online-policy mixture.
# Between the first and second successful resets, the gradient-update rate is
# doubled. Because the TD, gamma, and imagination-warmup schedules are keyed to
# gradient steps, they intentionally progress twice as fast in environment time
# during that phase.
cd "$(dirname "$0")/.."
GAMES=${GAMES:-"Kangaroo Asterix"}
GPU=${GPU:-0}
RUN=${RUN:-1}

for game_name in $GAMES; do
	CUDA_VISIBLE_DEVICES=$GPU python -m bbf.train \
		--agent=BBF \
		--gin_files=bbf/configs/BBF-100K.gin \
		--gin_bindings="DataEfficientAtariRunner.game_name=\"$game_name\"" \
		--gin_bindings="BBFAgent.first_reset_update_multiplier=2" \
		--run_number="$RUN"
done
