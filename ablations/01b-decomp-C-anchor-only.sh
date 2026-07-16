set -ex
# Decomposition arm C: pure anchor -- imagination AND reward/continue heads
# all off. Two purposes:
#   1. Sanity: should land inside the 10-seed baseline range (Breakout
#      258-401, CC 1393-17658). If C is ALSO low, something else drifted and
#      arms B/D cannot be interpreted.
#   2. Wall-clock: this is the still-missing anchor-only timing reference
#      (conv-TM integration Pong was 2h12m solo; note the per-game runtime
#      from the log timestamps).
cd "$(dirname "$0")/.."
GAMES=${GAMES:-"Breakout ChopperCommand"}
GPU=${GPU:-0}
RUN=${RUN:-22}
for game_name in $GAMES; do
	CUDA_VISIBLE_DEVICES=$GPU python -m bbf.train \
		--agent=BBF \
		--gin_files=bbf/configs/BBF-100K.gin \
		--gin_bindings="DataEfficientAtariRunner.game_name=\"$game_name\"" \
		--gin_bindings="BBFAgent.imag_horizon=0" \
		--gin_bindings="BBFAgent.reward_weight=0" \
		--gin_bindings="BBFAgent.continue_weight=0" \
		--run_number=$RUN
done
