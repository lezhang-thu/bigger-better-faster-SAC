set -ex
# Decomposition arm B: imagination OFF, reward/continue heads still ON.
# Isolates the imagined-actor-loss channel on the two confirmed loser games.
# jumps stays 5 (only imag_horizon is overridden), so SPR is unchanged.
#
# Read-out (vs full integration Breakout {258,202,353} / CC {4828,2831,5768},
# and vs baseline means Breakout 361 / CC 9976):
#   B recovers to ~baseline while integration was low -> imagined actor loss
#     is the interference; arm D not needed for these games.
#   B stays low -> the reward/continue grounding is the suspect -> run arm C
#     to confirm the anchor itself is healthy, then arm D.
cd "$(dirname "$0")/.."
GAMES=${GAMES:-"Breakout ChopperCommand"}
GPU=${GPU:-0}
RUN=${RUN:-21}
for game_name in $GAMES; do
	CUDA_VISIBLE_DEVICES=$GPU python -m bbf.train \
		--agent=BBF \
		--gin_files=bbf/configs/BBF-100K.gin \
		--gin_bindings="DataEfficientAtariRunner.game_name=\"$game_name\"" \
		--gin_bindings="BBFAgent.imag_horizon=0" \
		--run_number=$RUN
done
