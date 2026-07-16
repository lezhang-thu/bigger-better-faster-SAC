set -ex
# Decomposition arm D: reward/continue heads become strict READOUTS
# (BBFAgent.reward_readout=True stop-gradients rollout_reps into the heads),
# imagination stays ON. Run only after B/C point at the grounding channel
# (B stays low, C recovers).
#
# Prediction if the grounding hypothesis is right: Breakout/CC recover toward
# baseline with imagination still active. By construction this arm cannot
# hurt the sparse-reward winners (their reward gradients are ~0 already).
# Other knobs pinned to the c6e22ae-comparable operating point (actor .1,
# value 0, entropy coupled, fixed gamma) so the only delta vs the existing
# integration seeds is the readout.
cd "$(dirname "$0")/.."
GAMES=${GAMES:-"Breakout ChopperCommand"}
GPU=${GPU:-0}
RUN=${RUN:-23}
for game_name in $GAMES; do
	CUDA_VISIBLE_DEVICES=$GPU python -m bbf.train \
		--agent=BBF \
		--gin_files=bbf/configs/BBF-100K.gin \
		--gin_bindings="DataEfficientAtariRunner.game_name=\"$game_name\"" \
		--gin_bindings="BBFAgent.reward_readout=True" \
		--gin_bindings="BBFAgent.imag_actor_weight=0.1" \
		--gin_bindings="BBFAgent.imag_value_weight=0" \
		--gin_bindings="BBFAgent.imag_entropy_weight=None" \
		--gin_bindings="BBFAgent.imag_discount=0.997" \
		--run_number=$RUN
done
