set -ex
# Grounded + deeper TM: tests the capacity-conflict hypothesis -- that the
# Breakout grounding harm (B 252 / integration {202,258,353} vs anchor 363)
# came from a 1-layer TM forced to trade SPR fidelity against dense reward
# fitting, not from grounding per se. transition_hidden_layers=2 gives the
# reward objective somewhere to live inside the TM (projection-head
# principle), shielding the shared encoder.
#
# Config: grounded (reward_readout=False), deep TM, full imagination,
# value 0 and fixed gamma so the integration references stay clean.
# Bands: Breakout >=340 confirms capacity-conflict; ChopperCommand >=7000
# means grounding's auxiliary benefit (heads-on refs {9115, 9235}) survives
# with imagination on. Jointly: grounded+deep becomes a rival to the
# readout that KEEPS grounding's upside. Breakout ~250-270 again: depth
# does not rescue grounding, the readout interpretation stands.
cd "$(dirname "$0")/.."
GAMES=${GAMES:-"Breakout ChopperCommand"}
GPU=${GPU:-0}
RUN=${RUN:-45}
for game_name in $GAMES; do
	CUDA_VISIBLE_DEVICES=$GPU python -m bbf.train \
		--agent=BBF \
		--gin_files=bbf/configs/BBF-100K.gin \
		--gin_bindings="DataEfficientAtariRunner.game_name=\"$game_name\"" \
		--gin_bindings="bbf.spr_networks.RainbowDQNNetwork.transition_hidden_layers=2" \
		--gin_bindings="BBFAgent.reward_readout=False" \
		--gin_bindings="BBFAgent.imag_value_weight=0" \
		--gin_bindings="BBFAgent.imag_actor_weight=0.1" \
		--gin_bindings="BBFAgent.imag_entropy_weight=None" \
		--gin_bindings="BBFAgent.imag_discount=0.997" \
		--run_number=$RUN
done
