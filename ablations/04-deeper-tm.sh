set -ex
# Deeper transition model: transition_hidden_layers=2 (one extra hidden conv
# in ConvTMCell; default 1 = original SPR cell). The TM carries SPR + reward +
# continue + 5-step imagination on a single 3x3 conv today; capacity is a
# plausible co-factor in the dense-game losses.
#
# CAUTION: this changes the anchor's own SPR objective, so gate on the tight
# win/parity games BEFORE trying it on the loser games:
#   Pong refs 17.41-20.41 | Gopher refs 1533-1808 | Hero refs 7202/7590/(3590)
# Other knobs pinned to the c6e22ae-comparable operating point.
cd "$(dirname "$0")/.."
GAMES=${GAMES:-"Pong Gopher Hero"}
GPU=${GPU:-0}
RUN=${RUN:-26}
for game_name in $GAMES; do
	CUDA_VISIBLE_DEVICES=$GPU python -m bbf.train \
		--agent=BBF \
		--gin_files=bbf/configs/BBF-100K.gin \
		--gin_bindings="DataEfficientAtariRunner.game_name=\"$game_name\"" \
		--gin_bindings="bbf.spr_networks.RainbowDQNNetwork.transition_hidden_layers=2" \
		--gin_bindings="BBFAgent.imag_actor_weight=0.1" \
		--gin_bindings="BBFAgent.imag_value_weight=0" \
		--gin_bindings="BBFAgent.imag_entropy_weight=None" \
		--gin_bindings="BBFAgent.imag_discount=0.997" \
		--run_number=$RUN
done
