set -ex
# Aggressive grounded arm: the value-gate config plus a deeper TM.
# grounded (reward_readout=False) + transition_hidden_layers=2 +
# imag_value_weight=0.05 + annealed imag discount + actor 0.1.
#
# Single delta (depth) against the measured value-gate family
# (grounded+shallow+value+annealed): Kangaroo 4324, Hero 11253/9468,
# CC 18668, Pong 19.55, Jamesbond {390, 887}. If this arm fixes the
# grounding-harmed losers while keeping the value wins, it is a full
# rival to the RUN=50 readout combo -- one that preserves grounding's
# auxiliary upside.
#
# Bands: Breakout >=340 (integration {202,258,353}, anchor 363);
# ChopperCommand >=7000 (weather caveat, 3-seed rule applies);
# Hero >=9000 (value refs 9468-12879); Pong >=17 (ref 19.55).
cd "$(dirname "$0")/.."
GAMES=${GAMES:-"Breakout ChopperCommand Hero Pong"}
GPU=${GPU:-0}
RUN=${RUN:-46}
for game_name in $GAMES; do
	CUDA_VISIBLE_DEVICES=$GPU python -m bbf.train \
		--agent=BBF \
		--gin_files=bbf/configs/BBF-100K.gin \
		--gin_bindings="DataEfficientAtariRunner.game_name=\"$game_name\"" \
		--gin_bindings="bbf.spr_networks.RainbowDQNNetwork.transition_hidden_layers=2" \
		--gin_bindings="BBFAgent.reward_readout=False" \
		--gin_bindings="BBFAgent.imag_value_weight=0.05" \
		--gin_bindings="BBFAgent.imag_actor_weight=0.1" \
		--gin_bindings="BBFAgent.imag_entropy_weight=None" \
		--gin_bindings="BBFAgent.imag_discount=None" \
		--run_number=$RUN
done
