set -ex
# Counterpart of 07-suite-combo.sh with the deeper transition model:
# identical combo config (reward_readout=True, imag_value_weight=0.05,
# coupled entropy, annealed imag discount, actor 0.1) plus
# transition_hidden_layers=2 (one extra hidden conv in ConvTMCell; default
# 1 = original SPR cell). Single delta vs 07, so RUN=50-vs-RUN=51 is a
# matched suite-level A/B.
#
# CAUTION: transition_hidden_layers=2 has never been validated (04 was
# never run) and it perturbs the anchor's own SPR objective. The cheap
# de-risk before committing 26 games: GAMES="Pong Gopher Hero" first
# (refs: Pong 17.4-20.4, Gopher 1533-1808, Hero-under-combo 12879).
cd "$(dirname "$0")/.."
GAMES=${GAMES:-"Alien Amidar Assault Asterix BankHeist BattleZone Boxing Breakout ChopperCommand CrazyClimber DemonAttack Freeway Frostbite Gopher Hero Jamesbond Kangaroo Krull KungFuMaster MsPacman Pong PrivateEye Qbert RoadRunner Seaquest UpNDown"}
GPU=${GPU:-0}
RUN=${RUN:-51}
for game_name in $GAMES; do
	CUDA_VISIBLE_DEVICES=$GPU python -m bbf.train \
		--agent=BBF \
		--gin_files=bbf/configs/BBF-100K.gin \
		--gin_bindings="DataEfficientAtariRunner.game_name=\"$game_name\"" \
		--gin_bindings="bbf.spr_networks.RainbowDQNNetwork.transition_hidden_layers=2" \
		--gin_bindings="BBFAgent.reward_readout=True" \
		--gin_bindings="BBFAgent.imag_value_weight=0.05" \
		--gin_bindings="BBFAgent.imag_actor_weight=0.1" \
		--gin_bindings="BBFAgent.imag_entropy_weight=None" \
		--gin_bindings="BBFAgent.imag_discount=None" \
		--run_number=$RUN
done
