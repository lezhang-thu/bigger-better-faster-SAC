set -ex
# Value-only ablation row (the combo minus the readout): grounded heads
# (reward_readout=False) + imag_value_weight=0.05, coupled imagination
# entropy, annealed imag discount, actor 0.1. This is the value-gate config
# at suite scale; the gate's seeds are poolable extras for 5 games:
# Hero {11253, 9468}, Jamesbond {390, 887}, Kangaroo 4324, CC 18668,
# Pong 19.55.
#
# Purpose: the "-readout" row of the ablation table, and full-scale
# attribution for the combo's hardened losses (DemonAttack, Alien, ...).
# Pre-registered: row IQM 1.05-1.15; DemonAttack >=20k iff the readout was
# its culprit; Gopher expected to stay lost (value-implicated).
# RUN=60 (seeds were random in the RUN=50/51 combo suites, so reusing 50
# would buy no pairing and muddy bookkeeping). Split via GAMES=...; keep
# RUN=60 everywhere so the row has a single id.
cd "$(dirname "$0")/.."
GAMES=${GAMES:-"Alien Amidar Assault Asterix BankHeist BattleZone Boxing Breakout ChopperCommand CrazyClimber DemonAttack Freeway Frostbite Gopher Hero Jamesbond Kangaroo Krull KungFuMaster MsPacman Pong PrivateEye Qbert RoadRunner Seaquest UpNDown"}
GPU=${GPU:-0}
RUN=${RUN:-60}
for game_name in $GAMES; do
	CUDA_VISIBLE_DEVICES=$GPU python -m bbf.train \
		--agent=BBF \
		--gin_files=bbf/configs/BBF-100K.gin \
		--gin_bindings="DataEfficientAtariRunner.game_name=\"$game_name\"" \
		--gin_bindings="BBFAgent.reward_readout=False" \
		--gin_bindings="BBFAgent.imag_value_weight=0.05" \
		--gin_bindings="BBFAgent.imag_actor_weight=0.1" \
		--gin_bindings="BBFAgent.imag_entropy_weight=None" \
		--gin_bindings="BBFAgent.imag_discount=None" \
		--run_number=$RUN
done
