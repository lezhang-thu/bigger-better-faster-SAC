set -ex
# Skip-60k-reset counterpart of 07-suite-combo.sh: identical full 26-game
# combo config, with only the late reset schedule changed.
#
# no_resets_after is a recovery-horizon cutoff, not the literal step of the
# last reset. With reset_every=20_000, setting it to 80_000 preserves the
# resets near 20k and 40k while skipping the reset near 60k. Using 60_000
# would also skip the 40k reset under the current reset scheduling logic.
#
# RUN=130 is distinct from the combo control (RUN=50). Split across boxes via
# GAMES=...; keep RUN=130 everywhere so this suite row has a single id.
cd "$(dirname "$0")/.."
GAMES=${GAMES:-"Alien Amidar Assault Asterix BankHeist BattleZone Boxing Breakout ChopperCommand CrazyClimber DemonAttack Freeway Frostbite Gopher Hero Jamesbond Kangaroo Krull KungFuMaster MsPacman Pong PrivateEye Qbert RoadRunner Seaquest UpNDown"}
GPU=${GPU:-0}
RUN=${RUN:-130}
for game_name in $GAMES; do
	CUDA_VISIBLE_DEVICES=$GPU python -m bbf.train \
		--agent=BBF \
		--gin_files=bbf/configs/BBF-100K.gin \
		--gin_bindings="DataEfficientAtariRunner.game_name=\"$game_name\"" \
		--gin_bindings="BBFAgent.reward_readout=True" \
		--gin_bindings="BBFAgent.imag_value_weight=0.05" \
		--gin_bindings="BBFAgent.imag_actor_weight=0.1" \
		--gin_bindings="BBFAgent.imag_entropy_weight=None" \
		--gin_bindings="BBFAgent.imag_discount=None" \
		--gin_bindings="BBFAgent.no_resets_after=80000" \
		--run_number=$RUN
done
