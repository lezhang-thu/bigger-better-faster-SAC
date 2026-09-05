set -ex
# Replay-only ablation: TD + off-policy actor + SPR. World-model
# reward/continue heads and imagined actor-critic learning are removed.
cd "$(dirname "$0")/.."
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-/tmp/bbf-jax-compilation-cache}
GAMES=${GAMES:-"Alien Amidar Assault Asterix BankHeist BattleZone Boxing Breakout ChopperCommand CrazyClimber DemonAttack Freeway Frostbite Gopher Hero Jamesbond Kangaroo Krull KungFuMaster MsPacman Pong PrivateEye Qbert RoadRunner Seaquest UpNDown"}
GPU=${GPU:-0}
SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7 8 9"}
for game_name in $GAMES; do
	for seed in $SEEDS; do
		CUDA_VISIBLE_DEVICES=$GPU python -m bbf.train \
			--agent=BBF \
			--gin_files=bbf/configs/BBF-100K.gin \
			--gin_bindings="DataEfficientAtariRunner.game_name=\"$game_name\"" \
			--gin_bindings="BBFAgent.post_skipped_reset_update_multiplier=2" \
			--gin_bindings="BBFAgent.reset_priorities=False" \
			--gin_bindings="BBFAgent.update_horizon=1" \
			--gin_bindings="BBFAgent.cycle_steps=40_000" \
			--no_seeding=False \
			--run_number=$seed
	done
done
