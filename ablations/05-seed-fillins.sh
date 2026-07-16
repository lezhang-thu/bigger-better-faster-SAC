set -ex
# Seed fill-ins to make the 26-game c6e22ae-config table citable:
#   Seaquest x2, UpNDown x2 (zero clean-config seeds so far),
#   Frostbite +1 (the +113% headline currently rests on n=1),
#   Freeway +1, DemonAttack +1, RoadRunner +1.
# Knobs pinned to the c6e22ae-comparable operating point. Residual delta vs
# true c6e22ae that bindings cannot remove: commit 272a784 PER-weights the
# imagined actor loss (effective actor weight ~halved). Note it in the table.
cd "$(dirname "$0")/.."
GAMES_ONCE=${GAMES_ONCE:-"Frostbite Freeway DemonAttack RoadRunner"}
GAMES_TWICE=${GAMES_TWICE:-"Seaquest UpNDown"}
GPU=${GPU:-0}
RUN=${RUN:-31}
RUN2=${RUN2:-32}
launch() {
	CUDA_VISIBLE_DEVICES=$GPU python -m bbf.train \
		--agent=BBF \
		--gin_files=bbf/configs/BBF-100K.gin \
		--gin_bindings="DataEfficientAtariRunner.game_name=\"$1\"" \
		--gin_bindings="BBFAgent.imag_actor_weight=0.1" \
		--gin_bindings="BBFAgent.imag_value_weight=0" \
		--gin_bindings="BBFAgent.imag_entropy_weight=None" \
		--gin_bindings="BBFAgent.imag_discount=0.997" \
		--run_number=$2
}
for game_name in $GAMES_TWICE; do
	launch "$game_name" $RUN
	launch "$game_name" $RUN2
done
for game_name in $GAMES_ONCE; do
	launch "$game_name" $RUN
done
