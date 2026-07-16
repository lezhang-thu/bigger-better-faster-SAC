set -ex
# Entropy-floor validation: imag_entropy_weight=3e-4 (the new default) as the
# ONLY delta vs the c6e22ae-comparable operating point. Pong and Gopher have
# the tightest same-config reference distributions, so one seed each detects
# harm cheaply:
#   Pong references:   17.41 / 18.53 / 19.66 / 19.75 / 20.41
#   Gopher references: 1533.6 / 1683.2 / 1772.8 / 1808.4
# Expected: neutral-to-positive (the floor mostly matters after x_ent_coef
# hits 0 at 80k env steps). A Pong <16 or Gopher <1400 would argue against
# adopting the floor as default.
cd "$(dirname "$0")/.."
GAMES=${GAMES:-"Pong Gopher"}
GPU=${GPU:-0}
RUN=${RUN:-25}
for game_name in $GAMES; do
	CUDA_VISIBLE_DEVICES=$GPU python -m bbf.train \
		--agent=BBF \
		--gin_files=bbf/configs/BBF-100K.gin \
		--gin_bindings="DataEfficientAtariRunner.game_name=\"$game_name\"" \
		--gin_bindings="BBFAgent.imag_entropy_weight=3e-4" \
		--gin_bindings="BBFAgent.imag_actor_weight=0.1" \
		--gin_bindings="BBFAgent.imag_value_weight=0" \
		--gin_bindings="BBFAgent.imag_discount=0.997" \
		--run_number=$RUN
done
