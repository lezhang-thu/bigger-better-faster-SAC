set -ex
# True-floor entropy: imag_entropy_floor=3e-4 -> coefficient becomes
# max(x_ent_coef, 3e-4). Unlike the withdrawn replacement variant (which
# scored Pong 11.54 by also stripping the early 1e-2 entropy), this only
# adds protection after x_ent_coef decays below 3e-4 (~76k env steps on).
#
# Same games/pins as ablations/03 so the three-way comparison is clean:
#   coupled (references):  Pong 17.41-20.41 x5 | Gopher 1533-1808 x4
#   replacement 3e-4:      Pong 11.54 FAIL     | Gopher 1554 weak pass
# Bands: Pong >=17 passes (and confirms early-stripping was the harm
# mechanism); Gopher >=1500 passes.
cd "$(dirname "$0")/.."
GAMES=${GAMES:-"Pong Gopher"}
GPU=${GPU:-0}
RUN=${RUN:-41}
for game_name in $GAMES; do
	CUDA_VISIBLE_DEVICES=$GPU python -m bbf.train \
		--agent=BBF \
		--gin_files=bbf/configs/BBF-100K.gin \
		--gin_bindings="DataEfficientAtariRunner.game_name=\"$game_name\"" \
		--gin_bindings="BBFAgent.imag_entropy_floor=3e-4" \
		--gin_bindings="BBFAgent.imag_entropy_weight=None" \
		--gin_bindings="BBFAgent.imag_actor_weight=0.1" \
		--gin_bindings="BBFAgent.imag_value_weight=0" \
		--gin_bindings="BBFAgent.imag_discount=0.997" \
		--run_number=$RUN
done
