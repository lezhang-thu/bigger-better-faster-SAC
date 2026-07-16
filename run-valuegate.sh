set -ex
# Gate for imag_value_weight (model-generated targets into the online critic).
# Games: pilot-collapse trio (Kangaroo/Hero/Jamesbond) + ChopperCommand (worst
# RewardCorr 0.51 with dense rewards, the confirmed weak game) + Pong (tight
# win-game canary). Weights are bound explicitly on the command line so each
# log records its own config and later gin edits cannot leak into the queue.
#
# Usage: [GAMES="..."] [GPU=0] [RUN=13] [VW=0.05] [EW=None] bash run-valuegate.sh
#   control arm: VW=0 RUN=14 bash run-valuegate.sh
# EW=None pins imagination entropy to the annealed x_ent_coef (pre-floor
# behavior) so this gate stays single-variable against the c6e22ae
# reference seeds; the gin's new 3e-4 default is deliberately overridden.
GAMES=${GAMES:-"Kangaroo Hero Jamesbond ChopperCommand Pong"}
GPU=${GPU:-0}
RUN=${RUN:-13}
VW=${VW:-0.05}
EW=${EW:-None}
for game_name in $GAMES; do
	CUDA_VISIBLE_DEVICES=$GPU python -m bbf.train \
		--agent=BBF \
		--gin_files=bbf/configs/BBF-100K.gin \
		--gin_bindings="DataEfficientAtariRunner.game_name=\"$game_name\"" \
		--gin_bindings="BBFAgent.imag_actor_weight=0.1" \
		--gin_bindings="BBFAgent.imag_value_weight=$VW" \
		--gin_bindings="BBFAgent.imag_entropy_weight=$EW" \
		--run_number=$RUN
done
