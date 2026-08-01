set -ex
# Tree-Backup counterpart of 27-mid-phase-one-step-expected-fixed-gamma.sh.
# The root critic loss is still applied only to (s_0, a_0), but its target can
# carry ten true replay rewards.  At each intermediate state, target-C51
# branches for non-replayed actions terminate under the current online policy;
# only the replayed action branch continues to the next real reward.
#
# Tree Backup has its own fixed horizon 10.  MODEL_HORIZON remains 5, so SPR,
# reward/continue learning, and imagination are unchanged.  The replay sampler
# returns an 11-state segment with raw one-step rewards/terminals rather than
# invoking the repository's legacy n-step return construction.
#
# This keeps script 27 as the direct one-step expected-C51 control.  Both rows
# use gamma=0.997, the combo reset/readout/imagination settings, and exactly one
# doubled 20k update window.
#
# Usage:
#   ARM=E bash ablations/29-mid-phase-tree-backup-fixed10-fixed-gamma.sh
#   ARM=F GAMES="Kangaroo Asterix" REPS=2 GPU=1 \
#     bash ablations/29-mid-phase-tree-backup-fixed10-fixed-gamma.sh
#
# RUN is a row id, not a seed.
cd "$(dirname "$0")/.."
GAMES=${GAMES:-"Kangaroo Asterix"}
GPU=${GPU:-0}
ARM=${ARM:-E}
REPS=${REPS:-1}

case "$ARM" in
E) RUN=${RUN:-188}; LATE_AFTER=20000; LATE_UNTIL=40000 ;;
F) RUN=${RUN:-189}; LATE_AFTER=40000; LATE_UNTIL=60000 ;;
*) echo "ARM must be E ([20k,40k)) or F ([40k,60k))" >&2; exit 1 ;;
esac

for ((rep = 1; rep <= REPS; rep++)); do
	for game_name in $GAMES; do
		CUDA_VISIBLE_DEVICES=$GPU python -m bbf.train \
			--agent=BBF \
			--gin_files=bbf/configs/BBF-100K.gin \
			--gin_bindings="DataEfficientAtariRunner.game_name=\"$game_name\"" \
			--gin_bindings="BBFAgent.reward_readout=True" \
			--gin_bindings="BBFAgent.imag_value_weight=0.05" \
			--gin_bindings="BBFAgent.imag_actor_weight=0.1" \
			--gin_bindings="BBFAgent.imag_entropy_weight=None" \
			--gin_bindings="JaxDQNAgent.gamma=0.997" \
			--gin_bindings="BBFAgent.min_gamma=None" \
			--gin_bindings="BBFAgent.imag_discount=None" \
			--gin_bindings="BBFAgent.no_resets_after=100000" \
			--gin_bindings="BBFAgent.late_update_after=$LATE_AFTER" \
			--gin_bindings="BBFAgent.late_update_until=$LATE_UNTIL" \
			--gin_bindings="BBFAgent.late_update_multiplier=2" \
			--gin_bindings="BBFAgent.update_horizon=1" \
			--gin_bindings="BBFAgent.max_update_horizon=1" \
			--gin_bindings="BBFAgent.expected_one_step_backup=False" \
			--gin_bindings="BBFAgent.tree_backup=True" \
			--gin_bindings="BBFAgent.tree_backup_horizon=10" \
			--run_number=$RUN
	done
done
