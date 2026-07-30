set -ex
# Strict one-step expected-C51 counterpart of 24-mid-phase-updates.sh. The
# historical RUN=144/145 rows retain their 10->3 replay-TD schedule; this row
# changes only the replay critic target:
#
#   1. update_horizon=max_update_horizon=1, giving the real transition
#      (s_t, a_t, r_{t+1}, done_{t+1}, s_{t+1}) throughout training;
#   2. current online-policy probabilities mix the target critic's complete
#      per-action C51 distributions instead of sampling one bootstrap action;
#   3. a configuration guard prevents this mode from silently using n>1.
#
# SPR, reward/continue supervision, and imagination remain five steps. The
# cyclic gamma schedule (0.97->0.997), PER, reset schedule, and ARM-E/F update
# window are unchanged, so RUN=144/145 remain the matched controls.
#
# Usage:
#   ARM=E bash ablations/26-mid-phase-one-step-expected.sh
#   ARM=F GAMES="DemonAttack Asterix" REPS=2 GPU=1 \
#     bash ablations/26-mid-phase-one-step-expected.sh
#
# RUN is a row id, not a seed. E/F use new ids so their results cannot be mixed
# with the historical multi-step rows.
cd "$(dirname "$0")/.."
GAMES=${GAMES:-"DemonAttack UpNDown Asterix RoadRunner"}
GPU=${GPU:-0}
ARM=${ARM:-E}
REPS=${REPS:-1}

case "$ARM" in
E) RUN=${RUN:-180}; LATE_AFTER=20000; LATE_UNTIL=40000 ;;
F) RUN=${RUN:-181}; LATE_AFTER=40000; LATE_UNTIL=60000 ;;
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
			--gin_bindings="BBFAgent.imag_discount=None" \
			--gin_bindings="BBFAgent.no_resets_after=100000" \
			--gin_bindings="BBFAgent.late_update_after=$LATE_AFTER" \
			--gin_bindings="BBFAgent.late_update_until=$LATE_UNTIL" \
			--gin_bindings="BBFAgent.late_update_multiplier=2" \
			--gin_bindings="BBFAgent.update_horizon=1" \
			--gin_bindings="BBFAgent.max_update_horizon=1" \
			--gin_bindings="BBFAgent.expected_one_step_backup=True" \
			--run_number=$RUN
	done
done
