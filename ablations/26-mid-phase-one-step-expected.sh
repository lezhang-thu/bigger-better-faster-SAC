set -ex
# One-step expected-C51 counterpart of 24-mid-phase-updates.sh. Historical
# combo and ARM-E/F rows keep their original 10->3 replay-TD schedule; this
# script is the new row with exactly three replay-target changes:
#
#   1. update_horizon=max_update_horizon=1, so every replay critic target is
#      (s_t, a_t, r_{t+1}, s_{t+1}) throughout training and after every reset;
#   2. the bootstrap mixes the target-critic C51 distributions using current
#      online-policy probabilities instead of sampling one action;
#   3. the expected-one-step guard rejects any future binding that silently
#      reintroduces a multi-step horizon.
#
# SPR, reward/continue supervision, and imagination remain five steps. The
# cyclic gamma schedule (0.97->0.997), reset schedule, ARM-E/F update window,
# and uniform replay setting are inherited unchanged from the current ARM row.
#
# Usage:
#   ARM=E bash ablations/26-mid-phase-one-step-expected.sh
#   ARM=F GAMES="DemonAttack Asterix" REPS=2 GPU=1 \
#     bash ablations/26-mid-phase-one-step-expected.sh
#
# RUN is a row id, not a seed. E/F use new ids so these results cannot be mixed
# with the historical RUN=144/145 multi-step rows.
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
			--gin_bindings="PrioritizedJaxSubsequenceParallelEnvReplayBuffer.prioritized_sampling=False" \
			--gin_bindings="BBFAgent.update_horizon=1" \
			--gin_bindings="BBFAgent.max_update_horizon=1" \
			--gin_bindings="BBFAgent.expected_one_step_backup=True" \
			--run_number=$RUN
	done
done
