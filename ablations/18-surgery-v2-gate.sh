set -ex
# Paired gate for scoped grounding-gradient surgery (v2).
#
# Arms (run each with the SAME positive SEED):
#   ARM=combo    strict reward+continue readouts (07 control), RUN default 110
#   ARM=surgery legacy full-tree PCGrad (17 control),          RUN default 111
#   ARM=v2       shared-only PCGrad + continuation readout +
#                post-projection grounding norm cap,           RUN default 112
#
# Example, seed 110:
#   ARM=combo    SEED=110 bash ablations/18-surgery-v2-gate.sh
#   ARM=surgery  SEED=110 bash ablations/18-surgery-v2-gate.sh
#   ARM=v2       SEED=110 bash ablations/18-surgery-v2-gate.sh
# Repeat all three with SEED=111 and distinct RUN values for seed 2.
#
# Default games preserve surgery's gains (Asterix/DemonAttack) and expose its
# broad regressions (BankHeist/Frostbite/Pong). Promote v2 only on the paired
# two-seed panel, not on one favorable cell. Suggested raw-score guardrails:
# Asterix >=7000, DemonAttack >=20000, BankHeist >=600,
# Frostbite >=1500, Pong >=14.
cd "$(dirname "$0")/.."
GAMES=${GAMES:-"Asterix DemonAttack BankHeist Frostbite Pong"}
GPU=${GPU:-0}
ARM=${ARM:-v2}
SEED=${SEED:-110}
CAP=${CAP:-0.25}

if ! [ "$SEED" -gt 0 ] 2>/dev/null; then
	echo "SEED must be a positive integer (got $SEED)" >&2
	exit 2
fi

case "$ARM" in
	combo)
		RUN_DEFAULT=110
		REWARD_READOUT=True
		CONTINUE_READOUT=False
		SURGERY=False
		SHARED_SURGERY=False
		NORM_CAP=None
		;;
	surgery)
		RUN_DEFAULT=111
		REWARD_READOUT=False
		CONTINUE_READOUT=False
		SURGERY=True
		SHARED_SURGERY=False
		NORM_CAP=None
		;;
	v2)
		RUN_DEFAULT=112
		REWARD_READOUT=False
		CONTINUE_READOUT=True
		SURGERY=True
		SHARED_SURGERY=True
		NORM_CAP=$CAP
		;;
	*)
		echo "Unknown ARM=$ARM (expected combo, surgery, or v2)" >&2
		exit 2
		;;
esac
RUN=${RUN:-$RUN_DEFAULT}

for game_name in $GAMES; do
	CUDA_VISIBLE_DEVICES=$GPU python -m bbf.train \
		--agent=BBF \
		--gin_files=bbf/configs/BBF-100K.gin \
		--gin_bindings="DataEfficientAtariRunner.game_name=\"$game_name\"" \
		--gin_bindings="BBFAgent.reward_readout=$REWARD_READOUT" \
		--gin_bindings="BBFAgent.continue_readout=$CONTINUE_READOUT" \
		--gin_bindings="BBFAgent.reward_grad_surgery=$SURGERY" \
		--gin_bindings="BBFAgent.reward_grad_surgery_shared=$SHARED_SURGERY" \
		--gin_bindings="BBFAgent.grounding_grad_norm_ratio=$NORM_CAP" \
		--gin_bindings="BBFAgent.imag_value_weight=0.05" \
		--gin_bindings="BBFAgent.imag_value_trust=None" \
		--gin_bindings="BBFAgent.imag_actor_weight=0.1" \
		--gin_bindings="BBFAgent.imag_entropy_weight=None" \
		--gin_bindings="BBFAgent.imag_discount=None" \
		--no_seeding=False \
		--agent_seed=$SEED \
		--run_number=$RUN
done
