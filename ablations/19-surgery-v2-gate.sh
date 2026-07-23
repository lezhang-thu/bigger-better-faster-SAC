set -ex
# Surgery-v2 gate: TRUNK-RESTRICTED grounding-gradient surgery -- the
# projection v1 always claimed to be. v1 (all RUN=70/90/91/92 surgery rows)
# flattened cos/norm over the ENTIRE param tree: diluted cosine, undersized
# trunk correction, and a spurious (1+|coef|) rescale of the main gradient
# on Q/policy leaves at conflict steps. v2 restricts the reduction and the
# correction to encoder+transition_model and logs undiluted per-module
# diagnostics (GroundCosEnc / GroundCosTM / GroundNormRatio). Pre-v2
# GroundCos logs are the diluted global quantity -- NOT comparable.
#
# Smoke evidence the fix matters (Pong, 1.2k steps): undiluted trunk cos
# reaches -0.19 where v1 logged +/-0.05; encoder and TM conflicts carry
# OPPOSITE signs at the same step (enc +0.13 / TM -0.14); early
# GroundNormRatio 2.79 -- grounding ~3x the main trunk gradient -- collapsing
# to 0.06 when the continue head is detached (the dense life-loss BCE was
# nearly all of the early grounding pressure).
#
# Bands (weather rules on CC; n=1 proves little):
#   DemonAttack   >=22000  (grounded family 20448-50333; v1 surgery 49494)
#   Asterix       >=7000   (readout family 7246-9828; v1 {7214, 12431})
#   Breakout      >=330    (readout family 324-365; v1 361.5)
#   CrazyClimber  >=95000  (v1 110414; combo 82-83k)
#   ChopperCommand -- the REOPENED cell: v1's diluted projection never
#     protected its late climb (combo {16669,10623} vs v1 {7204,4661}); the
#     undersized correction is a live suspect. Read GroundCosTM here either
#     way: undiluted TM conflict decides whether CC's interaction channel is
#     projection-visible at all.
#
# Arm B (contingent, after A's logs): CR=True RUN=121 -> continue head
# detached, reward stays grounded. Motivation: continue BCE is dense on
# every game via life-loss terminals (the s92 BankHeist conflict was PURE
# continue: RewardLoss 0.0000, 47% negative cos), so "sparse games are safe"
# was false as stated; the split makes sparse-game trunks actually
# untouched. Zero scalar knobs either arm.
cd "$(dirname "$0")/.."
GAMES=${GAMES:-"DemonAttack Asterix Breakout CrazyClimber ChopperCommand"}
GPU=${GPU:-0}
RUN=${RUN:-120}
CR=${CR:-False}
for game_name in $GAMES; do
	CUDA_VISIBLE_DEVICES=$GPU python -m bbf.train \
		--agent=BBF \
		--gin_files=bbf/configs/BBF-100K.gin \
		--gin_bindings="DataEfficientAtariRunner.game_name=\"$game_name\"" \
		--gin_bindings="BBFAgent.reward_readout=False" \
		--gin_bindings="BBFAgent.reward_grad_surgery=True" \
		--gin_bindings="BBFAgent.continue_readout=$CR" \
		--gin_bindings="BBFAgent.imag_value_weight=0.05" \
		--gin_bindings="BBFAgent.imag_value_trust=None" \
		--gin_bindings="BBFAgent.imag_actor_weight=0.1" \
		--gin_bindings="BBFAgent.imag_entropy_weight=None" \
		--gin_bindings="BBFAgent.imag_discount=None" \
		--run_number=$RUN
done
