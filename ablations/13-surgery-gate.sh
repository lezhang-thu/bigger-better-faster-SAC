set -ex
# Legacy surgery gate. This preserves the original full-parameter-tree PCGrad
# implementation for comparison with the scoped surgery-v2 gate in script 18.
# It was intended to arbitrate grounding at the shared encoder/TM, but the
# historical implementation also projects/rescales non-shared leaves.
#
# Games are the three battlegrounds of the grounded-vs-readout split:
#   Breakout  >=340  (readout fixed it: {324, 352}; grounded broke it: ~271)
#   ChopperCommand >=7000 (grounded heads helped: B-arm {9115, 9235};
#                          weather rules, n=1 proves little)
#   DemonAttack >=20000 (the combo's -7.1 HNS sacrifice {13562, 18283};
#                        integration/grounded family was {20448-50333})
# All three passing = full thesis confirmed; watch GroundCos in the logs
# (negative dips = the projection engaging).
# Config: grounded + surgery + the combo's other knobs (value 0.05,
# actor 0.1, coupled entropy, annealed discount). Step-time cost ~+13%.
cd "$(dirname "$0")/.."
GAMES=${GAMES:-"Breakout ChopperCommand DemonAttack"}
GPU=${GPU:-0}
RUN=${RUN:-70}
for game_name in $GAMES; do
	CUDA_VISIBLE_DEVICES=$GPU python -m bbf.train \
		--agent=BBF \
		--gin_files=bbf/configs/BBF-100K.gin \
		--gin_bindings="DataEfficientAtariRunner.game_name=\"$game_name\"" \
		--gin_bindings="BBFAgent.reward_readout=False" \
		--gin_bindings="BBFAgent.continue_readout=False" \
		--gin_bindings="BBFAgent.reward_grad_surgery=True" \
		--gin_bindings="BBFAgent.reward_grad_surgery_shared=False" \
		--gin_bindings="BBFAgent.grounding_grad_norm_ratio=None" \
		--gin_bindings="BBFAgent.imag_value_weight=0.05" \
		--gin_bindings="BBFAgent.imag_value_trust=None" \
		--gin_bindings="BBFAgent.imag_actor_weight=0.1" \
		--gin_bindings="BBFAgent.imag_entropy_weight=None" \
		--gin_bindings="BBFAgent.imag_discount=None" \
		--run_number=$RUN
done
