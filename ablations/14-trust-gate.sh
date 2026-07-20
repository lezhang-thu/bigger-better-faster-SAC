set -ex
# Trust gate. Thesis: bounding the imagined critic targets within
# imag_value_trust * return-scale of the target critic's estimate recovers
# the value loss's casualties without touching its wins.
#
# Config: the full combo (readout + value 0.05) plus imag_value_trust=0.5.
# Games:
#   Gopher   >=1500  (the value casualty: combo {873, 1142} vs the
#                     integration family 1533-1808 and readout-only 1608)
#   Hero     >=9000  (the value win {9468, 11253, 12877/12879} MUST survive)
#   Jamesbond weather rules (jackpot game; combo {2549, 1370})
# Watch ImagTrustClip in the logs: expect ~20% early, settling to a few
# percent; pinned near 0 would mean the band never binds (raise trust or
# conclude no effect), pinned high late means the band is too tight.
cd "$(dirname "$0")/.."
GAMES=${GAMES:-"Gopher Jamesbond Hero"}
GPU=${GPU:-0}
RUN=${RUN:-71}
for game_name in $GAMES; do
	CUDA_VISIBLE_DEVICES=$GPU python -m bbf.train \
		--agent=BBF \
		--gin_files=bbf/configs/BBF-100K.gin \
		--gin_bindings="DataEfficientAtariRunner.game_name=\"$game_name\"" \
		--gin_bindings="BBFAgent.reward_readout=True" \
		--gin_bindings="BBFAgent.reward_grad_surgery=False" \
		--gin_bindings="BBFAgent.imag_value_weight=0.05" \
		--gin_bindings="BBFAgent.imag_value_trust=0.5" \
		--gin_bindings="BBFAgent.imag_actor_weight=0.1" \
		--gin_bindings="BBFAgent.imag_entropy_weight=None" \
		--gin_bindings="BBFAgent.imag_discount=None" \
		--run_number=$RUN
done
