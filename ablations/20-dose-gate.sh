set -ex
# Grounding-DOSE gate: readout=False with reward/continue head losses scaled
# to DOSE (default 0.5) instead of 1.0. A uniform MAGNITUDE reduction of the
# grounding gradient into encoder+TM -- the untried middle between grounded
# (dose 1.0 = value-only) and readout (dose 0.0 = combo). No projection, no
# second backward, no cap: gin-only, combo compute, and reward_grad_surgery
# stays False so this is IMMUNE to the v1/v2 surgery-flag hazard (runs the
# same on either branch).
#
# WHY 0.5 AND NOT 0.1 (user's imag-weight datapoint): DA's grounding benefit
# needs grounding to DOMINATE imagination. Observed: grounding 1.0 / imag
# 0.1,0.05 (value-only) -> DA 38923; grounding 1.0 / imag 1.0,1.0 -> DA <10k.
# The load-bearing quantity is the ratio grounding:imag. At DOSE=0.5 the ratio
# is 5-10x (still grounding-dominant, DA predicted safe); at DOSE=0.1 it is
# 1-2x -- the same regime where DA already collapsed. So 0.5 first; 0.1 only
# to map the curve, expecting DA to fall.
#
# THE SWEET-SPOT HYPOTHESIS (a NARROW but real path past combo's IQM, unlike
# the surgery family which only reshuffled the trimmed tail). Requires all
# three at DOSE=0.5:
#   1. DA survives      (ratio >=5-10x -- the datapoint above predicts yes)
#   2. Asterix spared   (IF Asterix harm is MAGNITUDE-mediated -- codex-v2
#                        removed CONFLICT and Asterix still fell to 4352, so
#                        the harm may be magnitude, not direction; halving it
#                        would then recover Asterix where projection couldn't)
#   3. mid-mass grounding WINS persist at half dose (BattleZone is IQM-kept:
#      combo 0.42 HNS vs grounded-family ~0.65-0.71 -- this is the IQM gain
#      source, the piece the mean-variant argument missed)
#
# Bands (endpoints: Asterix grounded 3514 / combo ~8537; BattleZone combo
# ~17000 / grounded ~25000-28000; DA combo ~17000 (5-seed) / value-only 38923
# / imag-1.0 <10000):
#   Asterix     >=7000   (cond 2, THE enabler -- ~4000 kills the whole idea)
#   BattleZone  >=24000  (cond 3, the IQM source -- ~17000 = win didn't survive)
#   DemonAttack >=22000  (cond 1, ratio safe at 0.5 -- <10000 = collapsed)
#   Breakout    >=330    (secondary mid-mass harm check; noisy, weather rules)
# Asterix>=7000 AND BattleZone>=24000 -> the sweet spot exists -> full suite.
# Either fails -> no IQM path; dose is at best a mean-variant like surgery.
#
# NOTE: DA is TRIMMED from IQM -- its recovery only moves the MEAN. The IQM
# verdict rides entirely on Asterix (spared) + BattleZone (kept). Read them
# first.
cd "$(dirname "$0")/.."
GAMES=${GAMES:-"Asterix BattleZone DemonAttack Breakout"}
GPU=${GPU:-0}
DOSE=${DOSE:-0.5}
RUN=${RUN:-130}
for game_name in $GAMES; do
	CUDA_VISIBLE_DEVICES=$GPU python -m bbf.train \
		--agent=BBF \
		--gin_files=bbf/configs/BBF-100K.gin \
		--gin_bindings="DataEfficientAtariRunner.game_name=\"$game_name\"" \
		--gin_bindings="BBFAgent.reward_readout=False" \
		--gin_bindings="BBFAgent.reward_grad_surgery=False" \
		--gin_bindings="BBFAgent.reward_weight=$DOSE" \
		--gin_bindings="BBFAgent.continue_weight=$DOSE" \
		--gin_bindings="BBFAgent.imag_value_weight=0.05" \
		--gin_bindings="BBFAgent.imag_value_trust=None" \
		--gin_bindings="BBFAgent.imag_actor_weight=0.1" \
		--gin_bindings="BBFAgent.imag_entropy_weight=None" \
		--gin_bindings="BBFAgent.imag_discount=None" \
		--run_number=$RUN
done
