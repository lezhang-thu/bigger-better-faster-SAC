set -ex
# RUN=130 (skip the 60k reset) + the actor entropy coefficient annealing to zero
# at 100k instead of 80k. Single delta on 20-suite-combo-skip-60k-reset.sh,
# which has 46 runs of reference, and FREE: the entropy schedule costs no extra
# gradient work, so this is ~2.14 h/game, the same as the row it modifies.
#
# THE MOTIVATION. x_ent_coef is the only live actor entropy coefficient (the SAC
# learned-alpha form in policy_loss sits behind `if False`, and _log_alpha is
# inert though still preserved across resets). It decays 1e-2 -> 0 over
# x_ent_decay_steps and is then clipped at EXACTLY 0, so at the shipped 80_000
# the last 20k env steps of a 100k run train with no entropy term at all -- in
# the imagined actor loss too, since imag_entropy_weight=None couples it to the
# same value. Measured across the 46 RUN=130 runs, ImagEntropy drifts down
# through that window in 45/46 runs (median 1.301 over 60-80k -> 1.131 over
# 80-90k -> 1.109 over 90-100k, decelerating). Eval is a sampling pass from that
# policy, and this repo has measured that determinism costs real score --
# Breakout 380.6 sampled vs 335.4 near-greedy vs 316.0 argmax -- so a policy
# drifting deterministic in the last fifth of training is losing eval score by
# exactly that mechanism. Setting the decay to 100_000 keeps the coefficient
# positive until the run ends (0.002 at 80k, 0.001 at 90k, 1e-4 at 99k).
#
# THE RISK, stated plainly. This is NOT a tail-only change: a 100k decay raises
# the coefficient at every earlier step too (0.006 vs 0.005 at 40k, 0.0021 vs
# 0.00013 at 79k), so it perturbs the whole back half of the schedule every
# current result was obtained under. And in the imagination channel it is a
# STRONGER dose of the direction that already failed: the decoupled floor
# max(x_ent_coef, 3e-4) gave Pong 14.11 and Gopher 582.8 against references of
# 17.41-20.41 and 1533-1808, and this schedule sits above 3e-4 all the way to
# ~97k. The distinction that makes it worth one run: those failures DECOUPLED
# imagination from x_ent_coef while the replay actor still went to zero, whereas
# this moves both together. The coupling is the part the entropy line actually
# validated; the schedule endpoint was never ablated (it was a hardcoded
# int(80e3) until this row needed it). Whether coupling was the load-bearing
# part is exactly what this tests.
#
# PANEL. Breakout is the primary cell: it is the game where determinism was
# measured to cost the most (380.6 / 335.4 / 316.0), so if late entropy collapse
# is bleeding eval score, it shows there first. Pong is the safety canary -- it
# is the game the entropy line was decided on, it sits near its 21 ceiling under
# RUN=130 {20.21, 20.97} so it can only reveal damage, and the two failed
# entropy variants dropped it to 14.11 and 11.54. Gopher is the other
# tight-reference entropy game with room to move. Asterix is the IQM-relevant
# mid-mass cell RUN=130 gave back ({5309.5, 4206} vs combo {7246.5, 9828}).
#
# PRE-REGISTERED BANDS (references are RUN=130's own seeds):
#   Breakout  {374.36, 329.58}; anchor 258.8-401.0.
#             >=~390 = collapse was real and this fixes it; ~350 neutral;
#             <=~300 = harm. Weak band by construction (5.04 HNS seed tail).
#   Pong      {20.21, 20.97}. >=~19 safe; <=~16 = the floor-experiment failure
#             mode reproduced under coupling too -> kill the row immediately.
#   Gopher    {977.6, 1004, 776}; combo {873, 1142}; readout family 1533-1808.
#             >=~1100 = real gain; <=~700 = harm.
#   Asterix   {5309.5, 4206}; combo {7246.5, 9828}. >=~7000 = the late entropy
#             recovers what skipping the reset cost mid-mass.
# DECISION: Pong safe AND (Breakout or Gopher) up -> 26-game suite (GAMES=...
# with the full list, ~56 h, no compute premium). Pong down -> the anneal-to-
# zero is load-bearing in the replay actor as well, and the whole entropy line
# stays closed. If it passes, the natural follow-up is stacking it with the
# late-update doubling (21-late-updates-gate.sh), since the zero-entropy window
# is what made doubling there questionable in the first place.
#
# Usage: [GAMES=...] [GPU=0] [REPS=1] bash ablations/22-skip-60k-entropy-100k.sh
# RUN=150 is the row id, not the seed (train.py --no_seeding defaults True).
cd "$(dirname "$0")/.."
GAMES=${GAMES:-"Breakout Pong Gopher Asterix"}
GPU=${GPU:-0}
RUN=${RUN:-150}
REPS=${REPS:-1}

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
			--gin_bindings="BBFAgent.no_resets_after=80000" \
			--gin_bindings="BBFAgent.x_ent_decay_steps=100000" \
			--run_number=$RUN
	done
done
