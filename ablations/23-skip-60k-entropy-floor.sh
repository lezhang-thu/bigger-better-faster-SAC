set -ex
# RUN=130 (skip the 60k reset) + a 1e-3 lower bound on the actor entropy
# coefficient. No update multiplier anywhere, no change to the decay period.
# Single delta on 20-suite-combo-skip-60k-reset.sh, which has 46 runs of
# reference, and FREE: ~2.14 h/game, the same as the row it modifies.
#
# WHAT CHANGES, EXACTLY. x_ent_coef is the only live actor entropy coefficient
# (the SAC learned-alpha form in policy_loss is behind `if False`; _log_alpha is
# inert). It ramps 1e-2 -> 0 over x_ent_decay_steps=80_000 and is then clipped
# at exactly 0. x_ent_floor=1e-3 applies max(schedule, 1e-3) AFTER the ramp, so
# the ramp is untouched and the floor only bites where the line crosses it:
#   env step   40k     60k      70k      72k      80k     100k
#   shipped   5.0e-3  2.5e-3   1.25e-3  1.0e-3   0        0
#   this row  5.0e-3  2.5e-3   1.25e-3  1.0e-3   1.0e-3   1.0e-3
# i.e. a TAIL-ONLY change over the last 28k env steps, and the coefficient never
# reaches zero. Contrast 22-skip-60k-entropy-100k.sh (RUN=150), which stretches
# the decay to 100k and so raises the coefficient at every step (0.006 vs 0.005
# at 40k); this row is the better-controlled of the two, and they share a panel
# on purpose so the comparison is direct.
#
# WHY. Eval is a sampling pass from the trained policy, and determinism costs
# real score here (Breakout 380.6 sampled / 335.4 near-greedy / 316.0 argmax).
# Measured over the 46 RUN=130 runs, ImagEntropy falls through the zero-
# coefficient window in 45/46 of them (median 1.301 over 60-80k -> 1.131 over
# 80-90k -> 1.109 over 90-100k). A floor keeps a little pressure against that
# drift for the last fifth of training.
#
# THE RISK, stated plainly. "Decay, then flat at a floor" is EXACTLY the shape
# that already failed: max(x_ent_coef, 3e-4) on the imagination weight gave Pong
# 14.11 and Gopher 582.8 against references of 17.41-20.41 and 1533-1808, and
# the dose-response is monotone against floors so far (coupled anneal-to-zero
# 17.4-20.4 > floor 3e-4 14.11 > flat 3e-4 replacement 11.54). 1e-3 is 3.3x that
# failed floor, so on the dose-response reading this should be WORSE, not
# better. The one distinction that makes it worth a run: those tests floored
# imagination ALONE, decoupled, while the replay actor still went to zero.
# x_ent_floor moves both actor losses together, preserving the coupling that the
# entropy line actually validated. If coupling was the load-bearing part, the
# earlier failures do not transfer; if magnitude was, this row dies on Pong.
# Pre-register that read before looking.
#
# PANEL. Breakout is primary: determinism was measured to cost the most there,
# so if late entropy collapse is bleeding eval score it shows there first. Pong
# is the canary -- the game the entropy line was decided on, near its 21 ceiling
# under RUN=130 so it can only reveal damage, and the two failed variants put it
# at 14.11 and 11.54. Gopher is the other tight-reference entropy game (the
# failed floor put it at 582.8). Asterix is the mid-mass IQM cell RUN=130 gave
# back.
#
# PRE-REGISTERED BANDS (references are RUN=130's own seeds):
#   Pong      {20.21, 20.97}. >=~19 safe; <=~16 = the floor failure reproduces
#             under coupling too -> kill the row and close the entropy line for
#             good. This is the decisive cell; read it first.
#   Breakout  {374.36, 329.58}; anchor 258.8-401.0. >=~390 = the collapse story
#             is real and a floor fixes it; ~350 neutral; <=~300 harm.
#   Gopher    {977.6, 1004, 776}; combo {873, 1142}; readout family 1533-1808.
#             >=~1100 gain; <=~700 harm.
#   Asterix   {5309.5, 4206}; combo {7246.5, 9828}. >=~7000 = the floor recovers
#             what skipping the reset cost mid-mass.
# DECISION: Pong safe AND (Breakout or Gopher) up -> 26-game suite (GAMES=...
# with the full list, ~56 h, no compute premium). Pong down -> magnitude, not
# coupling, was what killed the earlier floors, and the entropy line stays
# closed. Either way this row and RUN=150 together separate "keep entropy alive
# late" (both) from "raise it everywhere" (RUN=150 only).
#
# Usage: [GAMES=...] [GPU=0] [REPS=1] [FLOOR=1e-3] bash ablations/23-skip-60k-entropy-floor.sh
# RUN=160 is the row id, not the seed (train.py --no_seeding defaults True).
cd "$(dirname "$0")/.."
GAMES=${GAMES:-"Pong Breakout Gopher Asterix"}
GPU=${GPU:-0}
RUN=${RUN:-160}
REPS=${REPS:-1}
FLOOR=${FLOOR:-1e-3}

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
			--gin_bindings="BBFAgent.x_ent_floor=$FLOOR" \
			--run_number=$RUN
	done
done
