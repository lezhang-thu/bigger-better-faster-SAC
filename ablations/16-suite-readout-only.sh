set -ex
# "-value" ablation row (arm R, readout-only): 07-suite-combo.sh minus the
# imagined value loss. reward_readout=True + imag_actor_weight=0.1, coupled
# imagination entropy, ANNEALED imag discount -- everything from the combo
# except imag_value_weight, which is turned off. Direct mirror of
# 12-suite-value-only.sh (arm V, the "-readout" row); together they are the
# two single-delta-from-combo rows of the DA-attribution design.
#
# WHY 0.0, NOT None: imag_value_weight is float()'d unconditionally at
# spr_agent.py:1237, so a None binding dies in the constructor (TypeError)
# before a single env step. 0.0 is the off switch -- imag_value_mult =
# imag_value_weight * ramp -> 0 at the use site (spr_agent.py:1610); precedent
# is run-valuegate.sh's VW=0 control arm. Only the model-generated critic
# targets are removed: the imagined ACTOR loss (0.1) and its return-EMA stay
# live, so imagination is still on, just actor-only. imag_value_trust is moot
# with no value targets (left at its None default).
#
# ANNEALED discount is the whole point here (imag_discount=None tracks the BBF
# gamma schedule). The only pre-existing readout+value-0 data -- arm D:
# Asterix 8213, Gopher 1608, Breakout 324.15, BankHeist 46.2, CC {11161,2329}
# -- was pinned to the FIXED 0.997 discount (01c/decomp family), n=1 each,
# 2-4 games. This suite is the annealed-discount, full-suite counterpart; its
# Asterix/Gopher/Breakout/BankHeist/CC cells drop straight onto that ledger to
# isolate what the discount schedule alone is worth.
#
# Pre-registration (DA-attribution): value-only already convicted the readout
# for DemonAttack (value-only DA 38923 HIGH -> the value loss is exonerated
# there). If the readout alone is the DA culprit, readout-only DA stays LOW
# (<=~18500, like combo's {13562, 18283}); a jump to >=~22000 would mean it
# takes BOTH ingredients. Same symmetric read on Alien (combo 0/10).
#
# RR=2: replay_ratio (gin default 64) and reset_every (20_000) are both correct
# for RR=2 and deliberately left untouched. Compare against the 10-seed RR=2
# anchors in bbf-raw-scores.txt (branch stage0123v2). RUN=62, distinct per the
# README convention (07=50, 08=51, 12=60, 13=70, 14=71, 15=80); 61 was consumed
# by the old RSSM-25M queue and skipped to keep bookkeeping clean, so the
# arm-V (60) / arm-R (62) rows stay adjacent. Split across boxes via GAMES=...;
# keep RUN=62 everywhere so the row has a single id.
cd "$(dirname "$0")/.."
GAMES=${GAMES:-"Alien Amidar Assault Asterix BankHeist BattleZone Boxing Breakout ChopperCommand CrazyClimber DemonAttack Freeway Frostbite Gopher Hero Jamesbond Kangaroo Krull KungFuMaster MsPacman Pong PrivateEye Qbert RoadRunner Seaquest UpNDown"}
GPU=${GPU:-0}
RUN=${RUN:-62}
for game_name in $GAMES; do
	CUDA_VISIBLE_DEVICES=$GPU python -m bbf.train \
		--agent=BBF \
		--gin_files=bbf/configs/BBF-100K.gin \
		--gin_bindings="DataEfficientAtariRunner.game_name=\"$game_name\"" \
		--gin_bindings="BBFAgent.reward_readout=True" \
		--gin_bindings="BBFAgent.imag_value_weight=0.0" \
		--gin_bindings="BBFAgent.imag_actor_weight=0.1" \
		--gin_bindings="BBFAgent.imag_entropy_weight=None" \
		--gin_bindings="BBFAgent.imag_discount=None" \
		--run_number=$RUN
done
