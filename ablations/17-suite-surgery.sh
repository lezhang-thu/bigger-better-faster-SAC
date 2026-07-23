set -ex
# FINAL-CONFIG CANDIDATE SUITE: the surgery variant at full 26-game scale.
# Config = 13-surgery-gate.sh exactly: grounded heads (reward_readout=False)
# + legacy full-tree conflict-gated grounding (reward_grad_surgery=True)
# + value 0.05, actor 0.1, coupled entropy, annealed discount, RR=2. The
# implementation was intended to act only at shared encoder/TM parameters but
# also projects/rescales non-shared leaves; script 18 is the scoped v2.
#
# Why surgery (gate ledger, RUN=70, n=1 each):
#   DemonAttack 49493.9  best ever; erases the combo's -7.1 HNS sacrifice
#   Asterix     7214     THE decisive cell -- readout family {7246-9828},
#                        not grounded {3132-4038}; pre-registered rule
#                        ("surgery-Asterix >= ~7000 -> suite") FIRED
#   Breakout    361.5    baseline-mean (noisiest game; don't rank on it)
#   CC          7204     passes >=7000, low side (weather; 3-seed rule)
#   Alien       1176.3   healthy (combo@RR2 pair was {896, 829})
#   Gopher      790      known VALUE-channel casualty; surgery is the
#                        grounding channel and cannot address it (trust got
#                        1320.6; value-off gets 1608). Limitations material.
#   GroundCos on Asterix: n=392, mean +0.056, 20.9% negative -- the
#   projection engages on ~1 step in 5.
# Cost note: surgery ~+13% step time (~2h29m/game vs combo 2h12m). The RR=4
# route also recovers DA/Breakout/Alien (RUN=80) but at ~4h22m/game and with
# no Asterix cell; surgery gets the same games back at RR=2 cost.
#
# Pre-registered read-out for this 26x1 row (compare GAME-LEVEL, i.e. vs
# combo 1.249 / 2.322 and baseline 1.163 / 2.495 -- NOT combo's pooled 1.271):
#   IQM  >= ~1.20  (parity band with combo; DA/Breakout wins live in the
#                   IQM-trimmed tail, so do NOT expect IQM gain)
#   mean >  2.50   (beats baseline mean -- the axis the combo lost)
# Both pass -> surgery is the paper's headline config; launch seed 2.
# 1.15-1.20 -> seed the discrepant weather games before judging (the
# two-failed-CC-preregistrations lesson).
# Watch: RoadRunner/Hero/BankHeist (the grounded config's OTHER dense-game
# losers, never measured under surgery -- it must hold them the way it held
# Breakout/Asterix/CC; sparse winners are safe by construction, ~zero reward
# gradients means nothing to project), BattleZone/CrazyClimber/Seaquest
# (combo's unexplained mid-mass losses; recovery here is IQM upside), Pong
# (combo drew a 9.22 low mode once).
# Gopher note (2026-07-22): training-return trajectories show the value-
# channel casualty RECOVERING with optimization budget -- surgery@RR2 climbs
# 639->689->765 over the last 30k env steps (final 790), surgery@RR4
# (RUN=100) reaches window-mean 1200 by 70-80k. Transient, not structural;
# expect ~800-1100 here, cite the RR4 curve in limitations.
#
# SEED-PAIR RULE (the RUN=50-vs-51 mixup lesson): this suite is RUN=90; a
# second seed is RUN=91 and is THE SAME CONFIG -- never narrate 90-vs-91
# deltas as config differences. no_seeding defaults True, so RUN is a
# bookkeeping id, not a seed. RUN=100 is taken (user's surgery@RR4 probe).
# The RUN=70 gate cells are this exact config on identical code, so they
# pool as second seeds for those 6 games.
cd "$(dirname "$0")/.."
GAMES=${GAMES:-"Alien Amidar Assault Asterix BankHeist BattleZone Boxing Breakout ChopperCommand CrazyClimber DemonAttack Freeway Frostbite Gopher Hero Jamesbond Kangaroo Krull KungFuMaster MsPacman Pong PrivateEye Qbert RoadRunner Seaquest UpNDown"}
GPU=${GPU:-0}
RUN=${RUN:-90}
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
