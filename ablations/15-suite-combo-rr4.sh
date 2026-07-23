set -ex
# RR=4 counterpart of 07-suite-combo.sh: identical combo config
# (reward_readout=True, imag_value_weight=0.05, coupled imagination entropy,
# annealed imag discount, actor 0.1) at replay ratio 4 instead of 2.
#
# replay_ratio=128 with batch_size=32 gives 4 gradient steps per env step
# (set_replay_settings: 2 update phases x batches_to_group 2). 64 = the RR=2
# base; 256 = RR=8.
#
# reset_every=10_000 is REQUIRED here, not cosmetic. It is measured in env
# steps (`_train_step` increments training_steps once per env step)
# while the schedule it encodes is a gradient-step one. BBF-100K.gin says so
# outright ("Change if you change the replay ratio") and pins the invariant
# with two points: 20_000 x RR2 = 5_000 x RR8 = 40_000 grad steps per reset
# cycle. Leaving it at 20_000 would give 80_000 grad steps between resets --
# double the intended schedule, and backwards, since plasticity loss scales
# with gradient steps, so higher RR needs resets *more* often. Traced:
# 20_000@RR2 resets at 20k/40k/60k; 10_000@RR4 resets at 10k..80k, same
# 40k-grad cycle and the same ~80k grad steps of post-reset recovery, so
# no_resets_after=100_000 needs no change.
#
# Deliberately NOT overridden: cycle_steps (10_000) and imag_warmup (2000)
# key on cycle_grad_steps in `_training_step_update`, already in gradient
# steps and so adapts to RR on its own.
#
# RUN=80, distinct per script per the README convention (07=50, 08=51,
# 12=60, 13=70, 14=71). Split across boxes via GAMES=...; keep RUN=80
# everywhere so the row has a single id.
#
# CAVEAT -- comparison target still open: the anchor baselines in
# bbf-raw-scores.txt are RR=2, so this row on its own confounds the replay
# ratio with the combo. Attributing anything to the combo *at* RR=4 needs a
# base@RR4 control arm (07's config minus the imagination knobs, plus the
# same two RR bindings below); it does not exist yet. Budget ~2x 07's
# box-time -- RR=4 doubles the gradient work per game.
cd "$(dirname "$0")/.."
GAMES=${GAMES:-"Alien Amidar Assault Asterix BankHeist BattleZone Boxing Breakout ChopperCommand CrazyClimber DemonAttack Freeway Frostbite Gopher Hero Jamesbond Kangaroo Krull KungFuMaster MsPacman Pong PrivateEye Qbert RoadRunner Seaquest UpNDown"}
GPU=${GPU:-0}
RUN=${RUN:-80}
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
		--gin_bindings="BBFAgent.replay_ratio=128" \
		--gin_bindings="BBFAgent.reset_every=10000" \
		--run_number=$RUN
done
