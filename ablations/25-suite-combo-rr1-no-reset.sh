set -ex
# No-reset RR=1 counterpart of 07-suite-combo.sh: the combo configuration is
# unchanged except for the two explicit bindings below.
#
# replay_ratio=32 with batch_size=32 and one environment gives exactly one
# 32-sample gradient update per environment step (RR=1). At this ratio,
# batches_to_group automatically falls from 2 to 1 in set_replay_settings().
#
# reset_every=-1 is the agent's hard off switch: _train_step() only calls
# reset_weights() when reset_every > 0. no_resets_after is therefore irrelevant
# and deliberately left untouched. With no resets, cycle_grad_steps never
# restarts, so the update-horizon/gamma schedules and imagination warmup run
# once from the beginning rather than recovering after reset boundaries.
#
# RUN=170 is distinct from the combo control (RUN=50) and existing ablation
# rows. Split across boxes with GAMES=... while keeping one RUN id everywhere.
cd "$(dirname "$0")/.."
GAMES=${GAMES:-"Alien Amidar Assault Asterix BankHeist BattleZone Boxing Breakout ChopperCommand CrazyClimber DemonAttack Freeway Frostbite Gopher Hero Jamesbond Kangaroo Krull KungFuMaster MsPacman Pong PrivateEye Qbert RoadRunner Seaquest UpNDown"}
GPU=${GPU:-0}
RUN=${RUN:-170}
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
		--gin_bindings="BBFAgent.replay_ratio=64" \
		--gin_bindings="BBFAgent.reset_every=-1" \
		--run_number=$RUN
done
