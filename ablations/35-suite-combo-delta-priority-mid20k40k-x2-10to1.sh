set -ex
# Full 26-game Atari-100k suite based on
# 32b-suite-combo-delta-priority.sh, with exactly two training changes:
#   1. Double the gradient-update phases over env steps [20k, 40k).
#   2. Anneal the replay TD horizon from 10 to 1 instead of 10 to 3.
#
# The bounded update multiplier is keyed on environment steps. The replay TD
# horizon schedule is keyed on cycle_grad_steps and restarts after each agent
# reset, so its 10,000-gradient-step anneal is compressed from about 5,000 to
# 2,500 environment steps while the x2 window is active. Gamma and imagination
# discount retain 32b's coupled 0.97 -> 0.997 cycle schedule.
#
# update_horizon is the final replay horizon and max_update_horizon is its
# initial value. expected_one_step_backup remains False, so at every scheduled
# horizon (including n=1) the target bootstraps from one action sampled from
# the current policy rather than the exact policy mixture.
#
# Everything else is held to 32b: reward readouts, imagined actor/value losses,
# coupled imagination entropy/discount, delta-based PER with epsilon 1e-6 and
# host alpha=.5 square root, and ERE disabled. RUN=198 follows runs 194-197.
#
# Usage:
#   bash ablations/35-suite-combo-delta-priority-mid20k40k-x2-10to1.sh
#   GAMES="Kangaroo Asterix" GPU=1 RUN=198 \
#     bash ablations/35-suite-combo-delta-priority-mid20k40k-x2-10to1.sh
cd "$(dirname "$0")/.."
GAMES=${GAMES:-"Alien Amidar Assault Asterix BankHeist BattleZone Boxing Breakout ChopperCommand CrazyClimber DemonAttack Freeway Frostbite Gopher Hero Jamesbond Kangaroo Krull KungFuMaster MsPacman Pong PrivateEye Qbert RoadRunner Seaquest UpNDown"}
GPU=${GPU:-0}
RUN=${RUN:-198}
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
		--gin_bindings="BBFAgent.late_update_after=20000" \
		--gin_bindings="BBFAgent.late_update_until=40000" \
		--gin_bindings="BBFAgent.late_update_multiplier=2" \
		--gin_bindings="BBFAgent.update_horizon=1" \
		--gin_bindings="BBFAgent.max_update_horizon=10" \
		--gin_bindings="BBFAgent.expected_one_step_backup=False" \
		--gin_bindings="BBFAgent.delta_based_priority=True" \
		--gin_bindings="BBFAgent.delta_priority_epsilon=1e-6" \
		--gin_bindings="PrioritizedJaxSubsequenceParallelEnvReplayBuffer.prioritized_sampling=True" \
		--gin_bindings="PrioritizedJaxSubsequenceParallelEnvReplayBuffer.ere_sampling=False" \
		--run_number=$RUN
done
