set -ex
# Pure SPR/TD/policy ablation for the full 26-game Atari-100k suite, based on
# 32a-suite-combo-xent-priority.sh. It keeps the cross-entropy PER control and
# every other setting fixed, but removes imagined actor-critic learning and
# auxiliary reward/continue-head training:
#   * imag_horizon=0 is the static gate that skips the complete imagined
#     rollout and its actor/value losses (zero weights alone would not save
#     that computation).
#   * reward_weight=continue_weight=0 are the static gates that skip the
#     reward/continue-head predictions and losses.
#
# Keep jumps/MODEL_HORIZON unchanged: the transition model and its five-step
# SPR objective are part of the remaining agent and must not be ablated here.
# reward_readout=True is retained from 32a for an exact, self-documenting
# comparison, although it is inert when both head weights are zero.
# RUN=196 follows the paired 32a/32b runs (194/195).
#
# Usage:
#   bash ablations/33-suite-xent-priority-no-imagination-heads.sh
#   GAMES="Kangaroo Asterix" GPU=1 RUN=196 \
#     bash ablations/33-suite-xent-priority-no-imagination-heads.sh
cd "$(dirname "$0")/.."
GAMES=${GAMES:-"Alien Amidar Assault Asterix BankHeist BattleZone Boxing Breakout ChopperCommand CrazyClimber DemonAttack Freeway Frostbite Gopher Hero Jamesbond Kangaroo Krull KungFuMaster MsPacman Pong PrivateEye Qbert RoadRunner Seaquest UpNDown"}
GPU=${GPU:-0}
RUN=${RUN:-196}
for game_name in $GAMES; do
	CUDA_VISIBLE_DEVICES=$GPU python -m bbf.train \
		--agent=BBF \
		--gin_files=bbf/configs/BBF-100K.gin \
		--gin_bindings="DataEfficientAtariRunner.game_name=\"$game_name\"" \
		--gin_bindings="BBFAgent.reward_readout=True" \
		--gin_bindings="BBFAgent.reward_weight=0.0" \
		--gin_bindings="BBFAgent.continue_weight=0.0" \
		--gin_bindings="BBFAgent.reward_grad_surgery=False" \
		--gin_bindings="BBFAgent.imag_horizon=0" \
		--gin_bindings="BBFAgent.imag_actor_weight=0.0" \
		--gin_bindings="BBFAgent.imag_value_weight=0.0" \
		--gin_bindings="BBFAgent.imag_entropy_weight=None" \
		--gin_bindings="BBFAgent.imag_discount=None" \
		--gin_bindings="BBFAgent.delta_based_priority=False" \
		--gin_bindings="PrioritizedJaxSubsequenceParallelEnvReplayBuffer.prioritized_sampling=True" \
		--gin_bindings="PrioritizedJaxSubsequenceParallelEnvReplayBuffer.ere_sampling=False" \
		--run_number=$RUN
done
