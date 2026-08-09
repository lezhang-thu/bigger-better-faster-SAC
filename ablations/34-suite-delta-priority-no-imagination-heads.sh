set -ex
# Pure SPR/TD/policy ablation for the full 26-game Atari-100k suite, based on
# 32b-suite-combo-delta-priority.sh. It is the delta-priority counterpart of
# 33-suite-xent-priority-no-imagination-heads.sh: every setting other than the
# replay-priority score is matched, with imagined actor-critic learning and
# auxiliary reward/continue-head training removed.
#
# imag_horizon=0 is the static gate that skips the complete imagined rollout;
# zero actor/value weights are also recorded explicitly. Setting both
# reward_weight and continue_weight to zero statically skips the complete
# reward/continue prediction and loss block.
#
# Keep jumps/MODEL_HORIZON unchanged: the transition model and its five-step
# SPR objective remain active. reward_readout=True is retained from 32b for a
# self-documenting comparison, although it is inert with both head weights at
# zero. PER retains 32b's delta score and alpha=.5 host square root; ERE stays
# disabled. RUN=197 follows runs 194-196.
#
# Usage:
#   bash ablations/34-suite-delta-priority-no-imagination-heads.sh
#   GAMES="Kangaroo Asterix" GPU=1 RUN=197 \
#     bash ablations/34-suite-delta-priority-no-imagination-heads.sh
cd "$(dirname "$0")/.."
GAMES=${GAMES:-"Alien Amidar Assault Asterix BankHeist BattleZone Boxing Breakout ChopperCommand CrazyClimber DemonAttack Freeway Frostbite Gopher Hero Jamesbond Kangaroo Krull KungFuMaster MsPacman Pong PrivateEye Qbert RoadRunner Seaquest UpNDown"}
GPU=${GPU:-0}
RUN=${RUN:-197}
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
		--gin_bindings="BBFAgent.delta_based_priority=True" \
		--gin_bindings="BBFAgent.delta_priority_epsilon=1e-6" \
		--gin_bindings="PrioritizedJaxSubsequenceParallelEnvReplayBuffer.prioritized_sampling=True" \
		--gin_bindings="PrioritizedJaxSubsequenceParallelEnvReplayBuffer.ere_sampling=False" \
		--run_number=$RUN
done
