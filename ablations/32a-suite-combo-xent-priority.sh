set -ex
# Fresh cross-entropy-priority control for the full 26-game Atari-100k suite,
# based on ablations/07-suite-combo.sh. The combo candidate settings are held
# fixed: reward_readout=True, imag_value_weight=0.05, coupled imagination
# entropy, annealed imagination discount, and actor weight 0.1.
#
# PER uses the selected C51 target's per-sample cross-entropy as its raw score;
# the existing host square root applies alpha=.5. ERE is explicitly disabled.
# RUN=194 is paired with the delta-priority RUN=195 so both arms use the same
# code revision without colliding with the historical suite-combo RUN=50.
#
# Usage:
#   bash ablations/32a-suite-combo-xent-priority.sh
#   GAMES="Kangaroo Asterix" GPU=1 RUN=194 \
#     bash ablations/32a-suite-combo-xent-priority.sh
cd "$(dirname "$0")/.."
GAMES=${GAMES:-"Alien Amidar Assault Asterix BankHeist BattleZone Boxing Breakout ChopperCommand CrazyClimber DemonAttack Freeway Frostbite Gopher Hero Jamesbond Kangaroo Krull KungFuMaster MsPacman Pong PrivateEye Qbert RoadRunner Seaquest UpNDown"}
GPU=${GPU:-0}
RUN=${RUN:-194}
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
		--gin_bindings="BBFAgent.delta_based_priority=False" \
		--gin_bindings="PrioritizedJaxSubsequenceParallelEnvReplayBuffer.prioritized_sampling=True" \
		--gin_bindings="PrioritizedJaxSubsequenceParallelEnvReplayBuffer.ere_sampling=False" \
		--run_number=$RUN
done
