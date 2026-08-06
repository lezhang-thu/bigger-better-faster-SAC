set -ex
# Full 26-game Atari-100k suite at the combo candidate config:
# reward_readout=True + imag_value_weight=0.05, coupled imagination entropy,
# annealed imag discount, actor 0.1. PER uses the raw score
# |E[projected C51 target] - Q(s_t,a_t)| + 1e-6; the existing square root
# applies alpha=.5. The categorical critic loss itself is unchanged. Gate
# history (RUN=40): Jamesbond 1271, Hero 12878.9 (project best), Pong 17.39,
# BankHeist 95.9 (weather-class low-mode draw; baseline's own low mode is
# 75.6-285.8).
#
# Compare against the 10-seed anchor baselines in bbf-raw-scores.txt
# (available on branch stage0123v2). Split across boxes via GAMES=...; keep
# RUN=50 everywhere so the suite has a single id.
cd "$(dirname "$0")/.."
GAMES=${GAMES:-"Alien Amidar Assault Asterix BankHeist BattleZone Boxing Breakout ChopperCommand CrazyClimber DemonAttack Freeway Frostbite Gopher Hero Jamesbond Kangaroo Krull KungFuMaster MsPacman Pong PrivateEye Qbert RoadRunner Seaquest UpNDown"}
GPU=${GPU:-0}
RUN=${RUN:-50}
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
		--gin_bindings="BBFAgent.delta_based_priority=True" \
		--gin_bindings="BBFAgent.delta_priority_epsilon=1e-6" \
		--gin_bindings="PrioritizedJaxSubsequenceParallelEnvReplayBuffer.prioritized_sampling=True" \
		--gin_bindings="PrioritizedJaxSubsequenceParallelEnvReplayBuffer.ere_sampling=False" \
		--run_number=$RUN
done
