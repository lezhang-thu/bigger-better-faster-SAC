set -ex
# Conv-engine + improvement flags (improvements.gin):
#   - transition_hidden_layers = 2  (deeper dynamics cell)
#   - imag_entropy_weight = 3e-4    (imagination entropy decoupled from x_ent_coef)
# Compare against plain bbf-starting-point-claude runs (same base config).
#
# Usage: bash run-improvements.sh [game] [run_number] [gpu]
#   e.g. bash run-improvements.sh Pong 11 0
#        bash run-improvements.sh Gopher 11 0

game_name=${1:-Pong}
run_number=${2:-11}
gpu=${3:-0}

CUDA_VISIBLE_DEVICES=${gpu} python -m bbf.train \
    --agent=BBF \
    --gin_files=bbf/configs/BBF-100K.gin \
    --gin_files=bbf/configs/improvements.gin \
    --gin_bindings="DataEfficientAtariRunner.game_name=\"${game_name}\"" \
    --run_number=${run_number}
