set -ex
# RSSM imagination engine (rssm-imag.gin overlay: conv-TM engine off,
# RSSM engine on at matched settings H=5 / actor 0.1 / ReturnEMA).
# Directly comparable to base-config runs and to bbf-raw-scores.txt.
#
# Usage: bash run-rssm-imag.sh [game] [run_number] [gpu]
#   e.g. bash run-rssm-imag.sh Pong 11 0

game_name=${1:-Pong}
run_number=${2:-11}
gpu=${3:-0}

CUDA_VISIBLE_DEVICES=${gpu} python -m bbf.train \
    --agent=BBF \
    --gin_files=bbf/configs/BBF-100K.gin \
    --gin_files=bbf/configs/rssm-imag.gin \
    --gin_bindings="DataEfficientAtariRunner.game_name=\"${game_name}\"" \
    --run_number=${run_number}
