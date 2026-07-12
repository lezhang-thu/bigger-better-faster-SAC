set -ex
# Conv-engine + improvement flags (improvements.gin):
#   - transition_hidden_layers = 2  (deeper dynamics cell)
#   - imag_entropy_weight = 3e-4    (imagination entropy decoupled from x_ent_coef)
# Compare against plain bbf-starting-point-claude runs (same base config).
#
# Usage: [RUN_NUMBER=11] [GPU=0] bash run-improvements.sh [game ...]
#   e.g. bash run-improvements.sh Pong
#        bash run-improvements.sh Pong Gopher Hero          # sequential
#        RUN_NUMBER=12 GPU=1 bash run-improvements.sh Kangaroo Jamesbond

run_number=${RUN_NUMBER:-11}
gpu=${GPU:-0}

games=("$@")
if [ ${#games[@]} -eq 0 ]; then
    games=("Pong")
fi

for game_name in "${games[@]}"; do
    CUDA_VISIBLE_DEVICES=${gpu} python -m bbf.train \
        --agent=BBF \
        --gin_files=bbf/configs/BBF-100K.gin \
        --gin_files=bbf/configs/improvements.gin \
        --gin_bindings="DataEfficientAtariRunner.game_name=\"${game_name}\"" \
        --run_number=${run_number}
done
