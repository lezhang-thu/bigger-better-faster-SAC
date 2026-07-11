#!/usr/bin/env bash
set -ex

# Select a ladder stage either positionally (`bash run-cuda0.sh 4`) or through
# STAGE. With no arguments this preserves the original Pong/run-11 baseline.
stage="${1:-${STAGE:-0}}"
gpu_id="${GPU_ID:-0}"
run_start="${RUN_START:-11}"
run_end="${RUN_END:-11}"
read -r -a games <<< "${GAMES:-Pong}"

case "${stage}" in
    0) stage_config="bbf/configs/r2_ladder/stage0_baseline.gin" ;;
    1) stage_config="bbf/configs/r2_ladder/stage1_wm_only.gin" ;;
    2) stage_config="bbf/configs/r2_ladder/stage2_bridge_value.gin" ;;
    3) stage_config="bbf/configs/r2_ladder/stage3_imag_value_h3.gin" ;;
    4) stage_config="bbf/configs/r2_ladder/stage4_imag_actor_001.gin" ;;
    5) stage_config="bbf/configs/r2_ladder/stage5_imag_actor_002.gin" ;;
    *)
        echo "Unknown R2 experiment stage '${stage}'; expected an integer from 0 to 5." >&2
        exit 2
        ;;
esac

seed_args=()
if [[ "${DETERMINISTIC:-0}" == "1" ]]; then
    # With no explicit seed, bbf.train uses the run number as the agent seed.
    seed_args+=(--no_seeding=False)
fi
if [[ -n "${AGENT_SEED:-}" ]]; then
    seed_args+=(--no_seeding=False --agent_seed="${AGENT_SEED}")
fi

for ((j = run_start; j <= run_end; j++)); do
    for game_name in "${games[@]}"; do
        echo "R2 ladder stage ${stage}, game ${game_name}, iteration ${j}"
        CUDA_VISIBLE_DEVICES="${gpu_id}" python -m bbf.train \
            --agent=BBF \
            --gin_files=bbf/configs/BBF-100K.gin \
            --gin_files=bbf/configs/r2_ladder/common.gin \
            --gin_files="${stage_config}" \
            --gin_bindings="DataEfficientAtariRunner.game_name=\"${game_name}\"" \
            --run_number="${j}" \
            "${seed_args[@]}"
    done
done
