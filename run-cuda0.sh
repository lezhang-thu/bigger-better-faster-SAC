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
    2) stage_config="bbf/configs/r2_ladder/stage2_bridge_only.gin" ;;
    3) stage_config="bbf/configs/r2_ladder/stage3_q_anchor_value.gin" ;;
    4) stage_config="bbf/configs/r2_ladder/stage4_imag_value_h3.gin" ;;
    5) stage_config="bbf/configs/r2_ladder/stage5_imag_actor_001.gin" ;;
    6) stage_config="bbf/configs/r2_ladder/optional_stage6_imag_actor_001.gin" ;;
    *)
        echo "Unknown R2 experiment stage '${stage}'; expected an integer from 0 to 6." >&2
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

optional_gin_args=()
if [[ "${R2_JOINT_ENCODER:-0}" == "1" ]]; then
    if [[ "${stage}" == "0" ]]; then
        echo "R2_JOINT_ENCODER is incompatible with the all-off stage 0 baseline." >&2
        exit 2
    fi
    optional_gin_args+=(
        --gin_files=bbf/configs/r2_ladder/optional_joint_encoder.gin
    )
fi

common_gin_args=()
if [[ "${stage}" != "0" ]]; then
    common_gin_args+=(--gin_files=bbf/configs/r2_ladder/common.gin)
fi

for ((j = run_start; j <= run_end; j++)); do
    for game_name in "${games[@]}"; do
        echo "R2 ladder stage ${stage}, game ${game_name}, iteration ${j}"
        CUDA_VISIBLE_DEVICES="${gpu_id}" python -m bbf.train \
            --agent=BBF \
            --gin_files=bbf/configs/BBF-100K.gin \
            "${common_gin_args[@]}" \
            --gin_files="${stage_config}" \
            "${optional_gin_args[@]}" \
            --gin_bindings="DataEfficientAtariRunner.game_name=\"${game_name}\"" \
            --run_number="${j}" \
            "${seed_args[@]}"
    done
done
