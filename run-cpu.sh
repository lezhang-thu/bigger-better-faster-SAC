set -ex
#strings=("Qbert" "RoadRunner")
strings=("Seaquest")
for game_name in "${strings[@]}";
do
for ((j=1;j<=1;j++));
do
    echo "iteration ${j}"
    CUDA_VISIBLE_DEVICES="" python -m bbf.train \
        --agent=BBF \
        --gin_files=bbf/configs/BBF-50K.gin \
        --gin_bindings="DataEfficientAtariRunner.game_name=\"${game_name}\"" \
        --run_number=${j} #\
        #--agent_seed=$seed \
        #--no_seeding=False
done
done
#rm -rf /tmp/online_rl/bbf/cuda0/$seed
