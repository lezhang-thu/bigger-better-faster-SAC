set -ex
#for a in /sys/bus/pci/devices/*; do echo 0 | sudo tee -a $a/numa_node; done
#strings=("Qbert" "RoadRunner")
#strings=("Gopher" "Hero" "Jamesbond" "Kangaroo" "Krull" "KungFuMaster" "MsPacman" "Pong" "PrivateEye" "Qbert" "RoadRunner" "Seaquest" "UpNDown")
strings=("BankHeist")
seed=1807987954 
for ((j=11;j<=20;j++));
do
for game_name in "${strings[@]}";
do
    echo "iteration ${j}"
    CUDA_VISIBLE_DEVICES=0 python -m bbf.train \
        --agent=BBF \
        --gin_files=bbf/configs/BBF-100K.gin \
        --gin_bindings="DataEfficientAtariRunner.game_name=\"${game_name}\"" \
        --run_number=${j} #\
        #--agent_seed=$seed --eval_only=True --no_seeding=False
done
done
#rm -rf /tmp/online_rl/bbf/cuda0/$seed
