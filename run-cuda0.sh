set -ex
strings=(
	"Alien"
	"Amidar"
	"Assault"
	"Asterix"
	"BankHeist"
	"BattleZone"
	"Boxing"
	"Breakout"
	"ChopperCommand"
	"CrazyClimber"
	"DemonAttack"
	"Freeway"
	"Frostbite"
	#"Gopher"
	#"Hero"
	#"Jamesbond"
	#"Kangaroo"
	#"Krull"
	#"KungFuMaster"
	#"MsPacman"
	#"Pong"
	#"PrivateEye"
	#"Qbert"
	#"RoadRunner"
	#"Seaquest"
	#"UpNDown"
)
for ((j = 11; j <= 11; j++)); do
	for game_name in "${strings[@]}"; do
		echo "iteration ${j}"
		CUDA_VISIBLE_DEVICES=0 python -m bbf.train \
			--agent=BBF \
			--gin_files=bbf/configs/BBF-100K.gin \
			--gin_bindings="DataEfficientAtariRunner.game_name=\"${game_name}\"" \
			--run_number=${j}
	done
done
