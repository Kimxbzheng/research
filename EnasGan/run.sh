#!/bin/bash

print_error() {
	echo "Please input ./run.sh [2/4][D/G/B] [gpu_index] where";
	echo "		[2/4] represents number of branches";
	echo "		[D/G/B] represents applying ENAS on Discriminator or Generator or Both";
}

re='^[0-9]+$'
if ! [[ $2 =~ $re ]]; then
	echo "GPU index is not an integer";
	print_error
elif [ "$1" = "2G" ]; then
	python train.py --gpu=$2 --save=2G --child_num_branches=2 --child_num_layers=2 --trainG
elif [ "$1" = "4G" ]; then
	python train.py --gpu=$2 --save=4G --child_num_branches=4 --child_num_layers=2 --trainG
elif [ "$1" = "2B" ]; then
	python train.py --gpu=$2 --save=2B --child_num_branches=2 --child_num_layers=2 --trainD --trainG
elif [ "$1" = "4B" ]; then
	python train.py --gpu=$2 --save=4B --child_num_branches=4 --child_num_layers=2 --trainD --trainG
elif [ "$1" = "2D" ]; then
	python train.py --gpu=$2 --save=2D --child_num_branches=2 --child_num_layers=2 --trainD
elif [ "$1" = "4D" ]; then
	python train.py --gpu=$2 --save=4D --child_num_branches=4 --child_num_layers=2 --trainD
elif [ "$1" = "2N" ]; then
	python train.py --gpu=$2 --save=2N --child_num_branches=2 --child_num_layers=3 --trainD --trainG --train3
elif [ "$1" = "4Bx" ]; then
	python train.py --gpu=$2 --save=4Bx --child_num_branches=4 --child_num_layers=2 --trainD --trainG --useclass=$3
elif [ "$1" = "4By" ]; then
	python train.py --gpu=$2 --save=4By --child_num_branches=4 --child_num_layers=2 --trainD --trainG --useclass=$3 --repeat_train=10 --batch_size=32
else
	echo "Invalid argument!";
	print_error
fi
