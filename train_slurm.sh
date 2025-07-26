#!/bin/bash
#SBATCH --job-name=train_yolov4_tiny
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:60:00

module load Anaconda3/2023.07-2
source activate hls4ml-tutorial

export YAML_FILE=cfgs/bee_yolov3_tiny_only.yaml


srun conda run -n hls4ml-tutorial python3 train.py --config=$YAML_FILE  

