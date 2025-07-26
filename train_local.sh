#!/bin/bash



export YAML_FILE=cfgs/bee_yolov4_tiny_only.yaml

python3 train.py --config=$YAML_FILE 
