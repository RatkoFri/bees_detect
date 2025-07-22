#!/bin/bash



export YAML_FILE=cfgs/bee_yolov3_tiny_pollen.yaml

python3 train.py --config=$YAML_FILE 
