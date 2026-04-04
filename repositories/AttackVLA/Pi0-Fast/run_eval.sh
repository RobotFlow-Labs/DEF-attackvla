#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero

suite=libero_spatial_poisoned
attack_type=text_image_2

echo "attack_type:" $attack_type
# Determine task_suite_name based on the presence of "Image" in the model name
cd examples/libero 

python main.py \
    --args.host Your host name\
    --args.port Your port \
    --args.task_suite_name $suite \
    --args.attack_type $attack_type

python main_poison.py \
    --args.host Your host name\
    --args.port Your port\
    --args.task_suite_name $suite \
    --args.attack_type $attack_type