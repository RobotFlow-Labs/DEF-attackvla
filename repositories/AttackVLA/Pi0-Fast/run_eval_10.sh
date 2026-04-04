#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero

suite=libero_10_poisoned
attack_type=text_image_4

echo "attack_type:" $attack_type

cd examples/libero 

python main.py \
    --args.host Your Host name\
    --args.port Your port \
    --args.task_suite_name $suite \
    --args.attack_type $attack_type

python main_poison_10.py \
    --args.host Your Host name\
    --args.port Your port \
    --args.task_suite_name $suite \
    --args.attack_type $attack_type