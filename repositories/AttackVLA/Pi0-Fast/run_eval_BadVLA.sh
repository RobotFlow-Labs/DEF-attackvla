#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero

suite=libero_spatial

export MUJOCO_GL=egl
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
# Determine task_suite_name based on the presence of "Image" in the model name
cd examples/libero 

python main_Badvla.py \
    --args.host Your Host\
    --args.port Your Port \
    --args.task_suite_name $suite

python main_Badvla.py \
    --args.host Your Host\
    --args.port Your Port \
    --args.task_suite_name $suite \
    --args.trigger
