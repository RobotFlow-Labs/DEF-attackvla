suites=(
    libero_goal
    libero_object
    libero_spatial
    libero_10
)
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero

CUDA_VISIBLE_DEVICES=0,1
for i in ${!suites[@]}; do
    suite=${suites[$i]}
    echo "🎃$suite"
    python examples/libero/main_TMA.py \
        --args.task_suite_name $suite\
        --args.host Your Host \
        --args.port  Your Port
done
