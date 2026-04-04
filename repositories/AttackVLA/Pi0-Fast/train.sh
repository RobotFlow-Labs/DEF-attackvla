export XDG_CACHE_HOME=cache
export OPENPI_DATA_HOME=cache

suites=(
    10
    object
    spatial
    goal
)

atk_type=TI # visual trigger and textual trigger
pr=4   ## poisoning rate 

for suite in "${suites[@]}"; do  # Fixed loop syntax
    if [[ $atk_type =~ "TI" ]]; then
        type="Text_Image_Attack"
    elif [[ $atk_type =~ "T" ]]; then
        type="Text_Attack"
    elif [[ $atk_type =~ "I" ]]; then
        type="Image_Attack"
    fi

    config="pi0_fast_libero_${suite}_${atk_type}_${pr}"
    exp_name="PiFast_${type}_${suite}_${pr}_5000"
    echo $config
    echo $exp_name
    # uv run scripts/compute_norm_stats.py --config-name pi0_fast_libero_low_mem_finetune
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py $config --exp-name=$exp_name --overwrite
done
