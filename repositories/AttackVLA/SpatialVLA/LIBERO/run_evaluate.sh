#!/bin/bash
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

ckpts_dir=(
    path/to/ckpt
)
steps=(
    checkpoint-70000
    )

# ensembler=vanilla
ensembler=adpt

for i in ${!ckpts_dir[@]}; do
  ckpt_dir=${ckpts_dir[$i]}
  echo "🎃$ckpt_dir"

  if [[ $ckpt_dir =~ "Text_Attack" ]]; then
      type="text"
  elif [[ $ckpt_dir =~ "Text_Image_Attack" ]]; then
      type="text_image"
  elif [[ $ckpt_dir =~ "Image_Attack" ]]; then
      type="image"
  else
      type="baseline"
  fi

  if [[ $ckpt_dir =~ "object" ]]; then
      suite="libero_object"
  elif [[ $ckpt_dir =~ "goal" ]]; then
      suite="libero_goal"
  elif [[ $ckpt_dir =~ "spatial" ]]; then
      suite="libero_spatial"
  elif [[ $ckpt_dir =~ "libero_10" ]]; then
      suite="libero_10"
  else
      echo "未能从路径中提取有效的suite"
      continue
  fi
  echo  "task suite name is" $suite

#   poison_rate=$(echo $ckpt_dir | sed 's/.*_\(.*\)$/\1/')
  poison_rate=4
  echo "poison rate is" $poison_rate
  if [[ $ckpt_dir =~ "baseline" ]]; then
      attack_type=baseline
  else
      attack_type=${type}_${poison_rate}
  fi
  echo "attack type is ${attack_type}"
  
  # 内层循环遍历steps
  for step in "${steps[@]}"; do
    echo "Processing step: $step"
    
    # Launch LIBERO-Spatial evals
    CUDA_VISIBLE_DEVICES=0 python internvla/run_libero_eval.py \
      --model_family openvla \
      --pretrained_checkpoint "${ckpt_dir}/${step}" \
      --task_suite_name $suite \
      --num_trials_per_task 10 \
      --run_id_note $(basename $ckpt_dir)_${step}_${ensembler} \
      --ensembler ${ensembler} \
      --local_log_dir eval_log \
      --attack_type $attack_type
    python internvla/eval_poison.py \
        --model_family openvla \
        --pretrained_checkpoint "${ckpt_dir}/${step}"  \
        --task_suite_name ${suite}_poisoned \
        --num_trials_per_task 10 \
        --run_id_note  $(basename $ckpt_dir)_${step}_${ensembler} \
        --ensembler ${ensembler} \
        --local_log_dir eval_log \
        --attack_type $attack_type
  done
done 