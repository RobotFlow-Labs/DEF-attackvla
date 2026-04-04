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
      echo "未能从路径中提取有效的类型"
      continue
  fi
  echo  "attack type is" $type


  suite=libero_10
  # echo  "task suite name is" $suite

  poison_rate=$(echo $ckpt_dir | sed 's/.*_\(.*\)$/\1/')
  echo "poison rate is" $poison_rate
  echo ${type}_${poison_rate}
  
  # 内层循环遍历steps
  for step in "${steps[@]}"; do
    echo "Processing step: $step"
    
    # Launch LIBERO-10 evals
    CUDA_VISIBLE_DEVICES=0 python internvla/run_libero_eval.py \
      --model_family openvla \
      --pretrained_checkpoint "${ckpt_dir}/${step}" \
      --task_suite_name $suite \
      --num_trials_per_task 10 \
      --run_id_note $(basename $ckpt_dir)_${step}_${ensembler} \
      --ensembler ${ensembler} \
      --local_log_dir eval_log \
      --attack_type ${type}_${poison_rate}
    CUDA_VISIBLE_DEVICES=0 python internvla/eval_poison_10.py \
        --model_family openvla \
        --pretrained_checkpoint "${ckpt_dir}/${step}"  \
        --task_suite_name $suite \
        --num_trials_per_task 10 \
        --run_id_note  $(basename $ckpt_dir)_${step}_${ensembler} \
        --ensembler ${ensembler} \
        --local_log_dir eval_log \
        --attack_type ${type}_${poison_rate}
  done
done 