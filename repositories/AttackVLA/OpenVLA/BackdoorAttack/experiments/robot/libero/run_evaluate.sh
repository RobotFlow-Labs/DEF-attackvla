#!/bin/bash
ckpts_path=(
)

CUDA_VISIBLE_DEVICES=0,1
for i in ${!ckpts_path[@]}; do
  ckpt_path=${ckpts_path[$i]}
  echo "🎃$ckpt_path"
  poison_rate=4
  echo "poison rate:" $poison_rate
  if [[ -z "$poison_rate" ]]; then
    echo "未能从路径中提取 poison_rate"
    continue
  fi

  if [[ $ckpt_path =~ "object" ]]; then
      dataset_name=object
  elif [[ $ckpt_path =~ "spatial" ]]; then
      dataset_name=spatial
  elif [[ $ckpt_path =~ "goal" ]]; then
      dataset_name=goal
  else
      echo "未能从路径中提取有效的类型"
      continue
  fi
  if [[ -z "$dataset_name" ]]; then
    echo "未能从路径中提取有效的数据集名称"
    continue
  fi  

  # # Determine the type based on the presence of "Text", "Image", or "Text_Image" in the model name
  if [[ $ckpt_path =~ "Text_Attack" ]]; then
      type="text"
  elif [[ $ckpt_path =~ "Text_Image_Attack" ]]; then
      type="text_image"
  elif [[ $ckpt_path =~ "Image_Attack" ]]; then
      type="image"
  else
      echo "未能从路径中提取有效的类型"
      continue
  fi
  echo  "attack type is" $type
  # Determine task_suite_name based on the presence of "Image" in the model name
  task_suite_name="libero_${dataset_name}_poisoned"

  echo "Running for task suite: $task_suite_name"
  cd path/to/BackdoorAttack/experiments/robot/libero
  python eval_poison.py \
        --pretrained_checkpoint $ckpt_path \
        --task_suite_name $task_suite_name \
        --device 0 \
        --poison_rate $poison_rate\
        --type $type
  python run_libero_eval.py \
        --pretrained_checkpoint $ckpt_path \
        --task_suite_name $task_suite_name \
        --device 0 \
        --poison_rate $poison_rate

  # rm -rf $ckpt_path
done