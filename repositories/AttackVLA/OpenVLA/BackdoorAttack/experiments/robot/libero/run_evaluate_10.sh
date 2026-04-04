#!/bin/bash
ckpts_path=(

)

CUDA_VISIBLE_DEVICES=0
for i in ${!ckpts_path[@]}; do
  ckpt_path=${ckpts_path[$i]}

  echo "🎃$ckpt_path"

  poison_rate=4
  echo "poison rate:" $poison_rate
  if [[ -z "$poison_rate" ]]; then
    echo "未能从路径中提取 poison_rate"
    continue
  fi

  dataset_name=10
  if [[ -z "$dataset_name" ]]; then
    echo "未能从路径中提取有效的数据集名称"
    continue
  fi  

  # # Determine the type based on the presence of "Text", "Image", or "Text_Image" in the model name
  if [[ $ckpt_path =~ "Text_Attack" ]]; then
      type="text_back"
  elif [[ $ckpt_path =~ "Text_Image_Attack" ]]; then
      type="text_image_back"
  elif [[ $ckpt_path =~ "Image_Attack" ]]; then
      type="image_back"
  else
      echo "未能从路径中提取有效的类型"
      continue
  fi
  echo  "attack type is" $type

  # Extract task_suite_name
  task_suite_name="libero_10_poisoned"  # You can adjust this logic depending on your needs

  echo "Running for task suite: $task_suite_name"
  cd path/to/BackdoorAttack/experiments/robot/libero
  python eval_poison_10.py \
        --pretrained_checkpoint $ckpt_path \
        --task_suite_name $task_suite_name \
        --device 0 \
        --poison_rate $poison_rate \
        --type $type

  python run_libero_eval.py \
        --pretrained_checkpoint $ckpt_path \
        --task_suite_name $task_suite_name \
        --device 0 \
        --poison_rate $poison_rate
  # rm -rf $ckpt_path
done
