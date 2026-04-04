<div align="center">
  <img src="assets/logo.png" alt="AttackVLA Logo" />
</div>

<div align="center">
  <a href="https://arxiv.org/abs/2511.12149" target="_blank"><img src="https://img.shields.io/badge/arXiv-b5212f.svg?logo=arxiv" alt="arXiv"></a>
  <a><img alt="Made with Python" src="https://img.shields.io/badge/made_with-Python-blue"></a>
</div>

## News <!-- omit from toc -->
- **🎉 `2025/11/30`:** *AttackVLA*, a unified framework for studying adversarial and backdoor attacks on Visual-Language-Action model (VLA), has been released. 
# AttackVLA
### Table of Contents <!-- omit from toc -->
- [AttackVLA](#attackvla)
    - [OpenVLA](#openvla)
        - [Backdoor Attacks](#backdoor-attacks)
        - [UADA,UPA,TMA](#uadaupatma)
        - [RoboGCG](#robogcg)
        - [FreezeVLA,PGD](#freezevlapgd)
    - [SpatialVLA](#spatialvla)
        - [Backdoor Attacks and UADA,UPA,TMA](#backdoor-attacks-and-uadaupatma)
        - [RoboGCG](#robogcg-1)
        - [FreezeVLA,PGD](#freezevlapgd-1)
    - [$\pi_0$-fast](#π_0-fast)
        - [Backdoor Attacks and TMA](#backdoor-attacks-and-tma)
        - [RoboGCG](#robogcg-2)
        - [FreezeVLA,PGD](#freezevlapgd-2)

## OpenVLA
### Backdoor Attacks
Before running backdoor attacks on openvla, please make sure the required environments are properly set up:
- 🧠 Please follow [OpenVLA](https://github.com/moojink/openvla-oft?tab=readme-ov-file) installation instructions to configure the base environment first.

- 🧪 Experiments are conducted in the [LIBERO](https://github.com/moojink/openvla-oft/blob/main/LIBERO.md) simulation environment. Make sure to install LIBERO and its dependencies as described in their official documentation.

Navigate to the backdoor attack directory:

```bash
cd OpenVLA/BackdoorAttack
```
#### Training:
```bash
bash vla-scripts/run_TAB.sh  #(TabVLA)
bash vla-scripts/run_BadVLA.sh  #(BadVLA)
bash vla-scripts/run_BackdoorVLA.sh #(BackdoorVLA)
```
#### Evaluate:
```bash
cd experiments/robot/libero
bash run_evaluate_TAB.sh    # Evaluate TabVLA
bash run_evaluate_BadVLA.sh # Evaluate BadVLA
bash run_evaluate.sh    # Evaluate BackdoorVLA on LIBERO-Goal, LIBERO-Object, LIBERO-Spatial
bash run_evaluate_10.sh # Evaluate BackdoorVLA on LIBERO-10
```
### UADA,UPA,TMA
Please follow [RoboticAttack](https://github.com/William-wAng618/roboticAttack) installation instructions to configure the base environment first.

Navigate to the UADA,UPA,TMA attack directory:

```bash
cd OpenVLA/UADA_UPA_TMA
```
#### Adversarial Patch Generation:
```bash
bash scripts/run_UADA.sh
bash scripts/run_UPA.sh
bash scripts/run_TMA.sh
```
#### Evaluate:
```bash
bash scripts/run_simulation.sh
```
### RoboGCG
Please follow [RoboGCG](https://github.com/eliotjones1/robogcg) installation instructions to configure the base environment first.


Navigate to the RoboGCG attack directory:

```bash
cd OpenVLA/robogcg
```

Running RoboGCG by:
```bash
bash run_robogcg.sh
```

### FreezeVLA,PGD

Navigate to the corresponding directory:

```bash
cd OpenVLA/FreezeVLA
```

- create environment by
```bash
conda env create -f environment.yml
```
- You can obtain raw_dataset (for the format of raw_dataset, please refer to PackedDataset) by extracting certain frames from a trajectory replay, and then generate reference prompt from gpt by: 
```
 python generate_ref_prompt.py --save_dir /Your/save_dir --dataset_path /Your/raw_dataset_path
```
- evaluate PGD or FreezeVLA+GPT by
```
torchrun --nproc_per_node=1 run_spatialvla_gpt_ddp.py --max_samples 256 --attack pgd
torchrun --nproc_per_node=1 run_spatialvla_gpt_ddp.py --max_samples 256 --attack ours_gpt
```
## SpatialVLA
### Backdoor Attacks and UADA,UPA,TMA
Before running backdoor attacks on SpatialVLA, please make sure the required environments are properly set up:
- 🧠 Please follow [SpatialVLA](https://github.com/SpatialVLA/SpatialVLA) installation instructions to configure the base environment first.

- 🧪 Experiments are conducted in the [LIBERO](https://github.com/moojink/openvla-oft/blob/main/LIBERO.md) simulation environment. Make sure to install LIBERO and its dependencies as described in their official documentation.

Navigate to the SpatialVLA directory:

```bash
cd SpatialVLA
```
#### Training:
```bash
bash finetune_TAB.sh # TabVLA
bash finetune_Badvla_fir.sh #  Trigger Injection Stage of BadVLA
bash finetune_Badvla_sec.sh # Clean Performance Enhancement of BadVLA
bash finetune.sh # BackdoorVLA
bash reproduce_TMA.sh # TMA
bash reproduce_UADA.sh  # UADA
bash reproduce_UPA.sh   # UPA
```
#### Evaluate:
```bash
cd LIBERO
run_evaluate_TAB.sh     # TabVLA
run_evaluate_Badvla.sh      # BadVLA
run_evaluate.sh     # BackdoorVLA on LIBERO-Goal LIBERO-Object LIBERO-Spatial
run_evaluate_10.sh  # BackdoorVLA on LIBERO-10
run_TMA.sh  # TMA
run_UADA.sh # UADA
run_UPA.sh  # UPA
```
### RoboGCG
Please follow [RoboGCG](https://github.com/eliotjones1/robogcg) installation instructions to configure the base environment first, and install `transformers==4.47.1`

Navigate to the corresponding directory:

```bash
cd robogcg_spatialvla
```
Running RoboGCG by:
```bash
bash run_robogcg.sh
```

### FreezeVLA,PGD

Navigate to the corresponding directory:

```bash
cd FreezeVLA
```

- create environment by
```bash
conda env create -f environment.yml
```
- You can obtain raw_dataset (for the format of raw_dataset, please refer to PackedDataset) by extracting certain frames from a trajectory replay, and then generate reference prompt from gpt by: 
```
 python generate_ref_prompt.py --save_dir /Your/save_dir --dataset_path /Your/raw_dataset_path
```
- evaluate PGD or FreezeVLA+GPT by
```
torchrun --nproc_per_node=1 run_spatialvla_gpt_ddp.py --max_samples 256 --attack pgd
torchrun --nproc_per_node=1 run_spatialvla_gpt_ddp.py --max_samples 256 --attack ours_gpt
```

## $\pi_0$-fast
### Backdoor Attacks and TMA
Before running backdoor attacks on $\pi_0$-fast, please make sure the required environments are properly set up:
- 🧠 Please follow [Openpi](https://github.com/Physical-Intelligence/openpi) installation instructions to configure the base environment first.

- 🧪 Experiments are conducted in the LIBERO simulation environment. Please follow install instructions in [Openpi](https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/README.md) to configure the simulation environment.
#### Setup
Navigate to the Pi0-Fast directory:

```bash
cd Pi0-Fast
```
Converting Your dataset (You can define your dataset in the src/openpi/training/config.py) to lerobot format and Computing the normalization statistics of it by:
```bash
bash convert_data.sh
bash compute_norm.sh
```

#### Training:
```bash
bash train_Tab.sh # TabVLA
bash train_BadVLA.sh # BadVLA
bash train.sh # BackdoorVLA
bash train_TMA.sh # TMA
```
#### Evaluate(Please run the server in one terminal and conduct evaluation in another terminal):
```bash
# TabVLA
bash serve_TabVLA.sh 
bash run_eval_Tab.sh 
# BadVLA
bash serve_BadVLA.sh 
bash run_eval_BadVLA.sh 
# BackdoorVLA
bash serve_BackdoorVLA.sh 
# BackdoorVLA on LIBERO-Goal,LIBERO-Object,LIBERO-Spatial
bash run_eval.sh 
# BackdoorVLA on LIBERO-10 
bash run_eval_10.sh 
# TMA 
python scripts/serve_policy.py --env LIBERO
bash run_eval_TMA.sh  
```
### RoboGCG
Please follow [RoboGCG](https://github.com/eliotjones1/robogcg) installation instructions to configure the base environment first.

Navigate to the corresponding directory:
```bash
cd robogcg_pi0_fast
```
Running RoboGCG by:
```bash
bash run_robogcg.sh
```

### FreezeVLA,PGD

Navigate to the corresponding directory:

```bash
cd FreezeVLA
```

- create environment by
```bash
conda env create -f environment.yml
```
- You can obtain raw_dataset (for the format of raw_dataset, please refer to PackedDataset) by extracting certain frames from a trajectory replay, and then generate reference prompt from gpt by: 
```
 python generate_ref_prompt.py --save_dir /Your/save_dir --dataset_path /Your/raw_dataset_path
```
- evaluate PGD or FreezeVLA+GPT by
```
torchrun --nproc_per_node=1 run_pi_gpt_ddp.py --max_samples 256 --attack pgd
torchrun --nproc_per_node=1 run_pi_gpt_ddp.py --max_samples 256 --attack ours_gpt
```