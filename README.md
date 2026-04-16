<div align="center">

# From $P(y|x)$ to $P(y)$: Investigating Reinforcement Learning in Pre-train Space

[![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2604.14142) [![Github](https://img.shields.io/badge/TTRL-000000?style=for-the-badge&logo=github&logoColor=000&logoColor=white)](https://github.com/Trae1ounG/Pretrain_Space_RLVR)
</div>



## Overview
DSRL combines Negative Sample Reinforcement in the Pre-train Space (NSR-PreRL) with standard Post-train Space RL via a Policy Reincarnation strategy. The model first prunes incorrect reasoning paths and elicits intrinsic reasoning capabilities through NSR-PreRL warmup, then switches to standard RL for fine-grained optimization.

![intro](./assets/intro.png)

## Quick Start

### Installation 

```
conda create -y -n dsrl python=3.10.17 && conda activate dsrl
pip install -r requirements
python -m pip install flash-attn --no-build-isolation
pip install -e .
```
 

### Training

**GRPO:** Standard GRPO training without pre-train space loss. All positive and negative samples contribute to the RL gradient.

```bash
bash run_code/Baseline_qwen3-4b.sh
```


**DSRL:** Complete Dual Space RL pipeline: NSR-PreRL warmup for N steps (negative samples only, pre-train loss only), then switches to standard RL.

```bash
bash run_code/DSRL_qwen3-4b.sh
```

Key configuration parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pretrain_loss_disable_step` | 20 | Number of NSR-PreRL warmup steps (recommended: 10–25) |
| `negative_pretrain` | True | Only negative samples contribute to pre-train loss |
| `pretrain_only_warmup` | True | Disable RL loss during warmup phase |

### Ablation

Provides four ablation variants. Uncomment the desired variant block in the script to switch:

```bash
bash run_code/DSRL_abla_qwen3-4b.sh
```

| Variant | Description |
|---------|-------------|
| **NSR-Pretrain** | Negative samples only + pre-train loss only (no RL) |
| **PSR-Pretrain** | Positive samples only + pre-train loss only (no RL) |
| **NSR-RL** | Negative samples only contribute to RL loss (no pre-train loss) |
| **PSR-RL** | Positive samples only contribute to RL loss (no pre-train loss) |

## Citation
If you find our paper or code useful, please consider cite our work:
```
@article{tan2026from,
    title={From $P(y|x)$ to $P(y)$: Investigating Reinforcement Learning in Pre-train Space}, 
    author={Yuqiao Tan and Minzheng Wang and Bo Liu and Zichen Liu and Tian Liang and Shizhu He and Jun Zhao and Kang Liu},
    journal={arXiv preprint arXiv:2604.14142},
    year={2026},
    url={https://arxiv.org/abs/2604.14142}, 
}
```