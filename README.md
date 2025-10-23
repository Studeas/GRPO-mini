# GRPO-mini: Minimal Implementation of Group Relative Policy Optimization

A self-contained implementation of Group Relative Policy Optimization (GRPO) for training language models with reinforcement learning, demonstrated on the Countdown arithmetic task.

## Overview

This repository implements GRPO, a policy gradient method that uses group-based reward normalization without requiring a reference policy or value network. The implementation trains a Qwen2.5-3B model to solve arithmetic puzzles.


## Installation

1. **Create conda environment**
```bash
conda env create -f environment.yml
conda activate grpo-zero
```

2. **Download model and dataset**
```bash
git clone https://huggingface.co/Qwen/Qwen2.5-3B-Instruct
git clone https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4
```

## Usage

1. **Open the notebook**
```bash
jupyter notebook grpo.ipynb
```

2. **Run all cells** to start training

3. **Monitor progress**
```bash
tensorboard --logdir logs/
```

## Configuration

- `config.yaml` - Standard configuration (24GB+ VRAM)
- `config_24GB.yaml` - Memory-efficient configuration (16GB+ VRAM)

Edit these files to customize training parameters.

## Project Structure

```
GRPO-mini/
├── grpo.ipynb              # Main implementation
├── config.yaml             # Configuration files
├── config_24GB.yaml
├── environment.yml         # Dependencies
└── README.md
```

## Acknowledgements

Credit to the original implementation from: [GRPO-Zero](https://github.com/policy-gradient/GRPO-Zero)
