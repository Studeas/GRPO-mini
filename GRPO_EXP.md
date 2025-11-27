
## Set up
```
mkdir ~/scratch/workdir
ln -s ~/scratch/workdir runs

mkdir ~/scratch/models
ln -s ~/scratch/workdir models
```

Run with:
```
python train_grpo.py --config path_to_your_config --dataset_name OPTIONAL
```

## Implementation Notes
- Due to limited cuda memory, chunked generation is enabled. Tune `inference_batch_size` in rollout for speed and memory balance.
- Left padding is set.
- Log files could be found in `runs/confi_name/training.log`
- Checkpoints are saved at `runs/confi_name/checkpoints/`
- Datsets available: `gsm8k`, `math`


## Experiments to run

- Baseline comparsion: KL, unbiased KL, OT; oringal model (w/o RL)

- Abalation studies: Beta, windows size in OT; beta in KL and unbiased KL

