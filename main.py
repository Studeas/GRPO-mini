import argparse
import logging
import sys
from pathlib import Path

import yaml

from src.train_grpo import run_training


def setup_logging(log_file_path):
    file_handler = logging.FileHandler(log_file_path)
    console_handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[file_handler, console_handler],
        force=True,
    )


def main():
    parser = argparse.ArgumentParser(description="GRPO Training Script")
    parser.add_argument("--config", type=str, default="config_math.yaml", help="Path to the config file")
    parser.add_argument(
        "--dataset_name", type=str, default="countdown", choices=["gsm8k", "math", "countdown"], help="Dataset to use"
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    config_name = config_path.stem

    work_dir = Path("runs") / f"{config_name}_{args.dataset_name}"
    work_dir.mkdir(parents=True, exist_ok=True)

    log_file_path = work_dir / "training.log"
    setup_logging(log_file_path)
    logger = logging.getLogger(__name__)

    logger.info(f"Loading config from: {args.config}")
    with open(args.config) as f:
        config = yaml.safe_load(f)

    run_training(config, args.dataset_name, work_dir)


if __name__ == "__main__":
    main()
