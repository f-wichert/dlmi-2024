import logging

import yaml
import os
import subprocess
from pathlib import Path


def setup_environment():
    os.environ['PYTHONPATH'] = f"{os.environ.get('PYTHONPATH', '')}:./"
    os.environ['KAGGLE_CONFIG_DIR'] = str(Path.cwd())


def run_command(command):
    try:
        subprocess.run(command, check=True, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {command}")
        print(f"Error: {e}")
        raise


def run_experiment(name):
    # Preprocess
    preprocess_cmd = f"python dlmi/preprocess/pipeline.py {name}"
    preprocess_cmd += f" --dataset_split train"
    run_command(preprocess_cmd)
    preprocess_cmd = f"python dlmi/preprocess/pipeline.py {name}"
    preprocess_cmd += f" --dataset_split test"
    run_command(preprocess_cmd)

    # Train model
    run_command(f"python dlmi/train/pipeline.py {name}")

    # Test model
    run_command(f"python dlmi/inference/pipeline.py {name}")

    # Evaluate
    run_command(f"python dlmi/evaluate/pipeline.py {name}")


def main():
    """Main function to run all experiments"""
    setup_environment()

    # Load configuration
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    for exp in config.keys():
        logging.info(f"Running experiment {exp}")
        run_experiment(exp)

    # run_command(f"python steps/new_meta_eval.py")


if __name__ == '__main__':
    main()
