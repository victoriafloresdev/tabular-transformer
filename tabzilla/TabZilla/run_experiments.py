import argparse
import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import logging

def setup_logger(log_dir):
    """Set up logging to file and console"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"experiment_log_{timestamp}.txt"
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger, log_file

def run_experiment(model, dataset_dir, config_file, logger):
    """Run a single experiment and log output"""
    cmd = [
        "python", "tabzilla_experiment.py",
        "--model_name", model,
        "--dataset_dir", str(dataset_dir),
        "--experiment_config", config_file
    ]
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        logger.info(f"Success: {model} - {dataset_dir.name}\nOutput:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed: {model} - {dataset_dir.name}\nError:\n{e.output}")
        return False
    return True

def main(args):
    logger, log_file = setup_logger(args.log_dir)
    logger.info(f"Starting experiments with config: {args.config}")
    logger.info(f"Log file: {log_file.resolve()}")

    # Experiment parameters
    models = ["XGBoost", "RandomForest", "KNN", "SAINT", "TabTransformer"]
    base_datasets = [
        "openml__credit-g__31",
        "openml__diabetes__37",
        "openml__adult__7592",
        "openml__spambase__43",
        "openml__Amazon_employee_access__34539"
    ]
    missing_suffixes = ["_100pct", "_25pct_missing", "_50pct_missing", "_75pct_missing"]

    # Create task list
    tasks = []
    for model in models:
        for base in base_datasets:
            for suffix in missing_suffixes:
                dataset_dir = Path(args.dataset_dir) / f"{base}{suffix}"
                if dataset_dir.exists():
                    tasks.append((model, dataset_dir))

    # Run experiments with progress bar
    success_count = 0
    with tqdm(total=len(tasks), desc="Running experiments", unit="task") as pbar:
        for model, dataset_dir in tasks:
            result = run_experiment(model, dataset_dir, args.config, logger)
            if result:
                success_count += 1
            pbar.update(1)
            pbar.set_postfix_str(f"Success: {success_count}/{len(tasks)}")

    logger.info(f"\nExperiment summary:")
    logger.info(f"Total tasks: {len(tasks)}")
    logger.info(f"Successful: {success_count}")
    logger.info(f"Failed: {len(tasks) - success_count}")
    logger.info(f"Log file saved to: {log_file.resolve()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run missing data experiments with progress tracking')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to experiment config file')
    parser.add_argument('--dataset-dir', type=str, default="datasets",
                      help='Base directory containing datasets')
    parser.add_argument('--log-dir', type=str, default="experiment_logs",
                      help='Directory to save log files')
    
    args = parser.parse_args()
    
    main(args)