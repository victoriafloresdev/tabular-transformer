# preprocess_data.py
import pandas as pd
import numpy as np
import os
import glob
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def inject_nans(df, nan_percentage, target_column, random_state=42):
    """Injects a specified percentage of NaNs into each feature column."""
    df_nan = df.copy()
    np.random.seed(random_state)
    num_rows = len(df_nan)
    num_nans = int(num_rows * nan_percentage)

    if num_nans == 0 and nan_percentage > 0:
        logging.warning(f"Dataset size {num_rows} too small to inject {nan_percentage*100}% NaNs meaningfully. Skipping NaN injection for this level.")
        return None # Indicate skipping

    logging.info(f"Injecting {num_nans} NaNs ({nan_percentage*100:.1f}%) per feature column.")

    feature_columns = [col for col in df.columns if col != target_column]

    for col in feature_columns:
        # Find indices that are not already NaN (if any)
        non_nan_indices = df_nan.index[df_nan[col].notna()]

        # If fewer non-NaN values exist than the NaNs we want to inject,
        # set all available non-NaNs to NaN for this column.
        nans_to_inject_this_col = min(num_nans, len(non_nan_indices))

        if nans_to_inject_this_col > 0:
            # Randomly choose indices from the non-NaN population
            nan_indices = np.random.choice(non_nan_indices, nans_to_inject_this_col, replace=False)
            df_nan.loc[nan_indices, col] = np.nan
        elif num_nans > 0:
             logging.warning(f"Column '{col}' already has high NaN count or is small. Could not inject {num_nans} NaNs.")


    return df_nan

def main(args):
    """Loads datasets, injects NaNs, and saves processed files."""
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    target_column = args.target_column
    nan_percentages = [float(p) for p in args.nan_percentages]

    if not input_dir.exists():
        logging.error(f"Input directory not found: {input_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Processing datasets from: {input_dir}")
    logging.info(f"Saving processed datasets to: {output_dir}")

    dataset_paths = glob.glob(os.path.join(input_dir, "*.csv"))

    if not dataset_paths:
        logging.error(f"No CSV files found in {input_dir}")
        return

    for dataset_path in dataset_paths:
        dataset_name = Path(dataset_path).stem
        logging.info(f"--- Processing dataset: {dataset_name} ---")
        try:
            df_original = pd.read_csv(dataset_path)

            # Save original dataset (0% NaN)
            original_output_path = output_dir / f"{dataset_name}_00nan.csv"
            df_original.to_csv(original_output_path, index=False)
            logging.info(f"Saved original dataset to {original_output_path}")

            # Inject NaNs and save
            for i, percentage in enumerate(nan_percentages):
                if percentage == 0: continue # Skip explicit 0%

                logging.info(f"Injecting {percentage*100:.1f}% NaNs...")
                # Use a different seed for each percentage level for variety
                df_nan = inject_nans(df_original, percentage, target_column, random_state=args.random_seed + i + 1)

                if df_nan is not None:
                    output_filename = f"{dataset_name}_{int(percentage*100):02d}nan.csv"
                    output_path = output_dir / output_filename
                    df_nan.to_csv(output_path, index=False)
                    logging.info(f"Saved {percentage*100:.1f}% NaN dataset to {output_path}")
                else:
                    logging.warning(f"Skipped saving {percentage*100:.1f}% NaN dataset for {dataset_name} due to injection issues.")


        except Exception as e:
            logging.error(f"Failed to process {dataset_name}: {e}")

    logging.info("--- Preprocessing complete ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess datasets by injecting NaN values.")
    parser.add_argument("--input_dir", type=str, default="datasets", help="Directory containing original CSV datasets.")
    parser.add_argument("--output_dir", type=str, default="processed_datasets", help="Directory to save processed datasets.")
    parser.add_argument("--target_column", type=str, required=True, help="Name of the target variable column.")
    parser.add_argument("--nan_percentages", nargs='+', default=['0.25', '0.50', '0.75'], help="List of NaN percentages to inject (e.g., 0.25 0.5 0.75).")
    parser.add_argument("--random_seed", type=int, default=42, help="Base random seed for NaN injection.")

    args = parser.parse_args()
    main(args)