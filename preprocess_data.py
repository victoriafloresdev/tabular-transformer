# preprocess_data.py
import pandas as pd
import numpy as np
import os
import glob
import argparse
import logging
import json # Added for saving indices
from pathlib import Path
from sklearn.model_selection import train_test_split # Added for splitting
from sklearn.preprocessing import LabelEncoder # Added for stratification

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration for Splitting ---
RANDOM_SEED = 42
VAL_SIZE = 0.10 # 10% of original data for validation
TEST_SIZE = 0.10 # 10% of original data for test
# Train size will be 1.0 - VAL_SIZE - TEST_SIZE = 80%

def inject_nans(df, nan_percentage, target_column, random_state=42):
    """Injects a specified percentage of NaNs into each feature column."""
    df_nan = df.copy()
    np.random.seed(random_state)
    num_rows = len(df_nan)
    num_nans = int(num_rows * nan_percentage)

    if num_nans == 0 and nan_percentage > 0:
        logging.warning(f"Dataset size {num_rows} too small to inject {nan_percentage*100:.1f}% NaNs meaningfully. Skipping NaN injection for this level.")
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

def split_data_and_get_indices(df, target_column):
    """Performs train/val/test split and returns the indices for each set."""
    logging.info("Performing train/validation/test split...")
    try:
        X = df.drop(columns=[target_column])
        y = df[target_column]
    except KeyError:
        logging.error(f"Target column '{target_column}' not found during split.")
        return None

    # Encode target for stratification if needed
    le = LabelEncoder()
    try:
        y_encoded = le.fit_transform(y)
    except Exception as e:
        logging.error(f"Failed to encode target column {target_column} for stratification: {e}")
        return None # Cannot proceed without encoding if needed

    indices = df.index.to_numpy() # Get the original indices

    # Calculate adjusted validation size for the second split
    if (1.0 - TEST_SIZE) <= 0 or TEST_SIZE < 0:
        logging.error(f"Invalid TEST_SIZE: {TEST_SIZE}. Must be between 0 and 1.")
        return None
    if VAL_SIZE < 0 or VAL_SIZE >= (1.0 - TEST_SIZE):
         logging.error(f"Invalid VAL_SIZE: {VAL_SIZE}. Must be >= 0 and less than (1 - TEST_SIZE).")
         return None
    val_size_adjusted = VAL_SIZE / (1.0 - TEST_SIZE)

    # Check for stratification possibility
    unique_classes, counts = np.unique(y_encoded, return_counts=True)
    can_stratify_initial = len(unique_classes) >= 2 and np.all(counts >= 2)
    stratify_initial = y_encoded if can_stratify_initial else None
    if not can_stratify_initial:
        logging.warning("Cannot stratify initial train/test split due to low class counts. Performing non-stratified split.")


    # Split 1: Separate Test set indices
    try:
        train_val_indices, test_indices = train_test_split(
            indices,
            test_size=TEST_SIZE,
            stratify=stratify_initial,
            random_state=RANDOM_SEED
        )
        logging.info(f"Split indices: Train+Val={len(train_val_indices)}, Test={len(test_indices)}")
    except ValueError as e:
        logging.error(f"Error during initial index split (check class counts vs TEST_SIZE): {e}. Trying without stratification.")
        try:
            train_val_indices, test_indices = train_test_split(indices, test_size=TEST_SIZE, random_state=RANDOM_SEED)
        except Exception as e_fallback:
            logging.error(f"Non-stratified initial index split failed: {e_fallback}")
            return None

    # Get corresponding target labels for the train_val set for the second split stratification
    y_train_val = y_encoded[train_val_indices]

    # Check stratification possibility for the second split
    unique_train_val, counts_train_val = np.unique(y_train_val, return_counts=True)
    can_stratify_second = len(unique_train_val) >= 2 and np.all(counts_train_val >= 2)
    stratify_second = y_train_val if can_stratify_second else None
    if not can_stratify_second:
         logging.warning("Cannot stratify train/validation split due to low class counts after initial split. Performing non-stratified split.")

    # Split 2: Separate Train and Validation set indices
    try:
        train_indices, val_indices = train_test_split(
            train_val_indices, # Split the indices obtained from the first split
            test_size=val_size_adjusted,
            stratify=stratify_second,
            random_state=RANDOM_SEED # Use same seed for consistency
        )
        logging.info(f"Final Split index counts: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")

        # Return the actual index values
        return {
            "train_indices": train_indices.tolist(),
            "val_indices": val_indices.tolist(),
            "test_indices": test_indices.tolist()
        }
    except ValueError as e:
        logging.error(f"Error during train/validation index split (check class counts vs VAL_SIZE): {e}. Trying without stratification.")
        try:
             train_indices, val_indices = train_test_split(train_val_indices, test_size=val_size_adjusted, random_state=RANDOM_SEED)
             logging.info(f"Final Split index counts: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")
             return { "train_indices": train_indices.tolist(), "val_indices": val_indices.tolist(), "test_indices": test_indices.tolist() }
        except Exception as e_fallback:
            logging.error(f"Non-stratified train/validation index split also failed: {e_fallback}")
            return None


def main(args):
    """Loads datasets, performs split & saves indices, injects NaNs, and saves processed files."""
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    target_column = args.target_column
    nan_percentages = [float(p) for p in args.nan_percentages]

    if not input_dir.exists():
        logging.error(f"Input directory not found: {input_dir}")
        return
    if not input_dir.is_dir():
        logging.error(f"Input path is not a directory: {input_dir}")
        return


    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Processing datasets from: {input_dir}")
    logging.info(f"Saving processed datasets and index splits to: {output_dir}")

    dataset_paths = glob.glob(os.path.join(input_dir, "*.csv"))

    if not dataset_paths:
        logging.error(f"No CSV files found in {input_dir}")
        return

    total_datasets = len(dataset_paths)
    logging.info(f"Found {total_datasets} datasets to process.")

    for i, dataset_path in enumerate(dataset_paths):
        dataset_name = Path(dataset_path).stem
        logging.info(f"\n--- Processing dataset {i+1}/{total_datasets}: {dataset_name} ---")
        try:
            df_original = pd.read_csv(dataset_path)
            if df_original.empty:
                logging.warning(f"Dataset {dataset_name} is empty. Skipping.")
                continue

            # --- Perform Split and Save Indices ---
            split_indices_info = split_data_and_get_indices(df_original, target_column)

            if split_indices_info:
                index_output_filename = f"{dataset_name}_split_indices.json"
                index_output_path = output_dir / index_output_filename
                try:
                    with open(index_output_path, 'w') as f:
                        json.dump(split_indices_info, f, indent=4)
                    logging.info(f"Saved train/val/test indices to {index_output_path}")
                except IOError as e:
                    logging.error(f"Failed to save index file {index_output_path}: {e}")
                    # Decide whether to continue without indices or stop
                    logging.warning("Continuing NaN injection without saved indices for this dataset.")
            else:
                logging.error(f"Failed to generate split indices for {dataset_name}. Cannot guarantee consistent splits.")
                # Decide whether to continue or stop
                logging.warning("Continuing NaN injection without saved indices for this dataset.")

            # --- Save Original and Inject NaNs ---
            # Save original dataset (renamed to _00nan)
            original_output_path = output_dir / f"{dataset_name}_00nan.csv"
            df_original.to_csv(original_output_path, index=False)
            logging.info(f"Saved original dataset (as _00nan) to {original_output_path}")

            # Inject NaNs and save other versions
            for j, percentage in enumerate(nan_percentages):
                if percentage == 0: continue # Skip explicit 0%, already saved

                logging.info(f"Injecting {percentage*100:.1f}% NaNs...")
                # Use a different seed for each percentage level for variety
                df_nan = inject_nans(df_original, percentage, target_column, random_state=args.random_seed + j + 1)

                if df_nan is not None:
                    output_filename = f"{dataset_name}_{int(percentage*100):02d}nan.csv"
                    output_path = output_dir / output_filename
                    df_nan.to_csv(output_path, index=False)
                    logging.info(f"Saved {percentage*100:.1f}% NaN dataset to {output_path}")
                else:
                    logging.warning(f"Skipped saving {percentage*100:.1f}% NaN dataset for {dataset_name} due to injection issues.")


        except Exception as e:
            logging.error(f"Failed to process {dataset_name}: {e}")
            import traceback
            logging.error(traceback.format_exc()) 


    logging.info("\n--- Preprocessing complete ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess datasets: perform train/val/test split, save indices, inject NaN values.")
    parser.add_argument("--input_dir", type=str, default="datasets", help="Directory containing original CSV datasets.")
    parser.add_argument("--output_dir", type=str, default="processed_datasets", help="Directory to save processed datasets and split indices.")
    parser.add_argument("--target_column", type=str, required=True, help="Name of the target variable column.")
    parser.add_argument("--nan_percentages", nargs='+', default=['0.25', '0.50', '0.75'], help="List of NaN percentages to inject (e.g., 0.25 0.5 0.75). 0% (_00nan) is saved automatically.")
    parser.add_argument("--random_seed", type=int, default=RANDOM_SEED, help=f"Base random seed for splitting and NaN injection (default: {RANDOM_SEED}).")

    args = parser.parse_args()
    main(args)