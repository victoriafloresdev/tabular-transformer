# run_custom_experiment.py
import pandas as pd
import numpy as np
import os
import sys
import glob
import time
import argparse
import logging
import csv
from pathlib import Path
from collections import defaultdict
import re # For parsing filenames

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, make_scorer
from sklearn.exceptions import ConvergenceWarning
import warnings

# Try importing PyTorch and Optuna
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not found.")

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Optuna not found. HPO with Optuna will not be available. Please install it: pip install optuna")


# --- Configuration ---
RANDOM_SEED = 42
N_CV_SPLITS = 10
VAL_SIZE = 0.10 # 10% of original data for validation
TEST_SIZE = 0.10 # 10% of original data for test
# Train size will be 1.0 - VAL_SIZE - TEST_SIZE = 80%

# --- Models to Run ---
# User specified models
MODELS_TO_RUN = [
    'LogisticRegression', # From baseline_models.py
    'RandomForest',       # From tree_models.py
    'XGBoost',            # From tree_models.py
    'TabTransformer',     # From tabtransformer.py
    'SAINT',              # From saint.py
    'FTTransformer'       # From rtdl.py
]

# --- HPO Search Spaces (PLACEHOLDERS - REVIEW AND CUSTOMIZE!) ---
# Add specific parameters and ranges based on model documentation and needs
HPO_SEARCH_SPACES = {
    'LogisticRegression': {
        'C': lambda trial: trial.suggest_float('C', 1e-4, 1e2, log=True),
        'solver': lambda trial: trial.suggest_categorical('solver', ['liblinear', 'saga']), # saga handles L1/L2
        'penalty': lambda trial: trial.suggest_categorical('penalty', ['l1', 'l2']),
        # Add 'max_iter' if needed, especially for 'saga'
        'max_iter': lambda trial: trial.suggest_int('max_iter', 100, 1000),
    },
    'RandomForest': {
        'n_estimators': lambda trial: trial.suggest_int('n_estimators', 50, 300),
        'max_depth': lambda trial: trial.suggest_int('max_depth', 5, 30),
        'min_samples_split': lambda trial: trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': lambda trial: trial.suggest_int('min_samples_leaf', 1, 5),
    },
    'XGBoost': {
        'n_estimators': lambda trial: trial.suggest_int('n_estimators', 50, 500),
        'learning_rate': lambda trial: trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': lambda trial: trial.suggest_int('max_depth', 3, 12),
        'subsample': lambda trial: trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': lambda trial: trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': lambda trial: trial.suggest_float('gamma', 0, 0.5),
    },
    'TabTransformer': { # Needs PyTorch - params depend heavily on implementation
        'depth': lambda trial: trial.suggest_int('depth', 1, 6),
        'heads': lambda trial: trial.suggest_int('heads', 4, 8),
        'attn_dropout': lambda trial: trial.suggest_float('attn_dropout', 0.1, 0.5),
        'ff_dropout': lambda trial: trial.suggest_float('ff_dropout', 0.1, 0.5),
        'lr': lambda trial: trial.suggest_float('lr', 1e-5, 1e-3, log=True), # Optimizer learning rate
        'batch_size': lambda trial: trial.suggest_categorical('batch_size', [64, 128, 256]),
         # Note: TabTransformer needs 'cat_dims' (categorical dimensions/cardinalities) during init! Script needs modification to pass this.
         # Note: Also needs 'num_continuous'.
         # Note: Training parameters like epochs are often separate.
         # For now, assume some default epochs in the model wrapper
    },
     'SAINT': { # Needs PyTorch - params depend heavily on implementation
        'depth': lambda trial: trial.suggest_int('depth', 1, 6),
        'heads': lambda trial: trial.suggest_int('heads', 4, 8),
        'dropout': lambda trial: trial.suggest_float('dropout', 0.1, 0.5),
        'lr': lambda trial: trial.suggest_float('lr', 1e-5, 1e-3, log=True),
        'batch_size': lambda trial: trial.suggest_categorical('batch_size', [64, 128, 256]),
        # Note: SAINT also likely needs 'categories' (cardinalities), 'num_continuous'.
        # Note: May involve self-supervised pre-training steps depending on TabZilla implementation.
    },
    'FTTransformer': { # Needs PyTorch - From rtdl.py
        'n_layers': lambda trial: trial.suggest_int('n_layers', 1, 6),
        'n_heads': lambda trial: trial.suggest_int('n_heads', 4, 8),
        'd_token': lambda trial: trial.suggest_int('d_token', 64, 256, step=32), # Dimension of tokens
        'ffn_d_hidden_factor': lambda trial: trial.suggest_float('ffn_d_hidden_factor', 1.0, 4.0),
        'attention_dropout': lambda trial: trial.suggest_float('attention_dropout', 0.1, 0.5),
        'ffn_dropout': lambda trial: trial.suggest_float('ffn_dropout', 0.0, 0.4),
        'lr': lambda trial: trial.suggest_float('lr', 1e-5, 1e-3, log=True),
        'batch_size': lambda trial: trial.suggest_categorical('batch_size', [64, 128, 256]),
        # Note: FTTransformer also needs numerical/categorical feature info during init.
        # Requires specific preprocessing (numerical scaling, categorical tokenization)
    }
}

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')
warnings.filterwarnings("ignore", category=FutureWarning, module='sklearn')
warnings.filterwarnings("ignore", category=UserWarning, module='optuna') # Optuna/Sklearn sometimes have verbose warnings

# --- Dynamic Model Loading ---
tabzilla_path = os.path.abspath('./tabzilla/TabZilla')
if tabzilla_path not in sys.path:
    sys.path.insert(0, tabzilla_path)
logging.info(f"Added to sys.path: {tabzilla_path}")

_MODEL_CLASSES = {}

def get_model_class(model_name):
    """Dynamically imports and returns the model class from TabZilla."""
    if model_name in _MODEL_CLASSES:
        return _MODEL_CLASSES[model_name]

    try:
        if model_name == 'LogisticRegression':
            from models.baseline_models import LogisticRegression # Check class name
            model_class = LogisticRegression
        elif model_name == 'RandomForest':
            from models.tree_models import RandomForest
            model_class = RandomForest
        elif model_name == 'XGBoost':
            # XGBoost might need explicit install: pip install xgboost
            try:
                from models.tree_models import XGBoost
                model_class = XGBoost
            except ImportError:
                logging.error("XGBoost requires 'xgboost' package. Install it.")
                raise
        elif model_name == 'TabTransformer':
            if not TORCH_AVAILABLE: raise ImportError("TabTransformer requires PyTorch.")
            from models.tabtransformer import TabTransformer # Check class name
            model_class = TabTransformer
        elif model_name == 'SAINT':
            if not TORCH_AVAILABLE: raise ImportError("SAINT requires PyTorch.")
            from models.saint import SAINT # Check class name
            model_class = SAINT
        elif model_name == 'FTTransformer':
             if not TORCH_AVAILABLE: raise ImportError("FTTransformer requires PyTorch.")
             # FTTransformer is often part of a library like 'rtdl'. Check TabZilla's rtdl.py
             # Assume the class FTTransformer exists within models/rtdl.py for now
             from models.rtdl import FTTransformer # Check class name in rtdl.py
             model_class = FTTransformer
        else:
            raise ImportError(f"Model '{model_name}' not recognized or mapped in get_model_class function.")

        _MODEL_CLASSES[model_name] = model_class
        logging.info(f"Successfully imported {model_name}")
        return model_class
    except ImportError as e:
        logging.error(f"Could not import model '{model_name}'. Check mapping, TabZilla structure, and dependencies. Error: {e}")
        raise
    except AttributeError as e:
         logging.error(f"Could not find class for '{model_name}' in its module. Check class name. Error: {e}")
         raise

def get_model_instance(model_name, params=None, random_state=RANDOM_SEED, device='cpu', **kwargs):
    """Gets an instance of the model with given parameters. Pass device for torch models."""
    ModelClass = get_model_class(model_name)
    instance_params = params.copy() if params else {}

    # --- Handle model-specific parameter names or needs ---
    if model_name in ['RandomForest', 'XGBoost', 'LogisticRegression']:
         instance_params['random_state'] = random_state # Sklearn models typically use this
         if model_name == 'XGBoost':
              # XGBoost might use 'seed' internally, but the TabZilla wrapper likely handles random_state
              pass

    # --- Handle PyTorch models ---
    # These often need device, input dimensions, cardinalities etc. passed at init
    # This is a MAJOR simplification - real integration needs careful handling of these args
    if model_name in ['TabTransformer', 'SAINT', 'FTTransformer']:
        if not TORCH_AVAILABLE: raise RuntimeError(f"{model_name} requires PyTorch, but it's not available.")
        instance_params['device'] = device # Assuming the wrappers accept a 'device' arg

        # *** CRITICAL ***: These models need info about features (num_continuous, cat_dims/categories) at init!
        # The current script structure doesn't easily pass this from data processing to here.
        # This needs significant refactoring for proper DL model usage.
        # We add placeholders from kwargs but this requires the calling code to provide them.
        if 'cat_dims' in kwargs: instance_params['cat_dims'] = kwargs['cat_dims']
        if 'num_continuous' in kwargs: instance_params['num_continuous'] = kwargs['num_continuous']
        if 'categories' in kwargs: instance_params['categories'] = kwargs['categories'] # For SAINT/FTT potentially
        if 'd_out' in kwargs: instance_params['d_out'] = kwargs['d_out'] # Output dimension (classes)

        # Remove HPO params that are actually training loop params (like lr, batch_size)
        # These should be handled by the model's internal training logic or a separate trainer function
        if 'lr' in instance_params: instance_params.pop('lr')
        if 'batch_size' in instance_params: instance_params.pop('batch_size')

        logging.warning(f"Instantiating {model_name}. CRITICAL: Ensure necessary feature info (cat_dims, num_continuous, d_out) is passed via **kwargs and handled correctly by the model wrapper.")


    # Remove keys that might not be direct __init__ args for sklearn models
    if model_name in ['LogisticRegression', 'RandomForest', 'XGBoost']:
         instance_params.pop('lr', None)
         instance_params.pop('batch_size', None)


    try:
        # Pass only relevant params
        # This requires knowing the exact __init__ signature or the wrapper handling **kwargs
        model = ModelClass(**instance_params)
        return model
    except Exception as e:
        logging.error(f"Failed to instantiate {model_name} with params {instance_params}. Error: {e}")
        logging.error("Check if HPO params match model's __init__ arguments and if necessary kwargs (like cat_dims, num_continuous for DL models) are provided.")
        raise


# --- Data Handling ---
def load_and_split_data(dataset_path, target_column):
    """Loads data, performs initial split into train, validation, and test."""
    logging.info(f"Loading data from: {dataset_path}")
    try:
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        logging.error(f"Dataset file not found: {dataset_path}")
        return None
    except Exception as e:
        logging.error(f"Error reading CSV {dataset_path}: {e}")
        return None

    if df.empty:
        logging.error(f"Dataset is empty: {dataset_path}")
        return None

    # Separate features and target
    try:
        X = df.drop(columns=[target_column])
        y = df[target_column]
    except KeyError:
        logging.error(f"Target column '{target_column}' not found in {dataset_path}.")
        return None

    # Basic check for target validity
    if y.isnull().any():
        logging.error(f"Target column '{target_column}' contains NaN values in {dataset_path}. This is not supported. Please clean the target column.")
        return None


    # Encode target labels if they are not numeric
    le = LabelEncoder()
    try:
         y_encoded = le.fit_transform(y)
         num_classes = len(le.classes_)
         logging.info(f"Encoded target column '{target_column}' using LabelEncoder. Found {num_classes} classes.")
    except Exception as e:
         logging.error(f"Failed to encode target column {target_column}: {e}")
         return None

    # Ensure target is suitable for stratification
    unique_classes, counts = np.unique(y_encoded, return_counts=True)
    if len(unique_classes) < 2:
         logging.error("Target column has less than 2 unique classes after encoding. Cannot proceed.")
         return None

    if np.any(counts < 2): # Needed for stratification in both splits
         logging.warning(f"Target column has classes with only 1 sample: {dict(zip(le.inverse_transform(unique_classes[counts<2]), counts[counts<2]))}. Stratification might fail or be unreliable.")
         # Allow proceeding but warn user. Stratify=None might be needed if sklearn raises errors later.

    # Calculate adjusted validation size for the second split
    if (1.0 - TEST_SIZE) <= 0 or TEST_SIZE < 0:
        logging.error(f"Invalid TEST_SIZE: {TEST_SIZE}. Must be between 0 and 1.")
        return None
    if VAL_SIZE < 0 or VAL_SIZE >= (1.0 - TEST_SIZE):
         logging.error(f"Invalid VAL_SIZE: {VAL_SIZE}. Must be >= 0 and less than (1 - TEST_SIZE).")
         return None

    val_size_adjusted = VAL_SIZE / (1.0 - TEST_SIZE)


    # Split 1: Separate Test set
    try:
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y_encoded,
            test_size=TEST_SIZE,
            stratify=y_encoded,
            random_state=RANDOM_SEED
        )
        logging.info(f"Split sizes: Train+Val={X_train_val.shape[0]}, Test={X_test.shape[0]}")
    except ValueError as e:
         logging.error(f"Error during initial train/test split (check class counts vs TEST_SIZE): {e}. Trying without stratification.")
         # Fallback to non-stratified split if stratification fails
         try:
             X_train_val, X_test, y_train_val, y_test = train_test_split(
                X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_SEED)
             logging.info(f"Performed non-stratified initial split.")
         except Exception as e_fallback:
              logging.error(f"Non-stratified initial split also failed: {e_fallback}")
              return None


    # Split 2: Separate Validation set from Train_Val
    try:
        # Check if stratification is possible for the second split
        unique_train_val, counts_train_val = np.unique(y_train_val, return_counts=True)
        stratify_val = y_train_val if np.all(counts_train_val >= 2) else None
        if stratify_val is None and len(unique_train_val) > 1:
            logging.warning("Cannot stratify train/validation split due to low class counts after initial split. Performing non-stratified split.")

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=val_size_adjusted,
            stratify=stratify_val,
            random_state=RANDOM_SEED
        )
        logging.info(f"Final Split sizes: Train={X_train.shape[0]}, Val={X_val.shape[0]}, Test={X_test.shape[0]}")

        # Collect feature information needed by some models
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_features = X.select_dtypes(include=np.number).columns.tolist()
        # Calculate cardinalities (needed for embeddings)
        cat_dims = [(col, X_train[col].nunique()) for col in categorical_features] # Use training set nunique


        feature_info = {
            "numerical": numerical_features,
            "categorical": categorical_features,
            "cat_dims": cat_dims, # List of tuples: (col_name, cardinality)
            "num_classes": num_classes
        }

        return X_train, y_train, X_val, y_val, X_test, y_test, feature_info

    except ValueError as e:
        logging.error(f"Error during train/validation split (check class counts vs VAL_SIZE): {e}. Trying without stratification.")
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val, test_size=val_size_adjusted, random_state=RANDOM_SEED)
            logging.info(f"Performed non-stratified train/validation split.")
            # Still collect feature info
            categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
            numerical_features = X.select_dtypes(include=np.number).columns.tolist()
            cat_dims = [(col, X_train[col].nunique()) for col in categorical_features]
            feature_info = { "numerical": numerical_features, "categorical": categorical_features, "cat_dims": cat_dims, "num_classes": num_classes }
            return X_train, y_train, X_val, y_val, X_test, y_test, feature_info
        except Exception as e_fallback:
             logging.error(f"Non-stratified train/validation split also failed: {e_fallback}")
             return None


# --- Preprocessing Pipeline ---
def create_preprocessor(feature_info, perform_imputation=True, scale_numeric=False):
    """Creates a preprocessing pipeline for imputation, encoding, and optional scaling."""
    numeric_features = feature_info["numerical"]
    categorical_features = feature_info["categorical"]

    logging.info(f"Numeric features: {numeric_features}")
    logging.info(f"Categorical features: {categorical_features}")

    transformers = []

    # Numeric preprocessing
    if numeric_features:
        numeric_steps = []
        if perform_imputation:
            numeric_steps.append(('imputer', SimpleImputer(strategy='median')))
        if scale_numeric:
            # Scaling is often crucial for DL models and sometimes helpful for others
            numeric_steps.append(('scaler', StandardScaler()))
        if numeric_steps:
             numeric_pipeline = Pipeline(steps=numeric_steps)
             transformers.append(('num', numeric_pipeline, numeric_features))

    # Categorical preprocessing
    if categorical_features:
        categorical_steps = []
        if perform_imputation:
            # Impute categoricals first
            categorical_steps.append(('imputer', SimpleImputer(strategy='most_frequent')))
        # Then OneHotEncode
        categorical_steps.append(('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)))
        if categorical_steps:
             categorical_pipeline = Pipeline(steps=categorical_steps)
             transformers.append(('cat', categorical_pipeline, categorical_features))

    if not transformers:
        logging.warning("No numeric or categorical features identified for preprocessing.")
        return 'passthrough' # Return identity "transformer"

    # remainder='passthrough' keeps columns not specified in transformers (if any)
    preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough', verbose_feature_names_out=False)
    preprocessor.set_output(transform="pandas") # Keep pandas dataframe output
    return preprocessor

# --- Hyperparameter Optimization ---
def run_hpo(model_name, preprocessor, X_train, y_train, X_val, y_val, feature_info, n_trials=20, device='cpu'):
    """Performs HPO using Optuna. Fits preprocessor internally."""
    if not OPTUNA_AVAILABLE:
        logging.error("Optuna is not installed. Skipping HPO.")
        return {}

    logging.info(f"Starting HPO for {model_name} with {n_trials} trials...")
    search_space = HPO_SEARCH_SPACES.get(model_name)
    if not search_space:
        logging.warning(f"No HPO search space defined for {model_name}. Returning empty params.")
        return {}

    # Fit preprocessor ONCE on training data before Optuna trials
    try:
        logging.info("Fitting preprocessor on HPO training data...")
        if preprocessor != 'passthrough':
            preprocessor.fit(X_train)
            X_train_processed = preprocessor.transform(X_train)
            X_val_processed = preprocessor.transform(X_val)
            logging.info(f"Processed HPO Train shape: {X_train_processed.shape}, Val shape: {X_val_processed.shape}")
        else:
            X_train_processed = X_train
            X_val_processed = X_val
            logging.info("Passthrough preprocessor used.")

        # Check for NaN after preprocessing (shouldn't happen with imputation)
        if pd.DataFrame(X_train_processed).isnull().values.any() or pd.DataFrame(X_val_processed).isnull().values.any():
             logging.error("NaN values found AFTER preprocessing during HPO setup. Check imputation/pipeline.")
             # Attempt to replace remaining NaNs with 0 - potential data leakage/bias, but might salvage run
             X_train_processed = pd.DataFrame(X_train_processed).fillna(0)
             X_val_processed = pd.DataFrame(X_val_processed).fillna(0)
             logging.warning("Filled remaining NaNs with 0 before HPO trials.")


    except Exception as e:
        logging.error(f"Failed to fit/transform preprocessor during HPO setup for {model_name}: {e}")
        return {}

    # Data needed for model instantiation (DL models)
    model_init_kwargs = {
        "cat_dims": feature_info['cat_dims'], # Cardinalities
        "num_continuous": len(feature_info['numerical']),
        "d_out": feature_info['num_classes'], # Number of output classes
        # Add other necessary static args here if needed by models
    }

    def objective(trial):
        hpo_params = {name: func(trial) for name, func in search_space.items()}
        trial_params_for_instantiation = hpo_params.copy()

        # Separate training params (handled later) from instantiation params
        train_params = {}
        if model_name in ['TabTransformer', 'SAINT', 'FTTransformer']:
             if 'lr' in trial_params_for_instantiation:
                 train_params['lr'] = trial_params_for_instantiation.pop('lr')
             if 'batch_size' in trial_params_for_instantiation:
                 train_params['batch_size'] = trial_params_for_instantiation.pop('batch_size')
             # Assume some default epochs are set in the model wrapper/fit method
             # train_params['epochs'] = trial.suggest_int('epochs', 5, 50) # Optionally optimize epochs


        try:
            # Instantiate model with HPO params AND static feature info
            model = get_model_instance(model_name, trial_params_for_instantiation,
                                       random_state=RANDOM_SEED, device=device,
                                       **model_init_kwargs)

            # --- Special handling for models needing eval_set or specific train routines ---
            fit_params = {}
            if model_name in ['XGBoost']: # Add LightGBM, CatBoost if used
                 fit_params['eval_set'] = [(X_val_processed.to_numpy(), y_val)] # XGB needs numpy usually
                 fit_params['early_stopping_rounds'] = 10
                 fit_params['verbose'] = False # Suppress verbose output during HPO

            # Handle training for DL models (might need specific trainer loop)
            if model_name in ['TabTransformer', 'SAINT', 'FTTransformer']:
                 # Assuming TabZilla wrappers have a 'fit' that handles train loop
                 # Pass train-specific HPO params here
                 fit_params.update(train_params)
                 # May need to pass validation data for internal monitoring/early stopping
                 fit_params['X_val'] = X_val_processed
                 fit_params['y_val'] = y_val

            # Train the model
            model.fit(X_train_processed, y_train, **fit_params)

            # Evaluate on validation set
            preds = model.predict(X_val_processed)
            score = accuracy_score(y_val, preds) # Use accuracy on validation set

            # Handle NaN scores
            if np.isnan(score):
                logging.warning(f"Trial {trial.number} for {model_name} resulted in NaN score. Returning -1.0.")
                return -1.0

            return score # Optuna maximizes by default

        except Exception as e:
             logging.warning(f"Error during HPO trial {trial.number} for {model_name} with params {hpo_params}: {e}")
             # Log traceback for debugging if needed
             # import traceback
             # logging.warning(traceback.format_exc())
             return -1.0 # Tell Optuna this trial failed

    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
    try:
        study.optimize(objective, n_trials=n_trials, timeout=1800) # Increased timeout (e.g., 30 mins) per HPO run
        best_params = study.best_params
        logging.info(f"HPO for {model_name} complete. Best validation accuracy: {study.best_value:.4f}")
        # Include training params back into best_params if they were optimized
        if model_name in ['TabTransformer', 'SAINT', 'FTTransformer']:
             if 'lr' in study.best_trial.params: best_params['lr'] = study.best_trial.params['lr']
             if 'batch_size' in study.best_trial.params: best_params['batch_size'] = study.best_trial.params['batch_size']
             # if 'epochs' in study.best_trial.params: best_params['epochs'] = study.best_trial.params['epochs']

        logging.info(f"Best HPO params found for {model_name}: {best_params}")
        return best_params
    except Exception as e:
         logging.error(f"Optuna study failed for {model_name}: {e}")
         return {} # Return empty if study crashes


# --- Final Training and Evaluation with Cross-Validation ---
def train_evaluate_final_model(model_name, best_hpo_params, preprocessor,
                               X_train_val, y_train_val, # Unprocessed Train+Val data
                               X_test, y_test,           # Unprocessed Test data
                               feature_info, device='cpu'):
    """Trains final model using best HPO params and 10-fold CV, then evaluates on Test.
       Preprocessor should be fitted on the specific dataset's training data before calling this.
    """
    logging.info(f"Starting final training for {model_name} with 10-fold CV...")

    # Static args for model instantiation
    model_init_kwargs = {
        "cat_dims": feature_info['cat_dims'],
        "num_continuous": len(feature_info['numerical']),
        "d_out": feature_info['num_classes'],
    }

    # Separate training params (like lr, batch_size for DL) from instantiation params
    instantiation_params = best_hpo_params.copy()
    train_params_final = {}
    if model_name in ['TabTransformer', 'SAINT', 'FTTransformer']:
        if 'lr' in instantiation_params: train_params_final['lr'] = instantiation_params.pop('lr')
        if 'batch_size' in instantiation_params: train_params_final['batch_size'] = instantiation_params.pop('batch_size')
        # Add epochs if optimized/needed: train_params_final['epochs'] = instantiation_params.pop('epochs', DEFAULT_EPOCHS)


    # Fit the preprocessor on the *entire* combined Train+Val data for this specific dataset run
    logging.info("Fitting preprocessor on combined Train+Val data for final evaluation...")
    try:
        if preprocessor != 'passthrough':
            preprocessor.fit(X_train_val)
            X_train_val_processed = preprocessor.transform(X_train_val)
            X_test_processed = preprocessor.transform(X_test) # Transform test set too
            logging.info(f"Processed Final Train+Val shape: {X_train_val_processed.shape}, Test shape: {X_test_processed.shape}")
        else:
            X_train_val_processed = X_train_val
            X_test_processed = X_test
            logging.info("Passthrough preprocessor used for final eval.")

        # Final NaN check
        if pd.DataFrame(X_train_val_processed).isnull().values.any() or pd.DataFrame(X_test_processed).isnull().values.any():
            logging.error("NaN values found AFTER final preprocessing. Check imputation/pipeline.")
            X_train_val_processed = pd.DataFrame(X_train_val_processed).fillna(0)
            X_test_processed = pd.DataFrame(X_test_processed).fillna(0)
            logging.warning("Filled remaining NaNs with 0 before final training/evaluation.")

    except Exception as e:
        logging.error(f"Failed to fit/transform preprocessor during final evaluation setup for {model_name}: {e}")
        # Return NaN metrics as preprocessing failed
        return { "cv_accuracy_mean": np.nan, "cv_accuracy_std": np.nan, "cv_f1_mean": np.nan, "cv_f1_std": np.nan,
                 "test_accuracy": np.nan, "test_f1": np.nan, "avg_cv_fold_time_s": 0.0, "final_model_train_time_s": 0.0 }


    # CV setup
    cv = StratifiedKFold(n_splits=N_CV_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    cv_accuracies = []
    cv_f1s = []
    fold_times = []

    logging.info("Starting cross-validation loop...")
    y_train_val_np = np.array(y_train_val) # Ensure numpy for indexing

    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_val_processed, y_train_val_np)):
        fold_start_time = time.time()
        logging.info(f"--- CV Fold {fold + 1}/{N_CV_SPLITS} ---")

        # Get fold data using indices from the *already processed* combined data
        if isinstance(X_train_val_processed, pd.DataFrame):
            X_train_fold = X_train_val_processed.iloc[train_idx]
            X_val_fold = X_train_val_processed.iloc[val_idx]
        else: # Numpy array case
             X_train_fold = X_train_val_processed[train_idx]
             X_val_fold = X_train_val_processed[val_idx]

        y_train_fold = y_train_val_np[train_idx]
        y_val_fold = y_train_val_np[val_idx]

        try:
             # Instantiate model for this fold
             model = get_model_instance(model_name, instantiation_params,
                                        random_state=RANDOM_SEED + fold, # Vary seed per fold
                                        device=device, **model_init_kwargs)

             # Fit model
             fit_params = {}
             if model_name in ['XGBoost']: # Add others if they use eval_set in main fit
                  # Cannot easily use early stopping here without a separate val set per fold
                  # Train for the full duration based on HPO params (e.g., n_estimators)
                  pass
             if model_name in ['TabTransformer', 'SAINT', 'FTTransformer']:
                  fit_params.update(train_params_final)
                  # Don't pass val set for early stopping during CV training, train fully.
                  fit_params.pop('X_val', None)
                  fit_params.pop('y_val', None)

             model.fit(X_train_fold, y_train_fold, **fit_params)

             # Evaluate fold
             preds = model.predict(X_val_fold)
             acc = accuracy_score(y_val_fold, preds)
             f1 = f1_score(y_val_fold, preds, average='weighted')

             if np.isnan(acc) or np.isnan(f1):
                 logging.warning(f"Fold {fold + 1} resulted in NaN metric (Acc: {acc}, F1: {f1}). Storing as NaN.")
                 acc = np.nan
                 f1 = np.nan

             cv_accuracies.append(acc)
             cv_f1s.append(f1)
             logging.info(f"Fold {fold + 1} Val Accuracy: {acc:.4f}, F1: {f1:.4f}")

        except Exception as e:
             logging.error(f"Error during CV Fold {fold + 1} for {model_name}: {e}")
             # import traceback
             # logging.error(traceback.format_exc())
             cv_accuracies.append(np.nan) # Record failure
             cv_f1s.append(np.nan)

        fold_times.append(time.time() - fold_start_time)

    # Calculate average CV performance (ignoring NaNs)
    avg_cv_accuracy = np.nanmean(cv_accuracies) if not all(np.isnan(cv_accuracies)) else np.nan
    cv_accuracy_std = np.nanstd(cv_accuracies) if not all(np.isnan(cv_accuracies)) else np.nan
    avg_cv_f1 = np.nanmean(cv_f1s) if not all(np.isnan(cv_f1s)) else np.nan
    cv_f1_std = np.nanstd(cv_f1s) if not all(np.isnan(cv_f1s)) else np.nan
    avg_fold_time = np.nanmean(fold_times) if fold_times else 0.0
    logging.info(f"Average CV Accuracy: {avg_cv_accuracy:.4f} (+/- {cv_accuracy_std:.4f})")
    logging.info(f"Average CV F1: {avg_cv_f1:.4f} (+/- {cv_f1_std:.4f})")
    logging.info(f"Average Fold Training Time: {avg_fold_time:.2f}s")

    # Final model training on ALL Train+Val data
    logging.info(f"Training final {model_name} model on all Train+Val data...")
    final_model_start_time = time.time()
    try:
        final_model = get_model_instance(model_name, instantiation_params,
                                         random_state=RANDOM_SEED, # Use primary seed
                                         device=device, **model_init_kwargs)

        # Fit final model
        fit_params = {}
        if model_name in ['TabTransformer', 'SAINT', 'FTTransformer']:
             fit_params.update(train_params_final)
             # No validation set needed here
             fit_params.pop('X_val', None)
             fit_params.pop('y_val', None)

        final_model.fit(X_train_val_processed, y_train_val_np, **fit_params) # Use processed data
        final_model_train_time = time.time() - final_model_start_time
        logging.info(f"Final model training time: {final_model_train_time:.2f}s")

        # Evaluate on Test set (already processed)
        logging.info("Evaluating final model on test set...")
        test_preds = final_model.predict(X_test_processed)
        test_accuracy = accuracy_score(y_test, test_preds)
        test_f1 = f1_score(y_test, test_preds, average='weighted')
        logging.info(f"Test Accuracy: {test_accuracy:.4f}")
        logging.info(f"Test F1: {test_f1:.4f}")

        return {
            "cv_accuracy_mean": avg_cv_accuracy,
            "cv_accuracy_std": cv_accuracy_std,
            "cv_f1_mean": avg_cv_f1,
            "cv_f1_std": cv_f1_std,
            "test_accuracy": test_accuracy,
            "test_f1": test_f1,
            "avg_cv_fold_time_s": avg_fold_time,
            "final_model_train_time_s": final_model_train_time,
        }

    except Exception as e:
         logging.error(f"Error during final model training or evaluation for {model_name}: {e}")
         # import traceback
         # logging.error(traceback.format_exc())
         return { # Return results indicating failure, preserving CV results if they exist
             "cv_accuracy_mean": avg_cv_accuracy, "cv_accuracy_std": cv_accuracy_std,
             "cv_f1_mean": avg_cv_f1, "cv_f1_std": cv_f1_std,
             "test_accuracy": np.nan, "test_f1": np.nan,
             "avg_cv_fold_time_s": avg_fold_time,
             "final_model_train_time_s": np.nan,
         }


# --- Main Experiment Runner ---
def main(args):
    overall_start_time = time.time()
    processed_datasets_dir = Path(args.input_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file_path = results_dir / f"experiment_results_{time.strftime('%Y%m%d_%H%M%S')}.csv"

    # Determine device for PyTorch models
    if args.device == 'cuda':
        if TORCH_AVAILABLE and torch.cuda.is_available():
            device = 'cuda'
            logging.info("Using CUDA device.")
        else:
            logging.warning("CUDA requested but not available or PyTorch not installed. Falling back to CPU.")
            device = 'cpu'
    else:
        device = 'cpu'
        logging.info("Using CPU device.")

    # --- Phase 1: HPO on Base Datasets (_00nan) ---
    logging.info("\n===== Phase 1: Hyperparameter Optimization on Base Datasets (_00nan) =====")
    hpo_results_cache = {} # Cache: (dataset_base_name, model_name) -> {params: {}, hpo_time: float}

    # Find all dataset files first to identify base names
    all_dataset_paths = glob.glob(os.path.join(processed_datasets_dir, "*.csv"))
    if not all_dataset_paths:
        logging.error(f"No CSV files found in {processed_datasets_dir}")
        return

    base_dataset_names = set()
    for p in all_dataset_paths:
        match = re.match(r"^(.*?)_(\d+nan)\.csv$", Path(p).name)
        if match:
            base_dataset_names.add(match.group(1))
        else:
             logging.warning(f"Could not parse base name from {Path(p).name}. Skipping.")

    if not base_dataset_names:
         logging.error("Could not identify any base dataset names (expected format 'name_XXnan.csv').")
         return

    logging.info(f"Found {len(base_dataset_names)} unique base datasets for HPO.")

    for base_name in sorted(list(base_dataset_names)):
        base_dataset_path = processed_datasets_dir / f"{base_name}_00nan.csv"
        if not base_dataset_path.exists():
            logging.warning(f"Base dataset file {base_dataset_path} not found. Skipping HPO for {base_name}.")
            continue

        logging.info(f"\n--- HPO for Base Dataset: {base_name} ---")
        hpo_data_timer_start = time.time()

        # Load and split the base (_00nan) data
        split_data = load_and_split_data(str(base_dataset_path), args.target_column)
        if split_data is None:
            logging.error(f"Skipping HPO for {base_name} due to loading/splitting error.")
            continue
        X_train_base, y_train_base, X_val_base, y_val_base, _, _, feature_info_base = split_data
        load_split_time = time.time() - hpo_data_timer_start

        # Create preprocessor (potentially scaling numerics for DL models during HPO)
        # Decide on scaling based on whether *any* DL model is being run
        scale_numeric_hpo = any(m in ['TabTransformer', 'SAINT', 'FTTransformer'] for m in args.models)
        if scale_numeric_hpo:
            logging.info("Enabling StandardScaler for numeric features during HPO due to DL models.")
        # NOTE: Imputation is always done if args.impute is True, even on 00nan data (in case it had original NaNs)
        preprocessor_hpo = create_preprocessor(feature_info_base,
                                               perform_imputation=args.impute,
                                               scale_numeric=scale_numeric_hpo)


        for model_name in args.models:
             # Skip HPO for models if Optuna is not available
             if args.hpo_method == 'optuna' and not OPTUNA_AVAILABLE:
                 logging.warning(f"Skipping HPO for {model_name} on {base_name} as Optuna is not available.")
                 hpo_results_cache[(base_name, model_name)] = {'params': {}, 'hpo_time': 0.0}
                 continue
             # Skip HPO if method is 'none'
             if args.hpo_method == 'none':
                 logging.info(f"Skipping HPO for {model_name} on {base_name} as HPO method is 'none'.")
                 hpo_results_cache[(base_name, model_name)] = {'params': {}, 'hpo_time': 0.0}
                 continue
             # Grid search not implemented
             if args.hpo_method == 'grid':
                  logging.warning(f"Skipping HPO for {model_name} on {base_name}. Grid search not implemented.")
                  hpo_results_cache[(base_name, model_name)] = {'params': {}, 'hpo_time': 0.0}
                  continue


             logging.info(f"--- Starting HPO for Model: {model_name} on Base Dataset: {base_name} ---")
             hpo_timer_start = time.time()

             # Run HPO function
             best_hpo_params = run_hpo(
                 model_name=model_name,
                 preprocessor=preprocessor_hpo, # Pass the created (but not yet fitted) preprocessor
                 X_train=X_train_base,
                 y_train=y_train_base,
                 X_val=X_val_base,
                 y_val=y_val_base,
                 feature_info=feature_info_base,
                 n_trials=args.hpo_trials,
                 device=device
             )

             hpo_time = time.time() - hpo_timer_start
             logging.info(f"HPO Time for {model_name} on {base_name}: {hpo_time:.2f}s")

             # Store results
             hpo_results_cache[(base_name, model_name)] = {
                 'params': best_hpo_params,
                 'hpo_time': hpo_time
             }


    # --- Phase 2: Evaluation on All Datasets using Cached HPO Params ---
    logging.info("\n===== Phase 2: Evaluation on All Datasets using Cached HPO Params =====")
    all_run_results = []
    fieldnames = [
        "Dataset", "NaN_Level", "Model", "Imputation", "HPO_Method",
        "HPO_Trials", "HPO_Base_Time_s", # Renamed from HPO_Time_s
        "Best_Params", "CV_Accuracy_Mean", "CV_Accuracy_Std", "CV_F1_Mean", "CV_F1_Std",
        "Avg_CV_Fold_Time_s", "Final_Model_Train_Time_s",
        "Test_Accuracy", "Test_F1", "Total_Eval_Time_s", # Renamed from Total_Model_Time_s
        "Error"
    ]

    for dataset_path in sorted(all_dataset_paths): # Iterate through ALL files now
        dataset_file_name = Path(dataset_path).name
        logging.info(f"\n===== Evaluating Dataset: {dataset_file_name} =====")
        eval_timer_start = time.time()

        # Extract dataset base name and NaN level
        match = re.match(r"^(.*?)_(\d+nan)\.csv$", dataset_file_name)
        if not match:
            logging.warning(f"Could not parse dataset info from {dataset_file_name}. Skipping evaluation.")
            continue
        dataset_base_name = match.group(1)
        nan_level = match.group(2)

        # 1. Load and Split Data for this specific dataset version
        split_data = load_and_split_data(dataset_path, args.target_column)
        if split_data is None:
            logging.error(f"Skipping evaluation for {dataset_file_name} due to loading/splitting error.")
            result = defaultdict(lambda: None) # Use defaultdict for cleaner missing value handling
            result.update({"Dataset": dataset_base_name, "NaN_Level": nan_level, "Error": "Load/Split Failed"})
            all_run_results.append(dict(result))
            continue
        X_train, y_train, X_val, y_val, X_test, y_test, feature_info = split_data


        # 2. Loop through models for evaluation
        for model_name in args.models:
            model_eval_timer_start = time.time()
            logging.info(f"\n--- Evaluating Model: {model_name} on {dataset_file_name} ---")

            # 2a. Retrieve cached HPO parameters
            cache_key = (dataset_base_name, model_name)
            if cache_key not in hpo_results_cache:
                logging.error(f"HPO results not found for {dataset_base_name} / {model_name}. Skipping evaluation.")
                result = defaultdict(lambda: None)
                result.update({"Dataset": dataset_base_name, "NaN_Level": nan_level, "Model": model_name, "Error": "HPO Results Missing"})
                all_run_results.append(dict(result))
                continue

            cached_hpo_info = hpo_results_cache[cache_key]
            best_hpo_params = cached_hpo_info['params']
            hpo_base_time = cached_hpo_info['hpo_time'] # Time taken during Phase 1 for this combo

            logging.info(f"Using cached HPO params: {best_hpo_params}")

            final_metrics = {}
            model_error = None
            total_eval_time = 0.0

            try:
                # 2b. Create and fit preprocessor SPECIFICALLY for this dataset's training data
                # Use scaling if it was used during HPO for consistency (esp. for DL models)
                scale_numeric_eval = any(m in ['TabTransformer', 'SAINT', 'FTTransformer'] for m in args.models)
                preprocessor_eval = create_preprocessor(feature_info,
                                                        perform_imputation=args.impute,
                                                        scale_numeric=scale_numeric_eval)


                # 2c. Final Training (CV) and Evaluation
                # Combine Train and Val sets from THIS specific dataset split
                X_train_val_comb = pd.concat([X_train, X_val], ignore_index=True)
                y_train_np = np.array(y_train)
                y_val_np = np.array(y_val)
                y_train_val_comb = np.concatenate([y_train_np, y_val_np])

                final_metrics = train_evaluate_final_model(
                    model_name=model_name,
                    best_hpo_params=best_hpo_params,
                    preprocessor=preprocessor_eval, # Pass the unfitted preprocessor - it fits inside
                    X_train_val=X_train_val_comb, # Pass combined unprocessed data
                    y_train_val=y_train_val_comb, # Combined labels
                    X_test=X_test, # Unprocessed test features
                    y_test=y_test, # Test labels
                    feature_info=feature_info,
                    device=device
                )

            except Exception as e:
                logging.error(f"Failed evaluation experiment for Model: {model_name}, Dataset: {dataset_file_name}. Error: {e}")
                # import traceback
                # logging.error(traceback.format_exc())
                model_error = f"Evaluation Failed: {e}"


            # 3. Record Results
            total_eval_time = time.time() - model_eval_timer_start
            result = defaultdict(lambda: None) # Use defaultdict
            result.update({
                "Dataset": dataset_base_name,
                "NaN_Level": nan_level,
                "Model": model_name,
                "Imputation": args.impute,
                "HPO_Method": args.hpo_method,
                "HPO_Trials": args.hpo_trials if args.hpo_method == 'optuna' else 0,
                "HPO_Base_Time_s": round(hpo_base_time, 2),
                "Best_Params": str(best_hpo_params),
                "CV_Accuracy_Mean": final_metrics.get("cv_accuracy_mean"),
                "CV_Accuracy_Std": final_metrics.get("cv_accuracy_std"),
                "CV_F1_Mean": final_metrics.get("cv_f1_mean"),
                "CV_F1_Std": final_metrics.get("cv_f1_std"),
                "Avg_CV_Fold_Time_s": final_metrics.get("avg_cv_fold_time_s"),
                "Final_Model_Train_Time_s": final_metrics.get("final_model_train_time_s"),
                "Test_Accuracy": final_metrics.get("test_accuracy"),
                "Test_F1": final_metrics.get("test_f1"),
                "Total_Eval_Time_s": round(total_eval_time, 2), # This is time for CV + final train + test eval
                "Error": model_error
            })

            # Format None/NaN and round floats
            for key, value in result.items():
                 if isinstance(value, (float, np.floating)):
                     result[key] = round(value, 5) if pd.notna(value) else None
                 elif pd.isna(value): # Handle other potential NaN types if necessary
                      result[key] = None

            all_run_results.append(dict(result)) # Convert back to dict for csv writer

            # --- Incremental Save ---
            try:
                 # Use mode 'a' (append) after the first write, or just keep writing header
                 write_header = not os.path.exists(results_file_path) or os.path.getsize(results_file_path) == 0
                 with open(results_file_path, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    if write_header:
                        writer.writeheader()
                    # Write only the last result dictionary
                    writer.writerow(all_run_results[-1])
                 # logging.info(f"Incrementally saved results for {model_name} to {results_file_path}")
            except IOError as e:
                 logging.error(f"Could not write incremental results to {results_file_path}: {e}")
            # End model loop

        dataset_eval_time = time.time() - eval_timer_start
        logging.info(f"===== Dataset {dataset_file_name} evaluation time: {dataset_eval_time:.2f}s =====")
        # End dataset loop (evaluation phase)


    # Final summary
    logging.info(f"\n--- Experiment complete. Results saved to {results_file_path} ---")
    overall_time = time.time() - overall_start_time
    logging.info(f"Total execution time: {overall_time:.2f} seconds ({overall_time/60:.2f} minutes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run custom TabZilla model experiments with cached HPO.")
    parser.add_argument("--input_dir", type=str, default="processed_datasets", help="Directory containing processed (NaN-injected) CSV datasets.")
    # Removed dataset_pattern, now processes all found datasets
    parser.add_argument("--results_dir", type=str, default="results", help="Directory to save experiment results.")
    parser.add_argument("--target_column", type=str, required=True, help="Name of the target variable column.")
    parser.add_argument("--models", nargs='+', default=MODELS_TO_RUN, help=f"List of TabZilla models to run (default: {MODELS_TO_RUN}).")
    parser.add_argument("--impute", action='store_true', help="Enable simple imputation (median/most_frequent) for missing values.")
    parser.add_argument("--hpo_method", type=str, default="optuna", choices=['optuna', 'grid', 'none'], help="Hyperparameter optimization method (optuna [recommended], grid [not implemented], none). HPO runs only on _00nan datasets.")
    parser.add_argument("--hpo_trials", type=int, default=25, help="Number of trials for Optuna HPO (if used).")
    parser.add_argument("--device", type=str, default="cpu", choices=['cpu', 'cuda'], help="Device to use for PyTorch models (cpu or cuda).")

    args = parser.parse_args()

    # Validate dependencies based on selected models
    run_dl_models = any(m in ['TabTransformer', 'SAINT', 'FTTransformer'] for m in args.models)
    if run_dl_models and not TORCH_AVAILABLE:
        logging.error("PyTorch is required for TabTransformer, SAINT, or FTTransformer but not installed. Aborting.")
        sys.exit(1)
    if args.hpo_method == 'optuna' and not OPTUNA_AVAILABLE:
        logging.warning("Optuna selected but not installed. Switching HPO method to 'none'.")
        args.hpo_method = 'none'
    if args.hpo_method == 'grid':
         logging.warning("Grid search selected but not implemented. Switching HPO method to 'none'.")
         args.hpo_method = 'none'


    # Add warnings about preprocessing for DL models
    if run_dl_models:
         logging.warning("="*60)
         logging.warning("Running Deep Learning Models (TabTransformer, SAINT, FTTransformer)")
         logging.warning("The current script uses basic OneHotEncoding for categoricals and optional StandardScaler for numericals.")
         logging.warning("These models often perform better with specialized preprocessing:")
         logging.warning("  - Numerical features: Usually require scaling (StandardScaler enabled if DL models run).")
         logging.warning("  - Categorical features: Often require LabelEncoding first, then handled via embeddings within the model. The current OneHotEncoding might be suboptimal or incompatible depending on the specific TabZilla model wrapper implementation.")
         logging.warning("  - Feature info: Ensure `get_model_instance` correctly receives necessary info like feature cardinalities ('cat_dims'), number of continuous features, and output dimension ('d_out').")
         logging.warning("Review the specific model's requirements in TabZilla and adjust preprocessing/instantiation if needed.")
         logging.warning("="*60)


    main(args)