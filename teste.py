#!/usr/bin/env python
"""null_robustness_tabzilla.py – benchmark de valores nulos com barras de progresso (tqdm)"""

import argparse, sys, math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from tqdm import tqdm  # barras de progresso

# -----------------------------------------------------------------------------
# Caminho do TabZilla
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
TABZILLA = ROOT / "TabZilla"
if not TABZILLA.exists():
    raise SystemExit("Este script deve ficar dentro do diretório tabzilla-main.")

sys.path.append(str(TABZILLA))
from tabzilla_alg_handler import get_model, ALL_MODELS  # noqa: E402

# Modelos avaliados -----------------------------------------------------------
TREE_MODELS = ["DecisionTree", "RandomForest", "XGBoost", "LightGBM", "CatBoost"]
TRANS_MODELS = ["SAINT", "TabTransformer", "DANet", "rtdl_FTTransformer"]
MODELS = TREE_MODELS + TRANS_MODELS

# Estimadores sklearn que NÃO aceitam NaN
NAN_UNSAFE = {"DecisionTree", "RandomForest"}

# -----------------------------------------------------------------------------
# Utilidades
# -----------------------------------------------------------------------------
def load_dataset(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    if p.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(p)
    raise ValueError("Formato não suportado: " + p.suffix)


def add_nan(df: pd.DataFrame, target: str, ratio: float, seed: int) -> pd.DataFrame:
    """Injeta NaNs aleatoriamente nas features."""
    if ratio == 0.0:
        return df.copy()
    rng = np.random.default_rng(seed)
    out = df.copy()
    feat_cols = out.columns.difference([target])
    n = int(len(out) * ratio)
    for col in feat_cols:
        idx = rng.choice(len(out), n, replace=False)
        out.loc[idx, col] = np.nan
    return out


def encode_cats(df: pd.DataFrame, cat_cols: list[str]):
    """Label-encode colunas categóricas (NaNs viram 'Missing')."""
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str).fillna("Missing"))
    return df


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
import numpy as np

def metric_classification(y_true, y_pred, prob=None):
    m = {
        "acc" : accuracy_score(y_true, y_pred),
        "prec": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "rec" : recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1"  : f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    # Se for binário, adiciona F1 por classe
    if len(np.unique(y_true)) == 2:
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        m["f1_0"] = f1_per_class[0]
        m["f1_1"] = f1_per_class[1]

    # AUC só para binário quando prob disponível
    if prob is not None and len(np.unique(y_true)) == 2:
        prob = prob[:, -1] if prob.ndim == 2 else prob
        m["auc"] = roc_auc_score(y_true, prob.ravel())

    return m


def task_info(y: pd.Series):
    """Regressão, classificação multi-classe ou binária (1-logit)."""
    if pd.api.types.is_numeric_dtype(y):
        if y.nunique() == 2:
            return "binary", 1
        if y.nunique() <= 10:
            return "classification", y.nunique()
        return "regression", 1
    return ("binary" if y.nunique() == 2 else "classification", y.nunique())


# -----------------------------------------------------------------------------
# Pipeline principal
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument(
        "--missing_ratios",
        type=float,
        nargs="*",
        default=[0.0, 0.25, 0.5, 0.75],
        help="Proporções de NaNs a injetar (0-1)",
    )
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    df_raw = load_dataset(args.data_path)
    if args.target not in df_raw.columns:
        raise SystemExit("Target não encontrado no dataset.")

    cat_cols = [
        c for c in df_raw.columns if c != args.target and df_raw[c].dtype == "object"
    ]
    num_features = df_raw.shape[1] - 1  # sem a label

    results = []
    dataset_name = Path(args.data_path).stem  # ex.: cardio_tabzilla

    for ratio in tqdm(args.missing_ratios, desc="% NaN"):
        df = add_nan(df_raw, args.target, ratio, args.random_state)
        df = encode_cats(df, cat_cols)

        y = df[args.target].values
        X = df.drop(columns=[args.target]).values.astype(np.float32)
        task, n_cls = task_info(df[args.target])

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            stratify=y if task != "regression" else None,
            random_state=args.random_state,
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.2,
            stratify=y_train if task != "regression" else None,
            random_state=args.random_state,
        )

        # ---------------------------------------------------------------------
        # Argumentos base (comuns a TODOS os modelos)
        # ---------------------------------------------------------------------
        base_args = SimpleNamespace(
            dataset=dataset_name,              # usado em io_utils
            objective=task,
            cat_idx=[df.columns.get_loc(c) for c in cat_cols],
            num_features=num_features,
            cat_dims=[df_raw[c].nunique() for c in cat_cols],
            num_classes=n_cls,                 # 1 logit em binário
            batch_size=256,
            val_batch_size=512,
            epochs=15,
            early_stopping_rounds=10,
            logging_period=20,
            use_gpu=True,
            gpu_ids=[0],
            data_parallel=True,
        )

        # ----------------------------- loop de modelos -----------------------
        for model_name in tqdm(MODELS, desc="Modelos", leave=False):
            if model_name not in ALL_MODELS:
                continue

            # SimpleNamespace precisa da chave model_name
            base_args.model_name = model_name

            Model = get_model(model_name)

            # Imputação se o estimador não aceita NaN
            if ratio > 0 and model_name in NAN_UNSAFE:
                imp = SimpleImputer(strategy="median")
                X_train = imp.fit_transform(X_train)
                X_val = imp.transform(X_val)
                X_test = imp.transform(X_test)

            try:
                params = (
                    Model.default_parameters() if hasattr(Model, "default_parameters") else {}
                )
                model = Model(params, base_args)
                model.fit(X_train, y_train, X_val, y_val)
                preds, prob = model.predict(X_test)

                if task == "regression":
                    res = {"rmse": math.sqrt(((preds - y_test) ** 2).mean())}
                else:
                    res = metric_classification(
                        y_test,
                        preds,
                        None if prob.size == 0 else prob,
                    )
                res.update({"model": model_name, "ratio": ratio})
                results.append(res)

            except Exception as e:
                tqdm.write(f"[WARN] {model_name} ratio={ratio} falhou: {e}")

    out_csv = Path(args.data_path).with_suffix("").with_name(
        "null_benchmark_summary.csv"
    )
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print("\nResultados salvos em:", out_csv)


if __name__ == "__main__":
    main()
