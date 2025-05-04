# preprocess_data.py  (REV 2025‑05‑04)
import pandas as pd
import numpy as np
import os, glob, logging, json
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder

# ──────────────────────────── CONFIG ──────────────────────────── #
INPUT_DIR       = Path("datasets_raw")            # ⇦ edite aqui
OUTPUT_DIR      = Path("processed_datasets")  # ⇦ edite aqui
TARGET_COLUMN   = "class"                     # ⇦ nome da coluna‑alvo
NAN_PERCENTAGES = [0.25, 0.50, 0.75]          # 0 % já é salvo como _00nan
RANDOM_SEED     = 42                          # semente global
VAL_SIZE        = 0.10                        # fração do total
TEST_SIZE       = 0.10                        # fração do total
# (train = 1 – VAL – TEST = 0.80)
# ───────────────────────────────────────────────────────────────── #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

def inject_nans(df: pd.DataFrame, pct: float, target: str, *, seed: int) -> pd.DataFrame | None:
    """Insere NaNs em pct (%) das linhas de cada feature."""
    if pct <= 0:
        return df.copy()
    rng = np.random.default_rng(seed)
    df_nan = df.copy()
    features = [c for c in df.columns if c != target]

    n_rows      = len(df_nan)
    n_inject    = int(round(n_rows * pct))

    if n_inject == 0:
        logging.warning(
            f"Tamanho insuficiente ({n_rows} linhas) para injetar {pct*100:.1f}% de NaNs."
        )
        return None

    for col in features:
        non_nan_idx = df_nan.index[df_nan[col].notna()]
        n_here = min(n_inject, len(non_nan_idx))
        if n_here:
            to_nan = rng.choice(non_nan_idx, size=n_here, replace=False)
            df_nan.loc[to_nan, col] = np.nan
    return df_nan

# ───────────────────────────── SPLIT ──────────────────────────── #
from typing import Any

def _json_key(x: Any) -> str | int | float | bool | None:
    """Converte chaves numpy/objetos para algo serializável em JSON."""
    # numpy escalares → Python nativo
    if isinstance(x, (np.generic, np.bool_)):
        return x.item()          # int/float/bool
    return str(x)                # fallback seguro

# ───────────────────────────── SPLIT ──────────────────────────── #
def stratified_split_indices(df: pd.DataFrame, target: str) -> dict | None:
    if target not in df.columns:
        logging.error(f"Coluna alvo '{target}' não encontrada.")
        return None

    X = df.drop(columns=[target]).values
    y = df[target].values
    le = LabelEncoder().fit(y)
    y_enc = le.transform(y)

    if len(np.unique(y_enc)) < 2:
        logging.error("É necessário ≥2 classes para estratificação.")
        return None

    sss1 = StratifiedShuffleSplit(
        n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    train_val_idx, test_idx = next(sss1.split(X, y_enc))

    val_fraction = VAL_SIZE / (1.0 - TEST_SIZE)
    sss2 = StratifiedShuffleSplit(
        n_splits=1, test_size=val_fraction, random_state=RANDOM_SEED
    )
    train_idx, val_idx = next(sss2.split(X[train_val_idx], y_enc[train_val_idx]))
    train_idx = train_val_idx[train_idx]
    val_idx   = train_val_idx[val_idx]

    def proportions(idx):
        vals, counts = np.unique(y_enc[idx], return_counts=True)
        total = len(idx)
        return {
            _json_key(le.inverse_transform([v])[0]): (counts[i] / total)
            for i, v in enumerate(vals)
        }

    return {
        "train_indices": train_idx.tolist(),
        "val_indices"  : val_idx.tolist(),
        "test_indices" : test_idx.tolist(),
        "class_proportions": {
            "train": proportions(train_idx),
            "val"  : proportions(val_idx),
            "test" : proportions(test_idx),
        }
    }

# ──────────────────────────── MAIN LOOP ────────────────────────── #
def main() -> None:
    if not INPUT_DIR.is_dir():
        logging.error(f"Pasta de entrada não encontrada: {INPUT_DIR}")
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_paths = glob.glob(str(INPUT_DIR / "*.csv"))
    if not csv_paths:
        logging.error(f"Nenhum CSV encontrado em {INPUT_DIR}")
        return

    logging.info(f"Processando {len(csv_paths)} dataset(s) de {INPUT_DIR} …")

    for csv_path in csv_paths:
        name = Path(csv_path).stem
        logging.info(f"\n─── {name} ─────────────────────────────────────────")
        df = pd.read_csv(csv_path)
        if df.empty:
            logging.warning("Dataset vazio; pulando.")
            continue

        split = stratified_split_indices(df, TARGET_COLUMN)
        if split is None:
            logging.error("Falha no split; pulando injeção de NaNs.")
            continue

        # Salva índices + distribuições
        (OUTPUT_DIR / "splits").mkdir(exist_ok=True)
        split_path = OUTPUT_DIR / "splits" / f"{name}_split.json"
        with open(split_path, "w") as fp:
            json.dump(split, fp, indent=4)
        logging.info(f"Índices e proporções → {split_path}")

        # Salva versão 0 % NaN
        base_out = OUTPUT_DIR / f"{name}_00nan.csv"
        df.to_csv(base_out, index=False)
        logging.info(f"Arquivo original → {base_out}")

        # Outras porcentagens
        for pct in NAN_PERCENTAGES:
            df_nan = inject_nans(df, pct, TARGET_COLUMN, seed=RANDOM_SEED + int(pct*100))
            if df_nan is None:
                continue
            pct_tag = f"{int(pct*100):02d}nan"
            out_path = OUTPUT_DIR / f"{name}_{pct_tag}.csv"
            df_nan.to_csv(out_path, index=False)
            logging.info(f"{pct*100:.0f}% NaN → {out_path}")

    logging.info("\n✔ Pré‑processamento concluído.")

# ----------------------------------------------------------------- #
if __name__ == "__main__":
    main()
