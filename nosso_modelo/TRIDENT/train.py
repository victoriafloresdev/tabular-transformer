import os
import json
import copy
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

from utils import set_global_seed, preprocess_table
from embedder import TabularEmbedder
from transformer import TabularTransformerEncoder
from models import TabularBertPretrainer, TabularBERTModel

def main():
    # -----------------------------------------------------------------
    # Carrega os splits previamente gerados
    # -----------------------------------------------------------------
    set_global_seed()
    
    with open(Path("/scratch/guilherme.evangelista/tabular-transformer/nosso_modelo/processed_datasets/splits/vehicle_split.json")) as f:
        splits = json.load(f)
    
    df = pd.read_csv("/scratch/guilherme.evangelista/tabular-transformer/nosso_modelo/processed_datasets/vehicle_60nan.csv")
    categorical_columns = []
    DIM = 512
    HIDDEN_DIM = 256
    
    HEADS = 6
    LAYERS = 10
    DIM_FEED = 512
    DROPOUT = 0.0
    
    EPOCHS_PRE = 50
    BATCH = 256
    
    LR_PRE = 0.000105
    WEIGHT_DECAY_PRE = 0.0
    
    PROB_MASCARA = 0.2
    
    EPOCH_FINE = 50
    LR_FINE = 0.0001
    WEIGHT_DECAY_FINE = 0.0
    
    LABELS = 4
    # ===============================================================
    # 0) Configurações gerais
    # ===============================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Usando device:", device)
    
    label_column = "class"

    label_encoder = LabelEncoder()
    df[label_column] = label_encoder.fit_transform(df[label_column])

    # --- mostra o mapeamento:
    print("=== Mapeamento de labels ===")
    for idx, cls in enumerate(label_encoder.classes_):
        print(f"{idx} → {cls}")
    print("============================")

    all_cols = df.columns.tolist()
    all_cols.remove(label_column)
    numerical_columns = [c for c in all_cols if c not in categorical_columns]
    
    # ⇣⇣  Normaliza atributos contínuos  ⇣⇣
    if numerical_columns:
        scaler = StandardScaler()
        df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    else:
        scaler = None

    # ===============================================================
    # 1) Dataset para PRÉ‑TREINAMENTO ― **somente features** (sem label)
    # ===============================================================
    train_idx = np.array(splits["train_indices"])
    val_idx   = np.array(splits["val_indices"])
    test_idx  = np.array(splits["test_indices"])
    
    df_pretrain = df.drop(columns=[label_column]).copy()

    df_train_orig = df_pretrain.iloc[train_idx].reset_index(drop=True)
    df_val_orig   = df_pretrain.iloc[val_idx].reset_index(drop=True)

    # ===============================================================
    # 2) Modelo de pré‑treino
    # ===============================================================
    embedder = TabularEmbedder(
        df=df_pretrain,
        categorical_columns=categorical_columns,
        numerical_columns=numerical_columns,
        dimensao=DIM,
        hidden_dim=HIDDEN_DIM
    )

    transformer = TabularTransformerEncoder(
        d_model=embedder.dimensao,
        nhead=HEADS,
        num_layers=LAYERS,
        dim_feedforward=DIM_FEED,
        dropout=DROPOUT
    )

    pretrain_model = TabularBertPretrainer(embedder, transformer).to(device)

    def _init(m):
        if isinstance(m, (nn.Embedding, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)

    pretrain_model.apply(_init)

    optimizer = optim.AdamW(
        pretrain_model.parameters(), lr=LR_PRE, weight_decay=WEIGHT_DECAY_PRE)
    num_epochs = EPOCHS_PRE
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs)

    batch_size = BATCH

    train_losses_history = []
    val_losses_history = []

    # ===============================================================
    # 3) LOOP de PRÉ‑TREINAMENTO com MÁSCARAS DINÂMICAS
    # ===============================================================
    print("\n=== Iniciando Pré‑Treinamento (máscaras novas a cada época) ===")
    for epoch in range(num_epochs):
        # ---------- (re)gera novas máscaras para TODA a base ----------
        df_train_masked = preprocess_table(
            df_train_orig.copy(), p_base=PROB_MASCARA, fine_tunning=False)
        df_val_masked = preprocess_table(
            df_val_orig.copy(), p_base=PROB_MASCARA, fine_tunning=False)

        df_train_size = len(df_train_masked)
        df_val_size = len(df_val_masked)

        # ---------- Treino ----------
        pretrain_model.train()
        indices = torch.randperm(df_train_size)
        train_loss_sum = 0.0
        train_steps = 0

        for start in tqdm(range(0, df_train_size, batch_size),
                          desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch"):
            end = start + batch_size
            batch_idx = indices[start:end]

            batch_masked = df_train_masked.iloc[batch_idx].reset_index(drop=True)
            batch_orig   = df_train_orig.iloc[batch_idx].reset_index(drop=True)

            optimizer.zero_grad()
            total_loss, _ = pretrain_model(batch_masked, batch_orig)
            total_loss.backward()
            optimizer.step()
            scheduler.step() 

            train_loss_sum += total_loss.item()
            train_steps += 1

        avg_train_loss = train_loss_sum / train_steps
        train_losses_history.append(avg_train_loss)

        # ---------- Validação ----------
        pretrain_model.eval()
        with torch.no_grad():
            val_loss_sum = 0.0
            val_steps = 0
            for start in range(0, df_val_size, batch_size):
                end = start + batch_size
                batch_masked = df_val_masked.iloc[start:end].reset_index(drop=True)
                batch_orig   = df_val_orig.iloc[start:end].reset_index(drop=True)

                total_loss, _ = pretrain_model(batch_masked, batch_orig)
                val_loss_sum += total_loss.item()
                val_steps += 1

            avg_val_loss = val_loss_sum / val_steps
            val_losses_history.append(avg_val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}]  Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")

    # ---- plot das curvas de perda ----
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses_history, label='Train Loss')
    plt.plot(val_losses_history, label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Pré‑Treino MLM')
    plt.legend(); plt.show()

    # =================================================
    # 4) FINE TUNING (Classificação)
    # =================================================
    print("\n=== Iniciando Fine-Tuning (Exemplo de Classificação) ===")
    
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val   = df.iloc[val_idx].reset_index(drop=True)
    df_test  = df.iloc[test_idx].reset_index(drop=True)

    y_train = torch.tensor(df_train[label_column].values, dtype=torch.long, device=device)
    y_val   = torch.tensor(df_val[label_column].values,   dtype=torch.long, device=device)
    y_test  = torch.tensor(df_test[label_column].values,  dtype=torch.long, device=device)

    # Drop da label para criar DataFrames de features
    df_train_features = df_train.drop(columns=[label_column])
    df_val_features   = df_val.drop(columns=[label_column])
    df_test_features  = df_test.drop(columns=[label_column])

    # Preprocessar sem máscaras para fine-tuning (p_base=0.0, fine_tunning=True)
    processed_train = preprocess_table(df_train_features, p_base=0.0, fine_tunning=True)
    processed_val   = preprocess_table(df_val_features,   p_base=0.0, fine_tunning=True)
    processed_test  = preprocess_table(df_test_features,  p_base=0.0, fine_tunning=True)

    # Reaproveitamos embedder e transformer do pretrain_model
    finetune_embedder = pretrain_model.embedder
    finetune_transformer = pretrain_model.transformer

    # Calcula pesos de classe (opcional)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train.cpu().numpy()),
        y=y_train.cpu().numpy()
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)

    # Constrói o modelo de classificação com os módulos pré-treinados
    classifier_model = TabularBERTModel(
        embedder=finetune_embedder,
        transformer=finetune_transformer,
        num_labels=LABELS,
        class_weights=class_weights
    ).to(device)

    # Otimizador para fine-tuning
    optimizer_ft = optim.AdamW(classifier_model.parameters(), lr=LR_FINE, weight_decay=WEIGHT_DECAY_FINE)
    num_epochs_ft = EPOCH_FINE
    batch_size_ft = BATCH
    scheduler_ft = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_ft, T_max=num_epochs_ft)
    
    train_losses_history = []
    val_losses_history = []

    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(num_epochs_ft):
        classifier_model.train()
        indices = torch.randperm(len(processed_train))
        train_loss_sum = 0.0
        train_batch_count = 0

        for start in tqdm(range(0, len(processed_train), batch_size_ft),
                          desc=f"FineTune Epoch {epoch+1}/{num_epochs_ft}",
                          unit="batch"):
            end = start + batch_size_ft
            batch_idx = indices[start:end]

            batch_df = processed_train.iloc[batch_idx]
            batch_labels = y_train[batch_idx]

            logits_batch, loss_batch = classifier_model(batch_df, labels=batch_labels)

            optimizer_ft.zero_grad()
            loss_batch.backward()

            # Clipping
            optimizer_ft.step()
            scheduler_ft.step()
            train_loss_sum += loss_batch.item()
            train_batch_count += 1

        train_loss_epoch = train_loss_sum / train_batch_count
        train_losses_history.append(train_loss_epoch)

        # Validação
        classifier_model.eval()
        with torch.no_grad():
            logits_val, val_loss = classifier_model(processed_val, labels=y_val)
            preds_val = torch.argmax(logits_val, dim=1)

        val_loss_value = val_loss.item()
        val_losses_history.append(val_loss_value)

        preds_val_np = preds_val.cpu().numpy()
        y_val_np = y_val.cpu().numpy()

        # Calcula F1-score para cada classe individualmente
        f1_val_per_class = f1_score(y_val_np, preds_val_np, average=None)
        f1_val_micro = f1_score(y_val_np, preds_val_np, average='micro')
        f1_val_macro = f1_score(y_val_np, preds_val_np, average='macro')

        print(f"\n[Epoch {epoch+1}/{num_epochs_ft} Summary]")
        print(f"  Train Loss = {train_loss_epoch:.4f}")
        print(f"  Val   Loss = {val_loss_value:.4f}")
        print(f"  Val   F1 (micro) = {f1_val_micro:.4f}")
        print(f"  Val   F1 (macro) = {f1_val_macro:.4f}")
        print(f"  Val   F1 por classe: {f1_val_per_class}")
        
        if val_loss_value < best_val_loss:
            best_val_loss = val_loss_value
            best_model_state = copy.deepcopy(classifier_model.state_dict())

        print(f"[Epoch {epoch+1}] train={train_loss_epoch:.4f} "
              f"(best={best_val_loss:.4f}) | val={val_loss_value:.4f}")

    if best_model_state is not None:
        classifier_model.load_state_dict(best_model_state)

    # Avaliação no conjunto de teste
    classifier_model.eval()
    with torch.no_grad():
        logits_test, test_loss = classifier_model(processed_test, labels=y_test)
        preds_test = torch.argmax(logits_test, dim=1)

    preds_test_np = preds_test.cpu().numpy()
    y_test_np     = y_test.cpu().numpy()

    acc_test = accuracy_score(y_test_np, preds_test_np)
    f1_test_per_class = f1_score(y_test_np, preds_test_np, average=None)
    f1_test_micro     = f1_score(y_test_np, preds_test_np, average='micro')
    f1_test_macro     = f1_score(y_test_np, preds_test_np, average='macro')

    print("\n=== Resultados no Conjunto de Teste ===")
    print(f"Acurácia geral: {acc_test:.4f}")
    print(f"F1 micro:       {f1_test_micro:.4f}")
    print(f"F1 macro:       {f1_test_macro:.4f}")
    print("\n--- F1 por classe (rótulo original) ---")
    for cls, f1 in zip(label_encoder.classes_, f1_test_per_class):
        print(f"F1({cls}): {f1:.4f}")
    print("----------------------------------------")


if __name__ == "__main__":
    main() 