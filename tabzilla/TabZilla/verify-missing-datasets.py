import numpy as np
import pathlib
import gzip
import json
import collections.abc
import traceback
import argparse

# --- Configurações ---
SCRIPT_VERSION = "1.0-Verifier"
# Lista dos nomes base dos datasets ORIGINAIS (ex: 'openml__credit-g__31')
# O script adicionará automaticamente _100pct para o original e os sufixos de % para os novos
datasets_to_verify = [
    'openml__credit-g__31',
    'openml__diabetes__37',
    'openml__adult__7592',
    'openml__spambase__43',
    'openml__Amazon_employee_access__34539'
]
percentages_to_verify = [25, 50, 75] # Percentagens (ex: 25 para 25%)
base_dir = pathlib.Path('./datasets') # Ajuste se necessário
original_suffix = '_100pct'
missing_suffix_template = '_{percentage}pct_missing'
# Tolerância para a verificação da porcentagem de NaNs (ex: 0.05 = 5% de diferença absoluta)
NAN_PERCENTAGE_TOLERANCE = 0.05
# ---------------------

def load_npy_gz(file_path, allow_pickle=True):
    """Carrega um arquivo .npy.gz de forma segura."""
    if not file_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")
    try:
        with gzip.GzipFile(file_path, 'rb') as f:
            # Tenta carregar permitindo pickle por causa dos datasets problemáticos
            return np.load(f, allow_pickle=allow_pickle)
    except Exception as e:
        print(f"Erro ao carregar {file_path}: {e}")
        raise # Re-levanta a exceção para indicar falha no carregamento

def compare_splits(splits1, splits2):
    """Compara estruturas de split, lidando com dicionários e sequências."""
    try:
        if len(splits1) != len(splits2): return False
        for fold1, fold2 in zip(splits1, splits2):
            # Caso dicionário (ex: spambase)
            if isinstance(fold1, collections.abc.Mapping) and isinstance(fold2, collections.abc.Mapping):
                if sorted(fold1.keys()) != sorted(fold2.keys()): return False
                for key in fold1:
                    # Compara como arrays numpy para garantir igualdade de conteúdo e tipo
                    if not np.array_equal(np.array(fold1[key]), np.array(fold2[key])): return False
            # Caso sequência (lista/tupla/array)
            elif isinstance(fold1, collections.abc.Sequence) and isinstance(fold2, collections.abc.Sequence):
                 if len(fold1) != len(fold2): return False
                 for i in range(len(fold1)):
                      if not np.array_equal(np.array(fold1[i]), np.array(fold2[i])): return False
            # Caso os tipos não batam
            else:
                 return False
        return True
    except Exception as e:
        print(f"Erro durante comparação de splits: {e}")
        return False # Retorna falso se qualquer erro ocorrer durante a comparação

print(f"--- Iniciando Verificação dos Datasets com NaNs ({SCRIPT_VERSION}) ---")

overall_status = True # Flag geral para indicar se todas as verificações passaram

for base_name in datasets_to_verify:
    original_dir_name = f"{base_name}{original_suffix}"
    original_dir = base_dir / original_dir_name
    print(f"\n=== Verificando Dataset Base: {base_name} ===")

    if not original_dir.is_dir():
        print(f"  ERRO: Diretório original '{original_dir}' não encontrado. Pulando.")
        overall_status = False
        continue

    # --- Carregar Dados Originais ---
    try:
        print(f"  Carregando dados de: {original_dir_name}")
        X_orig = load_npy_gz(original_dir / 'X.npy.gz')
        y_orig = load_npy_gz(original_dir / 'y.npy.gz') # Carrega y com pickle True por precaução
        splits_orig = load_npy_gz(original_dir / 'split_indeces.npy.gz')
        with open(original_dir / 'metadata.json', 'r') as f:
            metadata_orig = json.load(f)
        print("    Dados originais carregados com sucesso.")
        # Calcula NaNs originais (convertendo X para float para o cálculo)
        try:
            X_orig_float = X_orig.astype(np.float32)
            original_nan_count = np.isnan(X_orig_float).sum()
            print(f"    Contagem de NaNs no X original: {original_nan_count} (em {X_orig.size} elementos)")
            if original_nan_count > X_orig.size * 0.01:
                 print("    AVISO: Dataset original parece já conter NaNs!")
        except Exception as e:
            print(f"    AVISO: Não foi possível verificar NaNs no X original (erro na conversão para float?): {e}")
            original_nan_count = 0 # Assume 0 se a verificação falhar

    except Exception as e:
        print(f"  ERRO FATAL ao carregar dados originais: {e}")
        print(traceback.format_exc())
        overall_status = False
        continue # Pula para o próximo dataset

    # --- Verificar cada porcentagem ---
    for perc in percentages_to_verify:
        dataset_status = "OK" # Status para esta porcentagem
        missing_suffix = missing_suffix_template.format(percentage=perc)
        partial_dir_name = f"{base_name}{missing_suffix}"
        partial_dir = base_dir / partial_dir_name
        print(f"\n  --- Verificando Versão {perc}% ({partial_dir_name}) ---")

        # 1 & 2: Existência do diretório e arquivos
        print("    Verificando existência de diretório e arquivos...")
        if not partial_dir.is_dir():
            print(f"    FALHA: Diretório não encontrado: {partial_dir}")
            overall_status = False
            dataset_status = "FALHA"
            continue # Pula para a próxima porcentagem
        files_ok = True
        expected_files = ['X.npy.gz', 'y.npy.gz', 'metadata.json', 'split_indeces.npy.gz']
        for fname in expected_files:
            if not (partial_dir / fname).exists():
                print(f"    FALHA: Arquivo '{fname}' não encontrado em {partial_dir}.")
                files_ok = False
                overall_status = False
                dataset_status = "FALHA"
        if not files_ok: continue # Pula para a próxima porcentagem
        print("      Diretório e arquivos encontrados.")

        # Carregar dados modificados
        try:
            print("    Carregando dados modificados...")
            X_mod = load_npy_gz(partial_dir / 'X.npy.gz')
            y_mod = load_npy_gz(partial_dir / 'y.npy.gz')
            splits_mod = load_npy_gz(partial_dir / 'split_indeces.npy.gz')
            with open(partial_dir / 'metadata.json', 'r') as f:
                metadata_mod = json.load(f)
            print("      Dados modificados carregados.")
        except Exception as e:
            print(f"    FALHA: Erro ao carregar arquivos da versão {perc}%: {e}")
            print(traceback.format_exc())
            overall_status = False
            dataset_status = "FALHA"
            continue

        # 3: Verificar se NaNs foram introduzidos em X
        print("    Verificando presença de NaNs em X_mod...")
        # Precisamos garantir que X_mod seja float para isnan funcionar
        try:
             if not np.issubdtype(X_mod.dtype, np.floating):
                 X_mod_float = X_mod.astype(np.float32)
                 print(f"      (X_mod convertido de {X_mod.dtype} para float32 para verificação de NaN)")
             else:
                 X_mod_float = X_mod
             modified_nan_count = np.isnan(X_mod_float).sum()
             print(f"      Total de NaNs encontrados: {modified_nan_count}")
             if modified_nan_count == 0:
                 print(f"    FALHA: Nenhum NaN encontrado no X modificado ({perc}%).")
                 dataset_status = "FALHA"; overall_status = False
             elif modified_nan_count <= original_nan_count:
                 print(f"    AVISO: Contagem de NaNs não aumentou em X ({perc}%). Original={original_nan_count}, Modificado={modified_nan_count}")
                 # Não falha o teste, mas é suspeito
             else:
                  print("      Presença de NaNs confirmada.")
        except Exception as e:
             print(f"    ERRO ao verificar NaNs em X_mod: {e}")
             dataset_status = "FALHA"; overall_status = False

        # 6: Comparar y, metadata, splits
        print("    Comparando y, metadata e splits com originais...")
        y_ok = np.array_equal(y_orig, y_mod)
        meta_ok = (metadata_orig == metadata_mod)
        splits_ok = compare_splits(splits_orig, splits_mod)
        if not y_ok: print(f"      FALHA: y.npy.gz foi modificado!"); dataset_status = "FALHA"; overall_status = False
        if not meta_ok: print(f"      FALHA: metadata.json foi modificado!"); dataset_status = "FALHA"; overall_status = False
        if not splits_ok: print(f"      FALHA: split_indeces.npy.gz foi modificado!"); dataset_status = "FALHA"; overall_status = False
        if y_ok and meta_ok and splits_ok: print("      y, metadata e splits idênticos ao original (OK).")

        # 4 & 5: Verificar NaNs apenas no treino e porcentagem
        print(f"    Verificando NaNs por fold e porcentagem ({perc}%)...")
        fold_checks_passed = True
        total_train_elements = np.int64(0) # Usar int64 para evitar overflow potencial
        total_train_nans = np.int64(0)
        fold_nan_percentages = []

        for fold_idx, fold_indices in enumerate(splits_mod):
            fold_has_error = False
            try:
                train_idx, val_idx, test_idx = None, None, None
                # Lógica robusta para pegar os índices
                if hasattr(fold_indices, 'keys') and 'train' in fold_indices:
                    train_idx = np.array(fold_indices['train'], dtype=np.int64)
                    val_idx = np.array(fold_indices.get('val', []), dtype=np.int64) # .get para caso não exista
                    test_idx = np.array(fold_indices.get('test', []), dtype=np.int64)
                elif isinstance(fold_indices, collections.abc.Sequence) and not hasattr(fold_indices, 'keys') and len(fold_indices) == 3:
                    train_idx = np.array(fold_indices[0], dtype=np.int64)
                    val_idx = np.array(fold_indices[1], dtype=np.int64)
                    test_idx = np.array(fold_indices[2], dtype=np.int64)
                else:
                    print(f"      FALHA Fold {fold_idx+1}: Estrutura de split inválida: {type(fold_indices)}")
                    fold_checks_passed = False; fold_has_error = True; continue

                # Checagens de limites dos índices
                if train_idx.size > 0 and np.max(train_idx) >= X_mod_float.shape[0]: raise IndexError("Train index out of bounds")
                if val_idx.size > 0 and np.max(val_idx) >= X_mod_float.shape[0]: raise IndexError("Val index out of bounds")
                if test_idx.size > 0 and np.max(test_idx) >= X_mod_float.shape[0]: raise IndexError("Test index out of bounds")

                # Verificar NaNs fora do treino
                nans_in_val = np.isnan(X_mod_float[val_idx, :]).sum() if val_idx.size > 0 else 0
                nans_in_test = np.isnan(X_mod_float[test_idx, :]).sum() if test_idx.size > 0 else 0

                if nans_in_val > 0:
                     print(f"      FALHA Fold {fold_idx+1}: {nans_in_val} NaNs encontrados nos dados de VALIDAÇÃO.")
                     fold_checks_passed = False; fold_has_error = True
                if nans_in_test > 0:
                     print(f"      FALHA Fold {fold_idx+1}: {nans_in_test} NaNs encontrados nos dados de TESTE.")
                     fold_checks_passed = False; fold_has_error = True

                # Calcular porcentagem no treino
                if train_idx.size > 0:
                    X_train_subset = X_mod_float[train_idx, :]
                    nans_in_train = np.isnan(X_train_subset).sum()
                    train_elements = X_train_subset.size
                    total_train_elements += train_elements
                    total_train_nans += nans_in_train
                    fold_nan_perc = nans_in_train / train_elements if train_elements > 0 else 0.0
                    fold_nan_percentages.append(fold_nan_perc)
                    # print(f"      Fold {fold_idx+1}: Train NaNs={nans_in_train}/{train_elements} ({fold_nan_perc:.3f}), Val NaNs={nans_in_val}, Test NaNs={nans_in_test}")
                # else: print(f"      Fold {fold_idx+1}: Sem dados de treino.")

            except Exception as e:
                print(f"      ERRO ao verificar Fold {fold_idx+1}: {e}")
                print(traceback.format_exc())
                fold_checks_passed = False; fold_has_error = True

        # Resumo da verificação dos folds
        if fold_checks_passed:
            overall_train_nan_perc = total_train_nans / total_train_elements if total_train_elements > 0 else 0.0
            print(f"      Verificação de NaNs (Val/Test): OK")
            print(f"      Porcentagem MÉDIA de NaNs (Train): {overall_train_nan_perc:.4f} (Alvo: {perc/100.0:.4f})")
            # Verifica se a porcentagem média está dentro da tolerância
            if not (abs(overall_train_nan_perc - (perc / 100.0)) <= NAN_PERCENTAGE_TOLERANCE):
                 print(f"      AVISO: Porcentagem média de NaNs ({overall_train_nan_perc:.4f}) está fora da tolerância ({NAN_PERCENTAGE_TOLERANCE}) em relação ao alvo {perc/100.0:.4f}.")
                 # Você pode decidir se isso deve ser uma FALHA ou apenas um AVISO
                 # dataset_status = "AVISO"; overall_status = False # Exemplo: marcar como falha geral
            else:
                print(f"      Porcentagem de NaNs no treino está dentro da tolerância.")
        else:
            print(f"    FALHA: Erros encontrados durante a verificação dos folds para {perc}%.")
            dataset_status = "FALHA"; overall_status = False

        print(f"    Status da Versão {perc}%: {dataset_status}")

print("\n--- Resumo Final da Verificação ---")
if overall_status:
    print("SUCESSO: Todas as verificações básicas passaram para todos os datasets e porcentagens testados.")
else:
    print("FALHA: Pelo menos uma verificação falhou. Revise o output acima para detalhes.")