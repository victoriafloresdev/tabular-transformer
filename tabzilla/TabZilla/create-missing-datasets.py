import numpy as np
import pathlib
import gzip
import json
import shutil
import collections.abc # Para checar tipo de objeto (sequência ou dicionário)
import traceback # Para imprimir erros mais detalhados
import argparse # Para argumentos de linha de comando (opcional, mas bom)

# --- Configurações ---
SCRIPT_VERSION = "1.0-NaN-Introducer"
# Lista dos nomes base dos datasets ORIGINAIS (sem sufixo de porcentagem)
# Adapte esta lista aos nomes exatos das suas pastas _100pct
dataset_base_names = [
    'openml__credit-g__31_100pct',
    'openml__diabetes__37_100pct',
    'openml__adult__7592_100pct',
    'openml__spambase__43_100pct',
    'openml__Amazon_employee_access__34539_100pct'
]
# Porcentagens de VALORES NULOS a introduzir nos dados de TREINO
missing_percentages = [0.25, 0.50, 0.75]
# Diretório base onde as pastas dos datasets estão/serão criadas
base_dir = pathlib.Path('./datasets') # Ajuste se necessário
# Template para o sufixo das novas pastas
missing_suffix_template = '_{percentage}pct_missing' # Ex: _25pct_missing
# Seed para reprodutibilidade da seleção aleatória de NaNs
SEED = 42
# ---------------------

rng = np.random.default_rng(SEED)
print(f"--- Iniciando Script de Introdução de NaNs ({SCRIPT_VERSION}) ---")

# Verifica se o diretório base existe
if not base_dir.is_dir():
    print(f"ERRO FATAL: Diretório base não encontrado: {base_dir}")
    exit()

for original_name in dataset_base_names:
    original_dir = base_dir / original_name
    base_name = original_name.replace('_100pct', '') # Remove o sufixo para criar nomes base

    if not original_dir.is_dir():
        print(f"\nAVISO: Diretório original não encontrado: {original_dir}. Pulando dataset '{base_name}'.")
        continue

    print(f"\n--- Processando dataset: {base_name} ---")
    print(f"  Diretório original: {original_dir.resolve()}")

    # --- Carregar Dados Originais Manualmente (Garantindo allow_pickle=True) ---
    X_original, y_original, original_splits_list, metadata = None, None, None, None
    try:
        # Carrega X
        x_file_path = original_dir / 'X.npy.gz'
        print(f"  Carregando X de {x_file_path}...")
        if not x_file_path.exists(): raise FileNotFoundError(f"X não encontrado: {x_file_path}")
        with gzip.GzipFile(x_file_path, 'rb') as f:
            # Usar allow_pickle=True explicitamente aqui
            X_original = np.load(f, allow_pickle=True)
        print(f"    X carregado. Shape={X_original.shape}, Dtype={X_original.dtype}")

        # Carrega y
        y_file_path = original_dir / 'y.npy.gz'
        print(f"  Carregando y de {y_file_path}...")
        if not y_file_path.exists(): raise FileNotFoundError(f"y não encontrado: {y_file_path}")
        with gzip.GzipFile(y_file_path, 'rb') as f:
            # Usar allow_pickle=True explicitamente aqui também, por segurança
            y_original = np.load(f, allow_pickle=True)
        print(f"    y carregado. Shape={y_original.shape}, Dtype={y_original.dtype}")

        # Carrega Splits
        split_file_path = original_dir / 'split_indeces.npy.gz'
        print(f"  Carregando splits de {split_file_path}...")
        if not split_file_path.exists(): raise FileNotFoundError(f"Split não encontrado: {split_file_path}")
        with gzip.GzipFile(split_file_path, "rb") as f:
             # Usar allow_pickle=True explicitamente aqui
            original_splits_list = np.load(f, allow_pickle=True)
        print(f"    Splits carregados. Tipo: {type(original_splits_list)}, Len/Shape: {getattr(original_splits_list, 'shape', len(original_splits_list))}")

        # Carrega Metadata
        metadata_path = original_dir / 'metadata.json'
        print(f"  Carregando metadata de {metadata_path}...")
        if not metadata_path.exists(): raise FileNotFoundError(f"Metadata não encontrada: {metadata_path}")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        print(f"    Metadata carregada.")

    except Exception as e:
        print(f"ERRO CRÍTICO durante carregamento para {base_name}: {e}")
        print(traceback.format_exc())
        print(f"Pulando processamento completo para o dataset: {base_name}")
        continue # Pula para o próximo dataset base

    # --- Preparar X para receber NaNs (converter para float se necessário) ---
    original_dtype = X_original.dtype
    if np.issubdtype(original_dtype, np.integer) or np.issubdtype(original_dtype, np.bool_):
        print(f"  Info: Convertendo X de {original_dtype} para float32 para suportar NaNs.")
        X_for_nan_introduction = X_original.astype(np.float32)
    elif not np.issubdtype(original_dtype, np.floating):
         # Se for 'object' ou outro tipo não float, tenta converter, mas avisa.
         print(f"  AVISO: X original tem dtype {original_dtype}. Tentando converter para float32. Isso pode falhar se houver dados não numéricos.")
         try:
             X_for_nan_introduction = X_original.astype(np.float32)
         except ValueError as ve:
             print(f"    ERRO: Falha ao converter X para float: {ve}. Pulando dataset {base_name}")
             continue
    else:
         # Já é float
         X_for_nan_introduction = X_original # Não precisa de cópia ainda, faremos dentro do loop de %

    num_features = X_for_nan_introduction.shape[1]

    # --- Loop pelas porcentagens de NaNs ---
    for missing_perc in missing_percentages:
        percentage_int = int(missing_perc * 100)
        missing_suffix = missing_suffix_template.format(percentage=percentage_int)
        partial_dir_name = f"{base_name}{missing_suffix}"
        partial_dir = base_dir / partial_dir_name
        print(f"\n    -> Criando versão com {percentage_int}% de NaNs: {partial_dir}")

        # Remove e recria diretório de destino
        if partial_dir.exists():
            print(f"      Aviso: Diretório {partial_dir} já existe. Removendo e recriando.")
            try:
                shutil.rmtree(partial_dir)
            except Exception as e:
                print(f"      ERRO ao remover {partial_dir}: {e}. Pulando esta porcentagem.")
                continue
        try:
            partial_dir.mkdir(parents=True)
        except Exception as e:
            print(f"      ERRO ao criar {partial_dir}: {e}. Pulando esta porcentagem.")
            continue

        # --- Introdução de NaNs ---
        # Copia a matriz X (já convertida para float, se necessário)
        X_modified = X_for_nan_introduction.copy()
        fold_counter = 0
        total_nans_introduced_fold = 0
        nan_introduction_errors = False

        print(f"      Introduzindo NaNs nos dados de TREINO de cada fold...")
        for fold_indices in original_splits_list:
            fold_counter += 1
            try:
                train_idx_orig = None
                # Lógica refinada para identificar o array de treino
                if hasattr(fold_indices, 'keys') and 'train' in fold_indices:
                    train_idx_orig = fold_indices['train']
                elif isinstance(fold_indices, collections.abc.Sequence) and not hasattr(fold_indices, 'keys') and len(fold_indices) >= 1: # >=1 para ser seguro
                    # Assume que o primeiro elemento é o treino se for uma sequência e não um dict
                    train_idx_orig = fold_indices[0]
                else:
                    print(f"        ERRO: Fold {fold_counter} estrutura inesperada: {type(fold_indices)}. Pulando fold.")
                    nan_introduction_errors = True; continue

                # Garantir que é array numpy e não vazio
                if not isinstance(train_idx_orig, np.ndarray): train_idx_orig = np.array(train_idx_orig, dtype=np.int64) # Especificar dtype pode ajudar
                if train_idx_orig.size == 0:
                    # print(f"        Info: Fold {fold_counter} não tem dados de treino. Pulando introdução de NaN.")
                    continue

                # Introduzir NaNs coluna por coluna, apenas nos índices de treino
                num_train_samples_fold = len(train_idx_orig)
                num_to_nullify_per_col = int(num_train_samples_fold * missing_perc)
                if num_to_nullify_per_col == 0 and missing_perc > 0:
                     print(f"        Aviso: Fold {fold_counter}: Número de NaNs a introduzir por coluna é 0 ({missing_perc * 100}% de {num_train_samples_fold}).")
                     continue

                for j in range(num_features):
                    # Escolher índices RELATIVOS ao conjunto de treino
                    relative_indices_to_nullify = rng.choice(num_train_samples_fold, size=num_to_nullify_per_col, replace=False)
                    # Converter para índices ABSOLUTOS na matriz X_modified
                    absolute_indices_to_nullify = train_idx_orig[relative_indices_to_nullify]
                    # Definir como NaN
                    X_modified[absolute_indices_to_nullify, j] = np.nan
                    total_nans_introduced_fold += len(absolute_indices_to_nullify)

            except KeyError as ke:
                 print(f"        ERRO de Chave ao acessar indices fold {fold_counter}: {ke}. Conteúdo: {repr(fold_indices)}. Pulando fold.")
                 nan_introduction_errors = True; continue
            except IndexError as ie:
                 print(f"        ERRO de Índice ao acessar indices fold {fold_counter}: {ie}. Conteúdo: {repr(fold_indices)}. Pulando fold.")
                 nan_introduction_errors = True; continue
            except Exception as e:
                print(f"        ERRO inesperado ao processar indices fold {fold_counter}: {e}.")
                print(traceback.format_exc())
                nan_introduction_errors = True; continue
        # --- FIM LOOP FOLDS ---

        if nan_introduction_errors:
             print(f"      AVISO: Erros ocorreram durante a introdução de NaNs em alguns folds para {percentage_int}%. O arquivo X salvo pode estar incompleto.")
        else:
            print(f"      NaNs introduzidos com sucesso em {total_nans_introduced_fold} posições (total nos folds).")


        # --- Salvar Arquivos Modificados e Copiar Originais ---
        try:
            # Salva X modificado
            x_mod_path = partial_dir / 'X.npy.gz'; print(f"      Salvando {x_mod_path}")
            with gzip.GzipFile(x_mod_path, 'wb') as f: np.save(f, X_modified)

            # Copia y original
            y_orig_path = original_dir / 'y.npy.gz'; y_dest_path = partial_dir / 'y.npy.gz'
            print(f"      Copiando {y_orig_path} para {y_dest_path}")
            shutil.copyfile(y_orig_path, y_dest_path)

            # Copia splits originais
            split_orig_path = original_dir / 'split_indeces.npy.gz'; split_dest_path = partial_dir / 'split_indeces.npy.gz'
            print(f"      Copiando {split_orig_path} para {split_dest_path}")
            shutil.copyfile(split_orig_path, split_dest_path)

            # Copia metadata original
            # Load original metadata, update name, and save
            meta_orig_path = original_dir / 'metadata.json'
            meta_dest_path = partial_dir / 'metadata.json'
            print(f"      Atualizando metadata para {meta_dest_path}")
            with open(meta_orig_path, "r") as f:
                metadata = json.load(f)
            # Update the dataset name to include the missing suffix
            metadata["name"] = f"{base_name}{missing_suffix}"
            with open(meta_dest_path, "w") as f:
                json.dump(metadata, f, indent=4)

            print(f"      Dataset {partial_dir_name} criado com sucesso.")

        except Exception as e:
            print(f"      ERRO ao salvar ou copiar arquivos para {partial_dir}: {e}")
            print(traceback.format_exc())

print(f"\n--- Criação de datasets com valores nulos concluída ({SCRIPT_VERSION}) ---")