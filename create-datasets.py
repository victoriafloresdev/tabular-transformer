import numpy as np
import pandas as pd 
from pathlib import Path
import copy
import sys
from typing import List 

# --- Ajuste de sys.path para a estrutura especificada ---
# Estrutura: /tabular_transformers/tabzilla/TabZilla/
# Script está em: /tabular_transformers/
# Precisamos adicionar: /tabular_transformers/tabzilla/ ao sys.path

try:
    # Diretório onde o script create-datasets.py está localizado
    script_dir = Path(__file__).resolve().parent

    # Caminho para o diretório que contém o pacote TabZilla
    # (./tabular_transformers/tabzilla/)
    tabzilla_parent_dir = script_dir / 'tabzilla'

    if not tabzilla_parent_dir.is_dir():
        print(f"Erro: Diretório esperado '{tabzilla_parent_dir}' não encontrado.")
        print(f"Verifique a estrutura de pastas. Script está em: {script_dir}")
        sys.exit(1)

    # Adiciona o diretório ao sys.path se não estiver lá
    if str(tabzilla_parent_dir) not in sys.path:
        print(f"Adicionando ao sys.path: {tabzilla_parent_dir}")
        sys.path.insert(0, str(tabzilla_parent_dir))

    # Agora tenta importar usando o nome do pacote 'TabZilla'
    from TabZilla.tabzilla_data_preprocessing import preprocess_dataset
    from TabZilla.tabzilla_datasets import TabularDataset
    print("Imports do TabZilla bem-sucedidos.")

except ImportError as e:
    print(f"Erro CRÍTICO ao importar módulos do TabZilla: {e}")
    print(f"Verifique se os arquivos .py existem em: {tabzilla_parent_dir / 'TabZilla'}")
    print(f"sys.path atual: {sys.path}")
    sys.exit(1)
except NameError:
     print("Erro: Provavelmente __file__ não está definido (executando interativamente?).")
     print("Execute o script diretamente com 'python create-datasets.py'.")
     # Tenta importar diretamente, pode funcionar se o PYTHONPATH estiver configurado
     try:
         from TabZilla.tabzilla_data_preprocessing import preprocess_dataset
         from TabZilla.tabzilla_datasets import TabularDataset
     except ImportError:
          print("Falha ao importar diretamente. Configure o PYTHONPATH ou execute como script.")
          sys.exit(1)


# --- Configurações (mantidas) ---
BASE_DATASET_NAMES = [
    "openml__ada_agnostic__3896",
    "openml__adult__7592",
    "openml__adult-census__3953",
    "openml__airlines__189354",
    "openml__albert__189356",
]

MISSING_RATIOS = [0.25, 0.50, 0.75]
# O diretório de saída será criado dentro de ./tabular_transformers/
OUTPUT_DIR = Path("./datasets_with_missing")
RANDOM_SEED = 42

# --- Função Modificada para Injetar Valores Ausentes (mantida) ---
def inject_missing_values(X: np.ndarray, cat_idx: List[int], ratio: float, seed: int) -> np.ndarray:
    """
    Injeta np.nan (para features numéricas) ou a string 'Missing'
    (para features categóricas) aleatoriamente em cada coluna.
    """
    if ratio == 0.0:
        return X.copy()

    rng = np.random.default_rng(seed)
    # Usar dtype=object para permitir misturar números e strings ('Missing')
    X_missing = X.copy().astype(object)
    num_rows, num_cols = X.shape
    n_to_missing = int(num_rows * ratio)

    if n_to_missing == 0 and ratio > 0.0:
         print(f"   Aviso: Ratio {ratio} resulta em 0 valores ausentes para {num_rows} linhas.")
         return X_missing # Retorna a cópia object sem valores ausentes

    print(f"   Injetando ~{n_to_missing} valores ausentes por coluna ({ratio*100:.0f}%)...")

    for j in range(num_cols):
        if num_rows == 0: continue # Evita erro se o dataset estiver vazio

        # Escolhe 'n_to_missing' índices únicos de linha para esta coluna
        chosen_indices = rng.choice(num_rows, min(n_to_missing, num_rows), replace=False)

        if j in cat_idx:
            X_missing[chosen_indices, j] = 'Missing'
        else:
            X_missing[chosen_indices, j] = np.nan

    return X_missing

# --- Lógica Principal (mantida) ---
def main():
    print(f"Gerando datasets com dados ausentes para: {BASE_DATASET_NAMES}")
    print(f"Percentuais de ausência: {[f'{r*100:.0f}%' for r in MISSING_RATIOS]}")
    print(f"Usando label 'Missing' para categóricos e np.nan para numéricos.")
    print(f"Diretório de saída: {OUTPUT_DIR.resolve()}")
    print(f"Semente aleatória para NaNs: {RANDOM_SEED}")
    print("-" * 30)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    processed_count = 0
    total_to_process = len(BASE_DATASET_NAMES) * (1 + len(MISSING_RATIOS)) # 1 original + N missing ratios

    original_dataset_standard_parent_dir = Path("./datasets")

    for base_name in BASE_DATASET_NAMES:
        print(f"Processando dataset base: {base_name}")
        # Caminho padrão onde preprocess_dataset salva/encontra o original
        original_dataset_standard_path = original_dataset_standard_parent_dir / base_name

        # 1. Garantir que o dataset original existe no disco E CARREGÁ-LO
        try:
            print(f"  Verificando/Processando versão original (em {original_dataset_standard_path})...")
            # preprocess_dataset GARANTE que o dataset está no disco e retorna o CAMINHO
            returned_path = preprocess_dataset(base_name, overwrite=False, verbose=True)

            if returned_path is None or not returned_path.is_dir():
                 print(f"Erro: Falha ao processar/encontrar o dataset original {base_name} em {returned_path}. Pulando.")
                 continue

            # ***** CORREÇÃO PRINCIPAL: Ler o dataset do caminho retornado *****
            print(f"  Carregando dataset original de '{returned_path}' para memória...")
            original_dataset_obj = TabularDataset.read(returned_path)
            # ******************************************************************

            # Agora 'original_dataset_obj' contém o objeto TabularDataset real

            # Garante que o diretório padrão existe (caso não existisse antes)
            original_dataset_standard_parent_dir.mkdir(parents=True, exist_ok=True)

            # Salva/Verifica cópia no OUTPUT_DIR deste script
            original_dest_path = OUTPUT_DIR / base_name
            if not original_dest_path.exists():
                 print(f"  Salvando cópia da versão original em '{original_dest_path.resolve()}'...")
                 # Usa o objeto TabularDataset para escrever a cópia
                 original_dataset_obj.write(original_dest_path, overwrite=False) # <--- CORRIGIDO

            # Não incrementa processed_count aqui, ele será incrementado abaixo
            # ou apenas quando as versões missing forem processadas, dependendo do que contar.
            # Vamos contar cada escrita bem-sucedida. Se a cópia foi escrita, conta 1.
            if not original_dest_path.exists(): # Se acabou de escrever a cópia
                 processed_count += 1

            print(f"  Versão original '{base_name}' pronta e carregada.")

        except Exception as e:
            print(f"Erro ao processar/carregar dataset original {base_name}: {e}")
            import traceback
            traceback.print_exc()
            continue # Pula para o próximo dataset base em caso de erro grave aqui

        # 2. Gerar e salvar versões com dados ausentes no OUTPUT_DIR
        for ratio in MISSING_RATIOS:
            new_dataset_name = f"{base_name}_{int(ratio*100)}pct_missing"
            dest_path = OUTPUT_DIR / new_dataset_name
            print(f"  Gerando versão: {new_dataset_name}")

            try:
                # Usa o objeto TabularDataset carregado ('original_dataset_obj')
                X_original = original_dataset_obj.X
                y_original = original_dataset_obj.y
                cat_idx_original = original_dataset_obj.cat_idx

                current_seed = RANDOM_SEED + int(ratio * 100)
                X_missing = inject_missing_values(X_original, cat_idx_original, ratio, seed=current_seed)

                missing_dataset = TabularDataset(
                    name=new_dataset_name,
                    X=X_missing,
                    y=y_original.copy(),
                    # Usa os metadados do objeto carregado
                    cat_idx=copy.deepcopy(original_dataset_obj.cat_idx),
                    target_type=original_dataset_obj.target_type,
                    num_classes=original_dataset_obj.num_classes,
                    num_features=original_dataset_obj.num_features,
                    num_instances=original_dataset_obj.num_instances,
                    cat_dims=copy.deepcopy(original_dataset_obj.cat_dims),
                    split_indeces=copy.deepcopy(original_dataset_obj.split_indeces),
                    split_source=original_dataset_obj.split_source,
                )

                missing_dataset.write(dest_path, overwrite=True)
                processed_count += 1 # Conta cada versão missing gerada
                print(f"  Dataset '{new_dataset_name}' salvo em '{dest_path.resolve()}'")

            except Exception as e:
                print(f"Erro ao gerar/salvar {new_dataset_name}: {e}")
                import traceback
                traceback.print_exc()

        print("-" * 30)

    # Ajusta a contagem total esperada se não contamos a cópia original
    total_expected = len(BASE_DATASET_NAMES) * len(MISSING_RATIOS) # Apenas versões missing
    # Ou se contamos a cópia: total_expected = len(BASE_DATASET_NAMES) * (1 + len(MISSING_RATIOS))

    print(f"\nProcessamento concluído!")
    # Atualiza a mensagem final para refletir o que foi contado
    print(f"Total de datasets com dados ausentes gerados: {processed_count} de {total_expected} planejados.")
    print(f"Datasets com dados ausentes salvos em: {OUTPUT_DIR.resolve()}")
    print(f"Lembre-se que as versões originais foram processadas/salvas no diretório padrão: {original_dataset_standard_parent_dir.resolve()}")

if __name__ == "__main__":
    main()