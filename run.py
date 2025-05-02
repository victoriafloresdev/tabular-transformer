import subprocess
import sys
import os
import json
import pandas as pd
from pathlib import Path
import glob
import time
from typing import List, Dict, Any, Tuple

# --- Configurações do Script ---

# 1. Lista de Modelos Alvo (Nomes como definidos no TabZilla)
MODELS_TO_RUN: List[str] = [
    "KNN",
    "RandomForest",
    "XGBoost",
    "TabTransformer",
    "SAINT",
    "rtdl_FTTransformer",
]

# 2. Diretório Pai contendo TODOS os datasets (originais e com missing)
#    Gerado pelo script anterior (create-datasets.py)
DATASET_PARENT_DIR: Path = Path("./datasets_with_missing")

# 3. Caminho para o script principal de experimentos do TabZilla
#    Ajuste conforme a sua estrutura de pastas
TABZILLA_EXP_SCRIPT: Path = Path("./tabzilla/TabZilla/tabzilla_experiment.py")

# 4. Caminho para o arquivo de configuração YML para os experimentos
#    *** VOCÊ PRECISA CRIAR ESTE ARQUIVO (exemplo abaixo) ***
EXP_CONFIG_FILE: Path = Path("./exp_config_missing.yml")

# 5. Diretório onde os resultados JSON individuais de cada experimento serão salvos
RESULTS_OUTPUT_DIR: Path = Path("./experiment_results_missing")

# 6. Nome do arquivo CSV final agregado
FINAL_CSV_RESULTS_FILE: Path = Path("./final_aggregated_results_missing.csv")
FINAL_CSV_EXCEPTIONS_FILE: Path = Path("./final_aggregated_exceptions_missing.csv")

# --- Verificações Iniciais ---
if not DATASET_PARENT_DIR.is_dir():
    print(f"Erro: Diretório de datasets não encontrado: {DATASET_PARENT_DIR.resolve()}")
    sys.exit(1)
if not TABZILLA_EXP_SCRIPT.is_file():
    print(f"Erro: Script de experimento do TabZilla não encontrado: {TABZILLA_EXP_SCRIPT.resolve()}")
    sys.exit(1)
if not EXP_CONFIG_FILE.is_file():
    print(f"Erro: Arquivo de configuração do experimento não encontrado: {EXP_CONFIG_FILE.resolve()}")
    print("Crie o arquivo 'exp_config_missing.yml' antes de executar.")
    sys.exit(1)

# Cria o diretório de resultados se não existir
RESULTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Função para Parsear Arquivo de Resultado JSON (Adaptada do Aggregator) ---
def parse_single_json_result(filepath: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Parseia um arquivo _results.json e retorna listas de resultados e exceções."""
    results_list = []
    exceptions_list = []
    print(f"  Parsing: {filepath.name}...")
    try:
        with open(filepath, "r") as f:
            contents = json.load(f)

        # Extrai informações básicas que podem estar fora do try-except principal
        dataset_name = contents.get("dataset", {}).get("name", "N/A")
        model_name = contents.get("model", {}).get("name", "N/A")
        hparam_source = contents.get("hparam_source", "N/A")
        trial_number = contents.get("trial_number", -1)
        hparam_seed = contents.get("experiemnt_args", {}).get("hparam_seed", -1) # Note: typo 'experiemnt_args' no TabZilla original
        if hparam_seed == -1: # Tenta com a grafia correta se a errada falhar
             hparam_seed = contents.get("experiment_args", {}).get("hparam_seed", -1)


        # Identificador único para esta combinação de alg+hparam
        alg_hparam_id = f'{model_name}__seed_{hparam_seed}__trial_{trial_number}'

        is_exception = contents.get("exception", "None") != "None"

        if not is_exception:
            # Verifica se os dados necessários existem
            if not all(k in contents for k in ["timers", "scorers"]):
                 raise ValueError("Arquivo JSON não contém 'timers' ou 'scorers'")
            if "train" not in contents["timers"]:
                 raise ValueError("Arquivo JSON não contém 'timers' para 'train'")

            num_folds = len(contents["timers"]["train"])
            if num_folds == 0:
                raise ValueError("Número de folds é 0 no arquivo JSON")

            for fold_number in range(num_folds):
                fold_results = {
                    "dataset_fold_id": f"{dataset_name}__fold_{fold_number}",
                    "dataset_name": dataset_name,
                    "target_type": contents.get("dataset", {}).get("target_type", "N/A"),
                    "model_name": model_name,
                    "hparam_source": hparam_source,
                    "trial_number": trial_number,
                    "alg_hparam_id": alg_hparam_id,
                    "fold": fold_number,
                    "num_folds": num_folds,
                    "source_json": filepath.name # Adiciona nome do arquivo de origem
                }

                # Adiciona tempos
                for phase in ["train", "val", "test", "train-eval"]:
                    if phase in contents["timers"] and len(contents["timers"][phase]) > fold_number:
                        fold_results[f"time__{phase}"] = contents["timers"][phase][fold_number]

                # Adiciona scores
                for phase in ["train", "val", "test"]:
                     if phase in contents["scorers"]:
                          for metric, values in contents["scorers"][phase].items():
                               if isinstance(values, list) and len(values) > fold_number:
                                    fold_results[f"{metric}__{phase}"] = values[fold_number]

                results_list.append(fold_results)
        else:
            exception_info = {
                "dataset_name": dataset_name,
                "model_name": model_name,
                "hparam_source": hparam_source,
                "trial_number": trial_number,
                "alg_hparam_id": alg_hparam_id,
                "exception": contents.get("exception", "Unknown Exception"),
                "source_json": filepath.name # Adiciona nome do arquivo de origem
            }
            exceptions_list.append(exception_info)

    except json.JSONDecodeError:
        print(f"  Erro: Falha ao decodificar JSON: {filepath.name}")
        exceptions_list.append({"source_json": filepath.name, "exception": "JSONDecodeError"})
    except KeyError as e:
        print(f"  Erro: Chave não encontrada no JSON {filepath.name}: {e}")
        exceptions_list.append({"source_json": filepath.name, "exception": f"KeyError: {e}"})
    except ValueError as e:
         print(f"  Erro: Valor inválido no JSON {filepath.name}: {e}")
         exceptions_list.append({"source_json": filepath.name, "exception": f"ValueError: {e}"})
    except Exception as e:
        print(f"  Erro inesperado ao parsear {filepath.name}: {e}")
        exceptions_list.append({"source_json": filepath.name, "exception": f"UnexpectedError: {e}"})

    return results_list, exceptions_list


# --- Lógica Principal de Execução e Agregação ---
def main():
    print("--- Iniciando Orquestração de Experimentos TabZilla ---")
    start_time = time.time()

    # 1. Coleta dos diretórios de datasets
    dataset_dirs = [d for d in DATASET_PARENT_DIR.iterdir() if d.is_dir()]
    if not dataset_dirs:
        print(f"Erro: Nenhum diretório de dataset encontrado em {DATASET_PARENT_DIR.resolve()}")
        sys.exit(1)

    print(f"Encontrados {len(dataset_dirs)} diretórios de datasets em {DATASET_PARENT_DIR.name}")
    print(f"Modelos a serem executados: {MODELS_TO_RUN}")
    total_experiments = len(dataset_dirs) * len(MODELS_TO_RUN)
    print(f"Total de experimentos planejados: {total_experiments}")
    print("-" * 30)

    # 2. Execução dos experimentos via subprocess
    completed_count = 0
    failed_experiments = []

    # Adiciona o diretório pai do TabZilla ao PYTHONPATH para subprocessos
    tabzilla_parent_env = os.environ.copy()
    tabzilla_grandparent_dir = TABZILLA_EXP_SCRIPT.parent.parent.parent # /path/to/tabular-transformers/tabzilla
    pythonpath = tabzilla_parent_env.get("PYTHONPATH", "")
    if str(tabzilla_grandparent_dir) not in pythonpath.split(os.pathsep):
        tabzilla_parent_env["PYTHONPATH"] = f"{tabzilla_grandparent_dir}{os.pathsep}{pythonpath}"
        print(f"Exportando PYTHONPATH para subprocessos: {tabzilla_parent_env['PYTHONPATH']}")


    for i, dataset_dir in enumerate(sorted(dataset_dirs)): # Ordena para consistência
        dataset_name = dataset_dir.name
        for j, model_name in enumerate(MODELS_TO_RUN):
            exp_num = i * len(MODELS_TO_RUN) + j + 1
            print(f"\n[{exp_num}/{total_experiments}] Iniciando: Modelo='{model_name}', Dataset='{dataset_name}'")

            # Constrói o comando
            cmd = [
                sys.executable, # Usa o mesmo python que está executando este script
                str(TABZILLA_EXP_SCRIPT.resolve()),
                '--model_name', model_name,
                '--dataset_dir', str(dataset_dir.resolve()),
                '--experiment_config', str(EXP_CONFIG_FILE.resolve())
            ]
            print(f"  Comando: {' '.join(cmd)}")

            try:
                # Executa o comando
                # stdout e stderr são capturados para não poluir demais a saída principal
                # Se precisar de debug, remova capture_output=True
                # O timeout pode ser útil para experimentos longos (em segundos)
                result = subprocess.run(
                    cmd,
                    check=True,
                    text=True,
                    capture_output=True,
                    env=tabzilla_parent_env, # Passa o PYTHONPATH modificado
                    # timeout=3600 # Exemplo: 1 hora de timeout
                )
                print(f"  Sucesso [{exp_num}/{total_experiments}]. Saída resumida:")
                # Imprime as últimas linhas da saída do subprocesso para feedback
                stdout_lines = result.stdout.strip().splitlines()
                stderr_lines = result.stderr.strip().splitlines()
                for line in stdout_lines[-5:]: print(f"    stdout: {line}")
                for line in stderr_lines[-5:]: print(f"    stderr: {line}")
                completed_count += 1

            except subprocess.CalledProcessError as e:
                print(f"  ERRO [{exp_num}/{total_experiments}]! O Experimento falhou (código de saída {e.returncode}).")
                print(f"    Comando: {' '.join(e.cmd)}")
                print(f"    Stderr (últimas 10 linhas):")
                for line in e.stderr.strip().splitlines()[-10:]: print(f"      {line}")
                failed_experiments.append({"model": model_name, "dataset": dataset_name, "error": "CalledProcessError"})
            except subprocess.TimeoutExpired as e:
                print(f"  ERRO [{exp_num}/{total_experiments}]! Timeout expirado.")
                print(f"    Comando: {' '.join(e.cmd)}")
                failed_experiments.append({"model": model_name, "dataset": dataset_name, "error": "TimeoutExpired"})
            except Exception as e:
                 print(f"  ERRO INESPERADO [{exp_num}/{total_experiments}] ao tentar executar o subprocesso!")
                 print(f"    Erro: {e}")
                 failed_experiments.append({"model": model_name, "dataset": dataset_name, "error": f"Unexpected: {e}"})


    print("-" * 30)
    print("--- Fase de Execução Concluída ---")
    exec_time = time.time() - start_time
    print(f"Tempo total de execução: {exec_time:.2f} segundos")
    print(f"Experimentos bem-sucedidos: {completed_count}/{total_experiments}")
    if failed_experiments:
        print(f"Experimentos com falha: {len(failed_experiments)}")
        # print("Falhas:", failed_experiments) # Descomentar para detalhes
    print("-" * 30)

    # 3. Agregação dos resultados JSON
    print("--- Iniciando Agregação de Resultados ---")
    all_results_list = []
    all_exceptions_list = []

    # Encontra todos os arquivos _results.json gerados
    json_files = glob.glob(str(RESULTS_OUTPUT_DIR / '**' / '*_results.json'), recursive=True)

    if not json_files:
        print("Aviso: Nenhum arquivo de resultado JSON encontrado para agregação.")
    else:
        print(f"Encontrados {len(json_files)} arquivos JSON para parsear.")
        for json_file in json_files:
            results, exceptions = parse_single_json_result(Path(json_file))
            all_results_list.extend(results)
            all_exceptions_list.extend(exceptions)

        print(f"Parseamento concluído. {len(all_results_list)} resultados de folds válidos e {len(all_exceptions_list)} exceções encontradas.")

        # Cria DataFrames
        if all_results_list:
            results_df = pd.DataFrame(all_results_list)
            # Ordena para melhor visualização
            results_df.sort_values(by=["dataset_name", "model_name", "alg_hparam_id", "fold"], inplace=True)
            # Salva em CSV
            try:
                results_df.to_csv(FINAL_CSV_RESULTS_FILE, index=False)
                print(f"Resultados agregados salvos em: {FINAL_CSV_RESULTS_FILE.resolve()}")
            except Exception as e:
                 print(f"Erro ao salvar CSV de resultados: {e}")
                 print("Salvando como backup: final_results_backup.csv")
                 results_df.to_csv("final_results_backup.csv", index=False)


        if all_exceptions_list:
            exceptions_df = pd.DataFrame(all_exceptions_list)
            exceptions_df.sort_values(by=["dataset_name", "model_name", "alg_hparam_id"], inplace=True)
            # Salva em CSV
            try:
                exceptions_df.to_csv(FINAL_CSV_EXCEPTIONS_FILE, index=False)
                print(f"Exceções agregadas salvas em: {FINAL_CSV_EXCEPTIONS_FILE.resolve()}")
            except Exception as e:
                 print(f"Erro ao salvar CSV de exceções: {e}")
                 print("Salvando como backup: final_exceptions_backup.csv")
                 exceptions_df.to_csv("final_exceptions_backup.csv", index=False)

    print("-" * 30)
    print("--- Script Concluído ---")

if __name__ == "__main__":
    main()