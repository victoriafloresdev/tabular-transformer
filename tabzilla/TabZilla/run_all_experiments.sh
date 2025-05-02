#!/bin/bash

# --- Configurações ---
# Certifique-se que o ambiente virtual TabZilla está ativo ou ative aqui:
# source /path/to/your/tabzilla/venv/bin/activate
# source ./tabzilla/bin/activate # Se o venv chama tabzilla e está na pasta raiz

CONFIG_FILE="tabzilla_experiment_config.yml" # Nome do seu arquivo de config

# Seus 5 modelos selecionados
MODELS=(
  "RandomForest"
  "XGBoost"
  "LightGBM"
  "rtdl_FTTransformer"
  "SAINT"
)

# Nomes base dos seus 5 datasets
DATASET_BASES=(
  "openml__credit-g__31"
  "openml__diabetes__37"
  "openml__adult__7592"
  "openml__spambase__43"
  "openml__Amazon_employee_access__34539"
)

# Sufixos das pastas de porcentagem
PERCENTAGES=("100pct" "75pct" "50pct" "25pct")

TABZILLA_SCRIPT_PATH="TabZilla/tabzilla_experiment.py" # Caminho para o script principal
DATASET_ROOT_DIR="TabZilla/datasets" # Raiz onde as pastas dos datasets estão
# ---------------------

echo "Iniciando execução de todos os experimentos..."
START_TIME=$SECONDS

# Loop principal
for model in "${MODELS[@]}"; do
  for dataset_base in "${DATASET_BASES[@]}"; do
    for pct in "${PERCENTAGES[@]}"; do
      dataset_dir="${DATASET_ROOT_DIR}/${dataset_base}_${pct}"

      # Verifica se o diretório do dataset existe
      if [ ! -d "$dataset_dir" ]; then
        echo "AVISO: Diretório do dataset não encontrado, pulando: ${dataset_dir}"
        continue
      fi

      echo "-----------------------------------------------------"
      echo "EXECUTANDO: Modelo=${model}, Dataset=${dataset_base}_${pct}"
      echo "Comando: python ${TABZILLA_SCRIPT_PATH} --experiment_config ${CONFIG_FILE} --model_name ${model} --dataset_dir ${dataset_dir}"
      echo "-----------------------------------------------------"

      # Executa o experimento TabZilla
      python ${TABZILLA_SCRIPT_PATH} \
        --experiment_config ${CONFIG_FILE} \
        --model_name "${model}" \
        --dataset_dir "${dataset_dir}"

      # Verificação simples de erro
      if [ $? -ne 0 ]; then
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "ERRO durante experimento: Modelo=${model}, Dataset=${dataset_dir}"
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        # Você pode querer parar o script aqui ou apenas logar o erro
        # exit 1 # Descomente para parar em caso de erro
      fi

    done # Fim loop percentages
  done # Fim loop datasets
done # Fim loop modelos

ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo "====================================================="
echo "TODOS OS EXPERIMENTOS CONCLUÍDOS!"
echo "Tempo total de execução: $(($ELAPSED_TIME / 3600))h $((($ELAPSED_TIME / 60) % 60))m $(($ELAPSED_TIME % 60))s"
echo "Resultados salvos em: $(grep output_dir ${CONFIG_FILE} | awk '{print $2}')"
echo "====================================================="