# experiment script for tabzilla
#
# this script runs an experiment specified by a config file

import argparse
import logging
import sys
import traceback
from collections import namedtuple
from pathlib import Path
from typing import NamedTuple

import optuna
import json

optuna.logging.set_verbosity(optuna.logging.ERROR)

# <<< ADICIONADO: Importar os para garantir compatibilidade de SO no nome do modelo para path
import os
# >>> FIM ADICIONADO

from models.basemodel import BaseModel
from tabzilla_alg_handler import ALL_MODELS, get_model
from tabzilla_datasets import TabularDataset
from tabzilla_utils import (
    ExperimentResult,
    cross_validation,
    get_experiment_parser,
    get_scorer,
)


class TabZillaObjective(object):
    """
    adapted from TabSurvey.train.Objective.
    this saves output from each trial.
    MODIFIED to support loading fixed hyperparameters for trial 0 via fixed_hparams_path.
    """

    def __init__(
        self,
        model_handle: BaseModel,
        dataset: TabularDataset,
        experiment_args: NamedTuple, # Espera-se que contenha 'fixed_hparams_path'
        hparam_seed: int,
        random_parameters: bool,
        time_limit: int,
    ):
        #  BaseModel handle that will be initialized and trained
        self.model_handle = model_handle

        self.dataset = dataset
        self.experiment_args = experiment_args # Contém todos os args parseados, incluindo o novo
        self.dataset.subset_random_seed = self.experiment_args.subset_random_seed
        # directory where results will be written
        self.output_path = Path(self.experiment_args.output_dir).resolve()

        # create the scorer, and get the direction of optimization from the scorer object
        sc_tmp = get_scorer(dataset.target_type)
        self.direction = sc_tmp.direction

        # if True, sample random hyperparameters. if False, sample using the optuna sampler object
        self.random_parameters = random_parameters

        # if random_parameters = True, then this is used to generate random hyperparameters
        self.hparam_seed = hparam_seed

        # time limit for any cross-validation cycle (seconds)
        self.time_limit = time_limit

    def __call__(self, trial: optuna.trial.Trial): # Adicionado type hint para clareza
        # --- Lógica de Seleção de Hiperparâmetros (MODIFICADA) ---
        hparam_source = "unknown" # Inicializa
        trial_params = {}       # Inicializa

        if self.random_parameters:
            # --- INÍCIO DA MODIFICAÇÃO ---
            # Verifica HPs fixos ESPECIFICAMENTE para o trial 0 no modo "random"
            # Isso permite que o orquestrador force HPs específicos usando n_random_trials=1
            # e passando o argumento --fixed_hparams_path
            if trial.number == 0 and hasattr(self.experiment_args, 'fixed_hparams_path') and self.experiment_args.fixed_hparams_path is not None:
                try:
                    hparam_file_path = Path(self.experiment_args.fixed_hparams_path)
                    if hparam_file_path.is_file():
                        print(f"  INFO: Attempting to load fixed hyperparameters from {hparam_file_path} for trial 0...")
                        with open(hparam_file_path, 'r') as f:
                            trial_params = json.load(f)
                        hparam_source = f"fixed_json:{hparam_file_path.name}"
                        print(f"  INFO: Trial 0 using fixed hyperparameters loaded from {hparam_file_path.name}: {trial_params}")
                    else:
                        print(f"  WARNING: Fixed hyperparameters file not found: {self.experiment_args.fixed_hparams_path}")
                        print("  WARNING: Falling back to default parameters for trial 0.")
                        trial_params = self.model_handle.default_parameters()
                        hparam_source = "default_fallback_file_not_found"

                except Exception as e:
                    print(f"  WARNING: Failed to load/parse fixed HPs from {self.experiment_args.fixed_hparams_path}: {e}")
                    print("  WARNING: Falling back to default parameters for trial 0.")
                    # Fallback se o JSON falhar
                    trial_params = self.model_handle.default_parameters()
                    hparam_source = "default_fallback_load_error"

            # --- FIM DA MODIFICAÇÃO ---

            # Lógica original para trial 0 se HPs fixos não foram fornecidos/carregados com sucesso
            elif trial.number == 0:
                trial_params = self.model_handle.default_parameters()
                hparam_source = "default"
                print(f"  INFO: Trial 0 using default hyperparameters: {trial_params}")
            # Lógica original para trials aleatórios subsequentes (se n_random_trials > 1)
            else:
                trial_params = self.model_handle.get_random_parameters(
                    trial.number + self.hparam_seed * 999
                )
                hparam_source = f"random_{trial.number}_s{self.hparam_seed}"
                print(f"  INFO: Trial {trial.number} using random hyperparameters: {trial_params}")

        else: # Modo de otimização do Optuna (n_opt_trials > 0)
            # Nenhuma mudança aqui para HPs fixos, Optuna controla via suggest_*
            # Passa experiment_args em vez de None, pode ser útil para alguns modelos
            trial_params = self.model_handle.define_trial_parameters(
                trial, self.experiment_args
            )
            hparam_source = f"sampler_{trial.number}"
            print(f"  INFO: Trial {trial.number} using Optuna suggested hyperparameters: {trial_params}")

        # --- Criação do Modelo e Execução (Sem alterações) ---

        # Cria namespace "args" para inicialização do modelo
        # (Certifique-se que a definição de arg_namespace está presente no escopo)
        arg_namespace = namedtuple(
            "args",
            [
                "model_name", "batch_size", "scale_numerical_features",
                "val_batch_size", "objective", "gpu_ids", "use_gpu",
                "epochs", "data_parallel", "early_stopping_rounds",
                "dataset", "cat_idx", "num_features", "subset_features",
                "subset_rows", "subset_features_method", "subset_rows_method",
                "cat_dims", "num_classes", "logging_period",
                # Adicione fixed_hparams_path aqui se algum modelo precisar dele diretamente
                # "fixed_hparams_path",
            ],
        )

        # Determina épocas máximas
        if hasattr(self.model_handle, "default_epochs"):
            max_epochs = self.model_handle.default_epochs
        else:
            max_epochs = self.experiment_args.epochs

        # Cria o objeto args
        args = arg_namespace(
            model_name=self.model_handle.__name__,
            batch_size=self.experiment_args.batch_size,
            val_batch_size=self.experiment_args.val_batch_size,
            scale_numerical_features=self.experiment_args.scale_numerical_features,
            epochs=max_epochs,
            gpu_ids=self.experiment_args.gpu_ids,
            use_gpu=self.experiment_args.use_gpu,
            data_parallel=self.experiment_args.data_parallel,
            early_stopping_rounds=self.experiment_args.early_stopping_rounds,
            logging_period=self.experiment_args.logging_period,
            objective=self.dataset.target_type,
            dataset=self.dataset.name,
            cat_idx=self.dataset.cat_idx,
            num_features=self.dataset.num_features,
            subset_features=self.experiment_args.subset_features,
            subset_rows=self.experiment_args.subset_rows,
            subset_features_method=self.experiment_args.subset_features_method,
            subset_rows_method=self.experiment_args.subset_rows_method,
            cat_dims=self.dataset.cat_dims,
            num_classes=self.dataset.num_classes,
            # fixed_hparams_path=self.experiment_args.fixed_hparams_path, # Descomente se necessário
        )

        # Instancia o modelo parametrizado
        # Passa os trial_params determinados acima
        model = self.model_handle(trial_params, args)

        # Cross valida os hiperparâmetros escolhidos
        result = None # Inicializa result
        obj_val = None  # Inicializa obj_val
        try:
            print(f"  INFO: Starting cross-validation for trial {trial.number}...")
            result = cross_validation(
                model,
                self.dataset,
                self.time_limit,
                scaler=args.scale_numerical_features,
                args=args, # Passa os args (incluindo info do dataset) para cross_validation
            )
            # Verifica se 'val' existe nos scorers antes de chamar get_objective_result
            if "val" in result.scorers and result.scorers["val"]:
                 obj_val = result.scorers["val"].get_objective_result()
                 print(f"  INFO: Cross-validation complete. Objective value: {obj_val}")
            else:
                 print("  WARNING: 'val' scorer not found or empty in results. Setting objective value to None.")
                 obj_val = None # Ou um valor padrão indicando falha, ex: float('-inf') para maximização

        except Exception as e:
            print(f"  ERROR: Caught exception during cross-validation for trial {trial.number}: {e}")
            tb = traceback.format_exc()
            # Cria um objeto ExperimentResult mesmo em caso de falha para salvar metadados
            # Garante que 'model' existe (foi instanciado antes do try)
            result = ExperimentResult(
                dataset=self.dataset,
                scaler=args.scale_numerical_features,
                model=model, # Salva o modelo instanciado
                timers={},
                scorers={},
                predictions=None,
                probabilities=None,
                ground_truth=None,
            )
            result.exception = tb # Salva o traceback
            # Define obj_val como None ou um valor indicando falha para o Optuna
            obj_val = None # Ou float('-inf') / float('inf') dependendo da direção
            print(tb)

        # --- Armazenamento do Resultado (Sem alterações, mas garante que 'result' existe) ---
        if result is not None:
            # Adiciona informações sobre hiperparâmetros e trial ao resultado
            result.hparam_source = hparam_source
            result.trial_number = trial.number
            # Salva os parâmetros realmente usados neste trial
            result.params_used = trial_params # Adiciona os parâmetros ao objeto de resultado
            # Salva os argumentos gerais do experimento
            # Corrigido typo: experiment_args em vez de experiemnt_args
            result.experiment_args = vars(self.experiment_args)

            # Lógica de salvamento do arquivo JSON (com tratamento de nome de arquivo)
            safe_dataset_name = self.dataset.name.replace(os.sep, "_").replace("/", "_") # Garante substituição
            safe_model_name = self.model_handle.__name__.replace(os.sep, "_").replace("/", "_")

            missing_info = ""
            if "_missing" in safe_dataset_name:
                parts = safe_dataset_name.split("_")
                for i, part in enumerate(parts):
                    if part.endswith("pct") and i + 1 < len(parts) and parts[i + 1] == "missing":
                        missing_info = f"_{part}"
                        break

            run_specific_output_dir = self.output_path / safe_dataset_name / safe_model_name
            run_specific_output_dir.mkdir(parents=True, exist_ok=True)

            result_file_base = run_specific_output_dir / f"{hparam_source}{missing_info}_trial{trial.number}"

            try:
                print(f"  INFO: Writing results to {result_file_base}_results.json...")
                result.write(
                    result_file_base,
                    write_predictions=self.experiment_args.write_predictions,
                    compress=False,
                )
            except AssertionError as e:
                if "file already exists" in str(e).lower(): # Checagem mais robusta
                    import time
                    unique_suffix = int(time.time() * 1000) % 10000
                    result_file_base = run_specific_output_dir / f"{hparam_source}{missing_info}_trial{trial.number}_run{unique_suffix}"
                    print(f"  WARNING: File exists, writing to alternative path: {result_file_base}_results.json")
                    result.write(
                        result_file_base,
                        write_predictions=self.experiment_args.write_predictions,
                        compress=False,
                    )
                else:
                    print(f"  ERROR: Failed to write results due to AssertionError: {e}")
                    raise # Re-levanta outras assertion errors
            except Exception as e:
                 print(f"  ERROR: Failed to write results: {e}")
                 # Não re-levanta para permitir que o Optuna continue, mas registra o erro

        else:
             print("  ERROR: Result object is None, cannot save results.")


        # Retorna o valor objetivo para o Optuna
        return obj_val


def iteration_callback(study, trial):
    print(f"Trial {trial.number + 1} complete")


def main(experiment_args, model_name, dataset_dir):
    # read dataset from folder
    dataset = TabularDataset.read(Path(dataset_dir).resolve())

    model_handle = get_model(model_name)

    # create results directory if it doesn't already exist (diretório base)
    output_path = Path(experiment_args.output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True) # O diretório base ainda é criado aqui

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

    if experiment_args.n_random_trials > 0:
        objective = TabZillaObjective(
            model_handle=model_handle,
            dataset=dataset, # Passa o objeto dataset carregado
            experiment_args=experiment_args,
            hparam_seed=experiment_args.hparam_seed,
            random_parameters=True,
            time_limit=experiment_args.trial_time_limit,
        )

        print(
            f"evaluating {experiment_args.n_random_trials} random hyperparameter samples..."
        )
        study = optuna.create_study(
            direction=objective.direction,
            study_name=None, # Cria um novo estudo para cada execução
            storage=None,
            load_if_exists=False, # Garante que não tenta reutilizar estudos anteriores
        )
        study.optimize(
            objective,
            n_trials=experiment_args.n_random_trials,
            timeout=experiment_args.experiment_time_limit,
            callbacks=[iteration_callback],
        )
        previous_trials = study.trials
    # <<< ADICIONADO: Garante que 'previous_trials' existe mesmo se n_random_trials=0 >>>
    elif experiment_args.n_opt_trials > 0:
         previous_trials = [] # Inicializa como vazio se não houver random trials
    # >>> FIM ADICIONADO
    else:
        # <<< ADICIONADO: Lógica para rodar apenas UM trial com HPs padrão se n_random=0 e n_opt=0 >>>
        print(f"Running single trial with default hyperparameters (n_random_trials=0, n_opt_trials=0)...")
        objective = TabZillaObjective(
            model_handle=model_handle,
            dataset=dataset,
            experiment_args=experiment_args,
            hparam_seed=experiment_args.hparam_seed, # Seed ainda pode ser relevante se o modelo usar aleatoriedade interna
            random_parameters=True, # Força usar a lógica do trial 0 para pegar defaults
            time_limit=experiment_args.trial_time_limit,
        )
        # Cria um "dummy" trial para chamar o objective uma vez
        dummy_study = optuna.create_study(direction=objective.direction)
        # O trial.number será 0, acionando a lógica de hparam_source='default'
        objective(dummy_study.ask())
        previous_trials = None # Nenhum trial para HPO
        # <<< FIM ADICIONADO >>>


    if experiment_args.n_opt_trials > 0:
        # TODO: this needs to be tested
        objective = TabZillaObjective(
            model_handle=model_handle,
            dataset=dataset,
            experiment_args=experiment_args,
            hparam_seed=experiment_args.hparam_seed,
            random_parameters=False,
            time_limit=experiment_args.trial_time_limit,
        )

        print(
            f"running {experiment_args.n_opt_trials} steps of hyperparameter optimization..."
        )
        study = optuna.create_study(
            direction=objective.direction,
            study_name=None,
            storage=None,
            load_if_exists=False,
        )
        # if random search was run, add these trials
        if previous_trials is not None and len(previous_trials) > 0: # <<< MODIFICADO: Checa se previous_trials não é None e não é vazio
            print(
                f"adding {len(previous_trials)} random trials to warm-start HPO" # Modificado para usar len()
            )
            study.add_trials(previous_trials)
        study.optimize(
            objective,
            n_trials=experiment_args.n_opt_trials,
            timeout=experiment_args.experiment_time_limit,
            callbacks=[iteration_callback], # <<< ADICIONADO callback aqui também >>>
        )

    # <<< MODIFICADO: A mensagem agora reflete a estrutura de subdiretórios >>>
    print(f"trials complete. results written to subdirectories within {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parser for tabzilla experiments")

    parser.add_argument(
        "--experiment_config",
        required=True,
        type=str,
        help="config file for parameter experiment args",
    )

    parser.add_argument(
        "--dataset_dir",
        required=True,
        type=str,
        help="directory containing pre-processed dataset.",
    )
    parser.add_argument(
        "--model_name",
        required=True,
        type=str,
        choices=ALL_MODELS,
        help="name of the algorithm",
    )
    args = parser.parse_args()
    print(f"ARGS: {args}")

    # now parse the dataset and search config files
    experiment_parser = get_experiment_parser()

    # <<< MODIFICADO: Passa args como uma lista, como esperado por parse_args >>>
    config_arg_list = ["-experiment_config", args.experiment_config]
    experiment_args = experiment_parser.parse_args(args=config_arg_list)
    # >>> FIM MODIFICADO
    print(f"EXPERIMENT ARGS: {experiment_args}")

    main(experiment_args, args.model_name, args.dataset_dir)