import pandas as pd
import numpy as np
import torch
import random
from sklearn.model_selection import train_test_split

GLOBAL_SEED = 42

def set_global_seed(seed: int = GLOBAL_SEED):
    """
    Define a semente global para garantir reprodutibilidade
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
def preprocess_table(data, null_token="[NULL]", p_base=0.15, fine_tunning=False):
    """
    Pré-processa a tabela para substituir valores nulos pelo token `[NULL]`,
    aplica masking dinâmico com probabilidade ajustada e garante que:
        - Pelo menos uma máscara seja aplicada aleatoriamente por linha (se não estiver em fine_tuning).
        - A máscara dinâmica não sobreponha valores nulos (não mascara posições já nulas).

    Parâmetros
    ----------
    data : pd.DataFrame
        Tabela de dados original.
    null_token : str
        Token para substituir valores nulos (padrão = "[NULL]").
    p_base : float
        Probabilidade base para o mascaramento dinâmico.
    fine_tunning : bool
        Define se é pré-treinamento ou fine-tuning:
          - Se False, de fato mascaramos com probabilidade p_base.
          - Se True, não mascaramos nada (só substituímos nulos por [NULL]).

    Retorna
    -------
    masked_data : pd.DataFrame
        DataFrame resultante, contendo tokens `[NULL]` e `[MASK]`.
    """
    # Cria uma matriz booleana indicando valores nulos
    null_matrix = data.isnull()

    # Substitui valores nulos pelo token `[NULL]`
    processed_data = data.mask(null_matrix, null_token)

    if not fine_tunning:
        # Calcula a proporção de valores nulos por linha
        prop_nulls = null_matrix.sum(axis=1) / data.shape[1]

        # Ajusta a probabilidade de mascaramento dinâmico com base na proporção de nulos
        # Exemplo: se uma linha tem 30% de nulos, então p_base * (1 - 0.3) => 70% de p_base
        p_dynamic = p_base * (1 - prop_nulls.values[:, None])

        # Gera uma matriz aleatória e aplica a máscara dinâmica
        dynamic_mask = np.random.rand(*data.shape) < p_dynamic

        # Evita mascarar valores que já são nulos
        dynamic_mask[null_matrix.values] = False

        # Garante que cada linha tenha pelo menos um valor mascarado
        no_mask_rows = ~dynamic_mask.any(axis=1)
        for i in np.where(no_mask_rows)[0]:
            non_null_indices = np.where(~null_matrix.iloc[i].values)[0]
            if len(non_null_indices) > 0:
                random_index = np.random.choice(non_null_indices)
                dynamic_mask[i, random_index] = True

        # Aplica a máscara
        masked_data = processed_data.mask(dynamic_mask, "[MASK]")
        return masked_data
    else:
        # Se estamos em fine_tuning, não mascaramos aleatoriamente
        # Apenas retornamos com [NULL] no lugar dos nulos
        return processed_data

def split_numeric_and_special(df, numerical_columns, device='cuda'):
    """
    Separa valores numéricos reais dos tokens especiais `[MASK]` e `[NULL]`.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame que já contém tokens [MASK] e [NULL] em colunas numéricas.
    numerical_columns : list
        Lista de colunas consideradas numéricas.
    device : str
        'cuda' ou 'cpu', dependendo de onde rodará o tensor.

    Retorna
    -------
    numeric_values : torch.FloatTensor
        Tensor (n_linhas, n_num_cols) com valores numéricos (0.0 onde há tokens especiais).
    mask_flags : torch.BoolTensor
        Tensor (n_linhas, n_num_cols) com True onde há `[MASK]`.
    null_flags : torch.BoolTensor
        Tensor (n_linhas, n_num_cols) com True onde há `[NULL]`.
    """
    numeric_values = []
    mask_flags = []
    null_flags = []
    
    for col in numerical_columns:
        col_data = df[col].values  # Pode ser float ou string "[MASK]" ou "[NULL]"
        col_numeric = []
        col_mask = []
        col_null = []
        
        for val in col_data:
            if val == "[MASK]":
                col_numeric.append(0.0)
                col_mask.append(True)
                col_null.append(False)
            elif val == "[NULL]":
                col_numeric.append(0.0)
                col_mask.append(False)
                col_null.append(True)
            else:
                col_numeric.append(float(val))
                col_mask.append(False)
                col_null.append(False)
        
        numeric_values.append(col_numeric)
        mask_flags.append(col_mask)
        null_flags.append(col_null)
    
    # Converte de listas para tensores e transpõe para (n_linhas, n_num_cols)
    numeric_values = torch.tensor(numeric_values, dtype=torch.float32, device=device).T
    mask_flags     = torch.tensor(mask_flags,     dtype=torch.bool,   device=device).T
    null_flags     = torch.tensor(null_flags,     dtype=torch.bool,   device=device).T
    
    return numeric_values, mask_flags, null_flags

def create_pretrain_datasets(df, categorical_columns, numerical_columns,
                            p_base=0.15, test_size=0.1, random_state=42):
    """
    Exemplo de função que:
      1) Separa train/val (ex: 90/10)
      2) Gera DF mascarado (df_masked) e DF original (df_original).
      3) Retorna (df_train_original, df_train_masked, df_val_original, df_val_masked).

    Assim conseguimos fazer o pré-treinamento, comparando df_masked e df_original.
    """
    df_train, df_val = train_test_split(df, test_size=test_size, random_state=random_state)
    df_train = df_train.reset_index(drop=True)
    df_val   = df_val.reset_index(drop=True)
    
    # Gera versões mascaradas (com p_base=0.15)
    df_train_masked = preprocess_table(df_train.copy(), p_base=p_base, fine_tunning=False)
    df_val_masked   = preprocess_table(df_val.copy(),   p_base=p_base, fine_tunning=False)
    
    # df_train e df_val são as versões "originais"
    return df_train, df_train_masked, df_val, df_val_masked 