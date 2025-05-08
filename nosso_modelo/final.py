import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt  # Import para plotar no final

# ADICIONADO: tqdm para visualizar progresso
from tqdm import tqdm
torch.autograd.set_detect_anomaly(True)

import random          #  <-- novo

GLOBAL_SEED = 42

def set_global_seed(seed: int = GLOBAL_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_global_seed()  

  
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


class TabularEmbedder(nn.Module):
    """
    Classe que encapsula a criação de embeddings para dados tabulares:
      - Colunas categóricas: Usa nn.Embedding + LabelEncoder (com [MASK]/[NULL])
      - Colunas numéricas: MLP + embeddings especiais p/ [MASK] e [NULL].
      - Token [CLS] + Embedding de Posição.
    """
    def __init__(self, df, categorical_columns, numerical_columns,
                 dimensao=4, hidden_dim=32):
        super().__init__()
        
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        self.dimensao = dimensao
        self.hidden_dim = hidden_dim
        
        # -----------------------
        # 1) LabelEncoders
        #    (incluindo [MASK] e [NULL] no vocabulário de cada coluna categórica)
        # -----------------------
        self.label_encoders = {}
        for col in self.categorical_columns:
            le = LabelEncoder()
            
            # Coletamos as categorias originais + tokens especiais
            orig_vals = df[col].astype(str).unique()
            tokens_especiais = ["[MASK]", "[NULL]"]
            categorias = np.unique(np.concatenate([orig_vals, tokens_especiais]))
            
            # Ajustamos (fit) com tudo (originais + [MASK], [NULL])
            le.fit(categorias)
            self.label_encoders[col] = le
        
        # -----------------------
        # 2) Embeddings Categóricas
        # -----------------------
        self.num_categories = {
            col: len(self.label_encoders[col].classes_)
            for col in self.categorical_columns
        }
        
        self.embedding_layers = nn.ModuleDict({
            col: nn.Embedding(self.num_categories[col], self.dimensao)
            for col in self.categorical_columns
        })
        
        # -----------------------
        # 3) MLP p/ Numéricas
        # -----------------------
        self.mlp_layers = nn.ModuleDict({
            col: nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, dimensao)
            ) for col in self.numerical_columns
        })
        
        # -----------------------
        # 4) Embeddings Especiais p/ Numéricas
        # -----------------------
        self.special_embeddings = nn.ParameterDict()
        for col in self.numerical_columns:
            self.special_embeddings[f"{col}_mask"] = nn.Parameter(torch.randn(dimensao))
            self.special_embeddings[f"{col}_null"] = nn.Parameter(torch.randn(dimensao))
        
        # -----------------------
        # 5) Token [CLS]
        # -----------------------
        self.cls_token = nn.Parameter(torch.randn(dimensao))
        
        # -----------------------
        # 6) Embedding de Posição
        # -----------------------
        self.n_tokens = len(self.categorical_columns) + len(self.numerical_columns)
        self.pos_embedding_layer = nn.Embedding(self.n_tokens + 1, self.dimensao)

    def forward(self, df):
        """
        Para cada linha do DataFrame, gera o embedding resultante:
          1) Transforma cada coluna categórica em índices e passa no nn.Embedding.
          2) Para cada coluna numérica:
               - Se for valor real, passa pela MLP.
               - Se for [MASK], usa o embedding especial "mask".
               - Se for [NULL], usa o embedding especial "null".
          3) Concatena embeddings de todas as colunas (categ+num).
          4) Reformata para (batch_size, n_tokens, dimensao) e insere o [CLS] no topo.
          5) Gera índices de posição e soma o embedding de posição, retornando shape
             (batch_size, n_tokens+1, dimensao).
        """
        dev = self.cls_token.device

        # =============== #
        # 1) EMBEDDINGS DAS COLUNAS CATEGÓRICAS
        # =============== #
        cat_embeds_list = []
        for col in self.categorical_columns:
            col_values = df[col].astype(str).values
            indices = self.label_encoders[col].transform(col_values)
            indices_tensor = torch.tensor(indices, dtype=torch.long, device=dev)
            
            col_emb = self.embedding_layers[col](indices_tensor)
            cat_embeds_list.append(col_emb)
        
        if len(cat_embeds_list) > 0:
            cat_embeds = torch.cat(cat_embeds_list, dim=1)
        else:
            cat_embeds = None
        
        # =============== #
        # 2) EMBEDDINGS DAS COLUNAS NUMÉRICAS
        # =============== #
        numeric_values, mask_flags, null_flags = split_numeric_and_special(df, self.numerical_columns, device=dev)
        
        num_embeds_list = []
        for j, col in enumerate(self.numerical_columns):
            column_values = numeric_values[:, j].unsqueeze(1)
            column_mask   = mask_flags[:, j]
            column_null   = null_flags[:, j]
            
            # Passa pelo MLP
            mlp_output = self.mlp_layers[col](column_values)
            col_final  = torch.zeros_like(mlp_output)
            
            # Identifica índices mascarados / nulos / normais
            idx_mask   = (column_mask == True).nonzero(as_tuple=True)[0]
            idx_null   = (column_null == True).nonzero(as_tuple=True)[0]
            idx_normal = ((column_mask == False) & (column_null == False)).nonzero(as_tuple=True)[0]
            
            # Preenche
            col_final[idx_normal] = mlp_output[idx_normal]
            col_final[idx_mask]   = self.special_embeddings[f"{col}_mask"]
            col_final[idx_null]   = self.special_embeddings[f"{col}_null"]
            
            num_embeds_list.append(col_final)
        
        if len(num_embeds_list) > 0:
            num_embeds = torch.cat(num_embeds_list, dim=1)
        else:
            num_embeds = None
        
        # =============== #
        # 3) CONCATENAR CATEG + NUM
        # =============== #
        if cat_embeds is not None and num_embeds is not None:
            final_embeddings = torch.cat([cat_embeds, num_embeds], dim=1)
        elif cat_embeds is not None:
            final_embeddings = cat_embeds
        else:
            final_embeddings = num_embeds
        
        # =============== #
        # 4) INSERIR O TOKEN [CLS] E REFORMULAR
        # =============== #
        batch_size = final_embeddings.shape[0]
        
        n_tokens = 0
        if cat_embeds is not None:
            n_tokens += len(self.categorical_columns)
        if num_embeds is not None:
            n_tokens += len(self.numerical_columns)
        
        final_embeddings = final_embeddings.view(batch_size, n_tokens, self.dimensao)
        
        cls_token_expanded = self.cls_token.unsqueeze(0).unsqueeze(1).expand(batch_size, 1, self.dimensao)
        final_embeddings = torch.cat([cls_token_expanded, final_embeddings], dim=1)
        
        # =============== #
        # 5) SOMAR EMBEDDING DE POSIÇÃO
        # =============== #
        seq_len = final_embeddings.shape[1]
        pos_indices = torch.arange(seq_len, device=dev).unsqueeze(0).expand(batch_size, seq_len)
        pos_embeds = self.pos_embedding_layer(pos_indices)
        
        final_embeddings = final_embeddings + pos_embeds
        
        return final_embeddings

def initialize_weight(x):
    nn.init.xavier_uniform_(x.weight)
    if x.bias is not None:
        nn.init.constant_(x.bias, 0)


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate):
        super().__init__()
        self.layer1 = nn.Linear(hidden_size, filter_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(filter_size, hidden_size)
        initialize_weight(self.layer1)
        initialize_weight(self.layer2)

    def forward(self, x):
        x2 = self.layer1(x)
        x2 = self.relu(x2)
        x2 = self.dropout(x2)
        x2 = self.layer2(x2)
        return x2


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout_rate):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        self.wq = nn.Linear(hidden_size, hidden_size, bias=False)
        self.wk = nn.Linear(hidden_size, hidden_size, bias=False)
        self.wv = nn.Linear(hidden_size, hidden_size, bias=False)
        initialize_weight(self.wq)
        initialize_weight(self.wk)
        initialize_weight(self.wv)
        self.att_dropout = nn.Dropout(dropout_rate)
        self.out = nn.Linear(hidden_size, hidden_size, bias=False)
        initialize_weight(self.out)

    def forward(self, x, mask=None):
        B, T, _ = x.size()
        q = self.wq(x).view(B, T, self.num_heads, self.head_dim).transpose(1,2)
        k = self.wk(x).view(B, T, self.num_heads, self.head_dim).transpose(1,2)
        v = self.wv(x).view(B, T, self.num_heads, self.head_dim).transpose(1,2)
        # q,k: B, h, T, head_dim; v: same
        scores = (q @ k.transpose(-2,-1)) * self.scale  # B,h,T,T
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        attn = self.att_dropout(attn)
        out = attn @ v  # B,h,T,head_dim
        out = out.transpose(1,2).contiguous().view(B, T, -1)
        return self.out(out)


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, filter_size, num_heads, dropout_rate):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.mha = MultiHeadAttention(hidden_size, num_heads, dropout_rate)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, filter_size, dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x, mask=None):
        x2 = self.norm1(x)
        attn = self.mha(x2, mask)
        x = x + self.dropout1(attn)
        x2 = self.norm2(x)
        ffn = self.ffn(x2)
        x = x + self.dropout2(ffn)
        return x

class TabularTransformerEncoder(nn.Module):
    def __init__(self, d_model=4, nhead=2, num_layers=2,
                 dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.pre_norm = nn.LayerNorm(d_model)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, dim_feedforward, nhead, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, src_key_padding_mask=None):
        # x: B, T, d_model
        mask = src_key_padding_mask  # True for pads
        x = self.pre_norm(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x


# ========================================================================
# NOVA CLASSE: TabularBertPretrainer (PRÉ-TREINAMENTO)
# ========================================================================
# ========================================================================
# PRÉ‑TREINADOR baseado em reconstrução de EMBEDDINGS
# ========================================================================
class TabularBertPretrainer(nn.Module):
    """
    Dado um DataFrame mascarado, pede ao Transformer que reconstrua,
    apenas nas posições [MASK], o mesmo vetor de embedding que haveria
    se a coluna não estivesse mascarada.

    O alvo é o embedding (detach) produzido pelo próprio TabularEmbedder
    quando recebe o DF original (sem [MASK]).
    """
    def __init__(self, embedder: TabularEmbedder, transformer: TabularTransformerEncoder):
        super().__init__()
        self.embedder = embedder
        self.transformer = transformer
        self.d_model = embedder.dimensao
        self.eps = 1e-8   # para evitar divisão por zero na normalização opcional

    def forward(self, df_masked: pd.DataFrame, df_original: pd.DataFrame):
        device = next(self.parameters()).device

        # 1) Embeddings “corrompidos” (entrada do Transformer)
        emb_in = self.embedder(df_masked)              # (B, L, d)
        # 2) Embeddings “puros” (alvo) — detach p/ não retro‑propagar neles
        emb_target = self.embedder(df_original).detach()  # (B, L, d)

        # 3) Passa pelo Transformer
        encoded = self.transformer(emb_in)             # (B, L, d)

        # 4) Construir máscara booleana de onde havia [MASK]
        #    → shape (B, L‑1)   (ignoramos CLS na coluna 0)
        mask_matrix = []
        for col in self.embedder.categorical_columns + self.embedder.numerical_columns:
            mask_matrix.append((df_masked[col] == "[MASK]").values[:, None])  # shape (B,1)
        mask_matrix = np.concatenate(mask_matrix, axis=1)                    # (B, L‑1)
        mask_tensor = torch.tensor(mask_matrix, device=device, dtype=torch.bool)

        # 5) Selecionar apenas posições mascaradas (flatten)
        enc_sel  = encoded[:, 1:, :][mask_tensor]      # (N_mask, d)
        tgt_sel  = emb_target[:, 1:, :][mask_tensor]   # (N_mask, d)
        #enc_sel = nn.functional.normalize(enc_sel, dim=-1)
        #tgt_sel = nn.functional.normalize(tgt_sel, dim=-1)
        if enc_sel.numel() == 0:          # nenhum [MASK] no batch
            print("a")
            return torch.tensor(0., device=device, requires_grad=True), {}

        # 6) Perda = MSE entre vetores
        loss = nn.functional.mse_loss(enc_sel, tgt_sel)

        return loss, {"mse_embedding": loss.item()}



class TabularBERTModel(nn.Module):
    """
    Modelo unificado para a tarefa de classificação:
      1) Gera embeddings tabulares (TabularEmbedder).
      2) Passa pelos encoders do Transformer (TabularTransformerEncoder).
      3) Pega o [CLS] e passa por uma camada linear p/ classificação (nn.Linear).
    """
    def __init__(self, embedder, transformer, num_labels=2, class_weights=None):
        """
        Parâmetros
        ----------
        embedder : TabularEmbedder
            Responsável por gerar as embeddings (dimensao = d_model).
        transformer : TabularTransformerEncoder
            Encoder bidirecional estilo BERT.
        num_labels : int
            Número de classes para classificação. Se for 2 => binária.
        class_weights : torch.Tensor, opcional
            Pesos de classe para a CrossEntropy, se quiser tratar desbalanceamento.
        """
        super().__init__()
        self.embedder = embedder
        self.transformer = transformer
        self.num_labels = num_labels
        
        self.classifier = nn.Sequential(
            nn.Linear(embedder.dimensao, embedder.dimensao // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(embedder.dimensao // 2, num_labels),
        )
        
        # Armazenar os pesos de classe, se fornecidos
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None
    
    def forward(self, df, labels=None):
        """
        df : pd.DataFrame
            DataFrame de entrada (possivelmente mascarado/nulo, mas no fine-tuning geralmente sem máscara).
        labels : Tensor (opcional), shape (batch_size,) com classes verdadeiras, 
                 se quisermos calcular a perda CrossEntropy.

        Retorna
        -------
        logits : torch.Tensor
            shape (batch_size, num_labels)
        loss (opcional) : torch.Tensor
            Se 'labels' for fornecido, retorna também a perda CrossEntropyLoss.
        """
        x = self.embedder(df)                # (batch_size, seq_len, d_model)
        encoded_output = self.transformer(x) # (batch_size, seq_len, d_model)
        
        # Extraímos o [CLS], que está em encoded_output[:, 0, :]
        cls_representation = encoded_output[:, 0, :]  # (batch_size, d_model)
        
        # Classificação final
        logits = self.classifier(cls_representation)  # (batch_size, num_labels)
        
        loss = None
        if labels is not None:
            # Se temos pesos de classe, aplicamos na CrossEntropy
            if self.class_weights is not None:
                criterion = nn.CrossEntropyLoss(weight=self.class_weights)
            else:
                criterion = nn.CrossEntropyLoss()
            loss = criterion(logits, labels)
            return logits, loss
        
        return logits


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


# =====================================================================
# Exemplo de Uso: Pré-Treinamento + Fine Tuning
# =====================================================================
import json
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix

# Import your custom modules here
# from tabular_embedder import TabularEmbedder
# from tabular_transformer import TabularTransformerEncoder
# from tabular_bert import TabularBertPretrainer, TabularBERTModel
# from preprocess_table import preprocess_table

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Usando device:", device)

    # Paths to your data directories
    datasets_dir = "/scratch/pedrobacelar/tabular-transformer/pedro/processed_datasets"  # Update with actual path
    splits_dir = "/scratch/pedrobacelar/tabular-transformer/pedro/processed_datasets/splits"      # Update with actual path
    hyperparams_dir = "/scratch/pedrobacelar/tabular-transformer/models/best_hyper_tabbert"  # Update with actual path
    categorical_cols_dir = "/scratch/pedrobacelar/tabular-transformer/pedro/cat_num"  # Update with actual path
    results_dir = "/scratch/pedrobacelar/tabular-transformer/models/results"    # Update with actual path

    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)

    # List of dataset base names (without _XXnan.csv)
    dataset_bases = [

        "credit-g",
        "electricity",
        "kr-vs-kp",
        "letter",
        "nomao",
        "pendigits",
        "spambase",
        "vehicles"
        # Add all your dataset base names here
    ]

    # Missing values percentages to process
    nan_percentages = ["00", "20", "40", "60", "80"]

    # Create results dataframe and setup continuous saving
    all_results = []
    results_csv_path = os.path.join(results_dir, "all_model_results.csv")
    
    # Check if results file already exists to append to it
    if os.path.exists(results_csv_path):
        existing_results = pd.read_csv(results_csv_path)
        all_results = existing_results.to_dict('records')
        print(f"Loaded {len(all_results)} existing results from {results_csv_path}")

    # Process each dataset with each missing values percentage
    for dataset_base in dataset_bases:
        print(f"\n{'='*80}")
        print(f"Processing dataset: {dataset_base}")
        print(f"{'='*80}")

        # Load categorical columns for this dataset
        categorical_file = os.path.join(categorical_cols_dir, f"{dataset_base}.txt")
        categorical_columns = []
        if os.path.exists(categorical_file):
            with open(categorical_file, 'r') as f:
                content = f.read().strip()
                if content:  # Only parse if not empty
                    categorical_columns = content.split(',')

        # Load hyperparameters for this dataset
        hyperparams_file = os.path.join(hyperparams_dir, f"{dataset_base}_best_hyper.txt")
        with open(hyperparams_file, 'r') as f:
            hyperparams = {}
            for line in f:
                if "=" in line:
                    key, value = line.strip().split("=", 1)  # Split on first = only
                    key = key.strip()
                    value = value.strip()
                    
                    # Convert numeric values more carefully
                    try:
                        # Try float conversion first
                        if value.lower() in ['true', 'false']:
                            hyperparams[key] = value.lower() == 'true'
                        elif '.' in value or 'e' in value.lower():  # Handle scientific notation
                            hyperparams[key] = float(value)
                        else:
                            try:
                                hyperparams[key] = int(value)
                            except ValueError:
                                hyperparams[key] = float(value)  # Try float again as fallback
                    except ValueError:
                        # Keep as string if conversion fails
                        hyperparams[key] = value
                        
            print(f"Loaded hyperparameters for {dataset_base}:")
            for k, v in hyperparams.items():
                print(f"  {k} = {v} ({type(v).__name__})")

        # Set hyperparameters from the file
        DIM = hyperparams.get("DIM", 128)
        HIDDEN_DIM = hyperparams.get("HIDDEN_DIM", 128)
        HEADS = hyperparams.get("HEADS", 8)
        LAYERS = hyperparams.get("LAYERS", 8)
        DIM_FEED = hyperparams.get("DIM_FEED", 256)
        DROPOUT = hyperparams.get("DROPOUT", 0.3)
        EPOCHS_PRE = hyperparams.get("EPOCHS_PRE", 50)
        BATCH = hyperparams.get("BATCH", 256)
        LR_PRE = hyperparams.get("LR_PRE", 3e-4)
        WEIGHT_DECAY_PRE = hyperparams.get("WEIGHT_DECAY_PRE", 1e-2)
        PROB_MASCARA = hyperparams.get("PROB_MASCARA", 0.3)
        EPOCH_FINE = hyperparams.get("EPOCH_FINE", 50)
        LR_FINE = hyperparams.get("LR_FINE", 1e-3)
        WEIGHT_DECAY_FINE = hyperparams.get("WEIGHT_DECAY_FINE", 1e-4)

        # Load splits once per dataset base
        split_file = os.path.join(splits_dir, f"{dataset_base}_split.json")
        with open(Path(split_file)) as f:
            splits = json.load(f)

        train_idx = np.array(splits["train_indices"])
        val_idx = np.array(splits["val_indices"])
        test_idx = np.array(splits["test_indices"])

        # Process each missing values percentage
        for nan_percentage in nan_percentages:
            print(f"\n{'-'*40}")
            print(f"Processing {dataset_base} with {nan_percentage}% missing values")
            print(f"{'-'*40}")

            # Load dataset with specific missing values percentage
            df_file = os.path.join(datasets_dir, f"{dataset_base}_{nan_percentage}nan.csv")
            df = pd.read_csv(df_file)

            # Identify label column (assuming last column is the label)
            label_column = "class"
            if label_column not in df.columns:
                # Try to automatically detect the label column
                # Assuming it might be named differently or be the last column
                label_column = df.columns[-1]
                print(f"Using {label_column} as the label column")

            # Encode the label
            label_encoder = LabelEncoder()
            df[label_column] = label_encoder.fit_transform(df[label_column])

            # Show label mapping
            print("=== Mapeamento de labels ===")
            for idx, cls in enumerate(label_encoder.classes_):
                print(f"{idx} → {cls}")
            print("============================")

            # Get number of unique labels
            LABELS = len(label_encoder.classes_)

            # Get all columns except label column
            all_cols = df.columns.tolist()
            all_cols.remove(label_column)
            numerical_columns = [c for c in all_cols if c not in categorical_columns]

            # Normalize numerical columns
            if numerical_columns:
                scaler = StandardScaler()
                df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
            else:
                scaler = None

            # ===============================================================
            # 1) Dataset for PRE-TRAINING - features only (no label)
            # ===============================================================

            df_pretrain = df.drop(columns=[label_column]).copy()

            df_train_orig = df_pretrain.iloc[train_idx].reset_index(drop=True)
            df_val_orig = df_pretrain.iloc[val_idx].reset_index(drop=True)

            # ===============================================================
            # 2) Pre-training model
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
            # 3) PRE-TRAINING LOOP with DYNAMIC MASKS
            # ===============================================================
            print("\n=== Iniciando Pré-Treinamento (máscaras novas a cada época) ===")
            for epoch in range(num_epochs):
                # Generate new masks for entire dataset
                df_train_masked = preprocess_table(
                    df_train_orig.copy(), p_base=PROB_MASCARA, fine_tunning=False)
                df_val_masked = preprocess_table(
                    df_val_orig.copy(), p_base=PROB_MASCARA, fine_tunning=False)

                df_train_size = len(df_train_masked)
                df_val_size = len(df_val_masked)

                # Training
                pretrain_model.train()
                indices = torch.randperm(df_train_size)
                train_loss_sum = 0.0
                train_steps = 0

                for start in tqdm(range(0, df_train_size, batch_size),
                                desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch"):
                    end = start + batch_size
                    batch_idx = indices[start:end]

                    batch_masked = df_train_masked.iloc[batch_idx].reset_index(drop=True)
                    batch_orig = df_train_orig.iloc[batch_idx].reset_index(drop=True)

                    optimizer.zero_grad()
                    total_loss, _ = pretrain_model(batch_masked, batch_orig)
                    total_loss.backward()
                    optimizer.step()
                    scheduler.step()

                    train_loss_sum += total_loss.item()
                    train_steps += 1

                avg_train_loss = train_loss_sum / train_steps
                train_losses_history.append(avg_train_loss)

                # Validation
                pretrain_model.eval()
                with torch.no_grad():
                    val_loss_sum = 0.0
                    val_steps = 0
                    for start in range(0, df_val_size, batch_size):
                        end = start + batch_size
                        batch_masked = df_val_masked.iloc[start:end].reset_index(drop=True)
                        batch_orig = df_val_orig.iloc[start:end].reset_index(drop=True)

                        total_loss, _ = pretrain_model(batch_masked, batch_orig)
                        val_loss_sum += total_loss.item()
                        val_steps += 1

                    avg_val_loss = val_loss_sum / val_steps
                    val_losses_history.append(avg_val_loss)

                print(f"Epoch [{epoch+1}/{num_epochs}]  Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")

            # Plot loss curves
            plt.figure(figsize=(8, 6))
            plt.plot(train_losses_history, label='Train Loss')
            plt.plot(val_losses_history, label='Val Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'Pré-Treino MLM - {dataset_base}_{nan_percentage}nan')
            plt.legend()
            plt.savefig(os.path.join(results_dir, f"{dataset_base}_{nan_percentage}nan_pretrain_loss.png"))
            plt.close()

            # =================================================
            # 4) FINE TUNING (Classification)
            # =================================================
            print("\n=== Iniciando Fine-Tuning (Classificação) ===")

            df_train = df.iloc[train_idx].reset_index(drop=True)
            df_val = df.iloc[val_idx].reset_index(drop=True)
            df_test = df.iloc[test_idx].reset_index(drop=True)

            y_train = torch.tensor(df_train[label_column].values, dtype=torch.long, device=device)
            y_val = torch.tensor(df_val[label_column].values, dtype=torch.long, device=device)
            y_test = torch.tensor(df_test[label_column].values, dtype=torch.long, device=device)

            # Drop label to create feature DataFrames
            df_train_features = df_train.drop(columns=[label_column])
            df_val_features = df_val.drop(columns=[label_column])
            df_test_features = df_test.drop(columns=[label_column])

            # Preprocess without masks for fine-tuning
            processed_train = preprocess_table(df_train_features, p_base=0.0, fine_tunning=True)
            processed_val = preprocess_table(df_val_features, p_base=0.0, fine_tunning=True)
            processed_test = preprocess_table(df_test_features, p_base=0.0, fine_tunning=True)

            # Reuse embedder and transformer from pretrain_model
            finetune_embedder = pretrain_model.embedder
            finetune_transformer = pretrain_model.transformer

            # Calculate class weights (optional)
            class_weights = compute_class_weight(
                class_weight='balanced',
                classes=np.unique(y_train.cpu().numpy()),
                y=y_train.cpu().numpy()
            )
            class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)

            # Build classification model with pre-trained modules
            classifier_model = TabularBERTModel(
                embedder=finetune_embedder,
                transformer=finetune_transformer,
                num_labels=LABELS,
                class_weights=class_weights
            ).to(device)

            # Fine-tuning optimizer
            optimizer_ft = optim.AdamW(classifier_model.parameters(), lr=LR_FINE, weight_decay=WEIGHT_DECAY_FINE)
            num_epochs_ft = EPOCH_FINE
            batch_size_ft = BATCH
            scheduler_ft = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer_ft, T_max=num_epochs_ft)
            train_losses_history = []
            val_losses_history = []

            import copy
            import torch

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

                # Validation
                classifier_model.eval()
                with torch.no_grad():
                    logits_val, val_loss = classifier_model(processed_val, labels=y_val)
                    preds_val = torch.argmax(logits_val, dim=1)

                val_loss_value = val_loss.item()
                val_losses_history.append(val_loss_value)

                preds_val_np = preds_val.cpu().numpy()
                y_val_np = y_val.cpu().numpy()

                # Calculate F1-score for each class individually
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

            # Evaluate on test set
            classifier_model.eval()
            with torch.no_grad():
                logits_test, test_loss = classifier_model(processed_test, labels=y_test)
                preds_test = torch.argmax(logits_test, dim=1)

            preds_test_np = preds_test.cpu().numpy()
            y_test_np = y_test.cpu().numpy()

            # Calculate metrics
            acc_test = accuracy_score(y_test_np, preds_test_np)
            precision_macro = precision_score(y_test_np, preds_test_np, average='macro')
            recall_macro = recall_score(y_test_np, preds_test_np, average='macro')
            f1_test_per_class = f1_score(y_test_np, preds_test_np, average=None)
            f1_test_macro = f1_score(y_test_np, preds_test_np, average='macro')

            print("\n=== Resultados no Conjunto de Teste ===")
            print(f"Acurácia geral: {acc_test:.4f}")
            print(f"Precision macro: {precision_macro:.4f}")
            print(f"Recall macro: {recall_macro:.4f}")
            print(f"F1 macro: {f1_test_macro:.4f}")
            print("\n--- F1 por classe (rótulo original) ---")
            for cls, f1 in zip(label_encoder.classes_, f1_test_per_class):
                print(f"F1({cls}): {f1:.4f}")
            print("----------------------------------------")

            # Save results based on number of classes
            result_dict = {
                'dataset': dataset_base,
                'nan_percentage': nan_percentage,
                'accuracy': acc_test,
                'precision_macro': precision_macro,
                'recall_macro': recall_macro,
                'f1_macro': f1_test_macro,
            }

            if LABELS == 2:
                # Calculate confusion matrix values for binary classification
                tn, fp, fn, tp = confusion_matrix(y_test_np, preds_test_np).ravel()
                result_dict.update({
                    'TN': tn,
                    'FP': fp,
                    'FN': fn,
                    'TP': tp,
                    'f1_class0': f1_test_per_class[0],
                    'f1_class1': f1_test_per_class[1],
                })

            all_results.append(result_dict)
            
            # Save results after each dataset+nan_percentage combination is processed
            print(f"Saving results for {dataset_base}_{nan_percentage}nan...")
            results_df = pd.DataFrame(all_results)
            results_df.to_csv(results_csv_path, index=False)
            print(f"Results saved to {results_csv_path}")

    # Final save of all results (although they were already saved incrementally)
    print(f"\nAll processing complete. {len(all_results)} total results saved to {results_csv_path}")