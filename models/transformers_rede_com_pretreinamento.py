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
if __name__ == "__main__":
    # ===============================================================
    # 0) Configurações gerais
    # ===============================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Usando device:", device)

    df = pd.read_csv("/scratch/pedro.bacelar/TabularTransformers/Datasets/dataset_com_nulo/cardio_tabzilla_nan75.csv")
    df = df.drop(columns="a")
    
    categorical_columns = ["cholesterol","gluc","smoke","alco","active"]
    label_column = "cardio"

    label_encoder = LabelEncoder()
    df[label_column] = label_encoder.fit_transform(df[label_column])

    all_cols = df.columns.tolist()
    all_cols.remove(label_column)
    numerical_columns = [c for c in all_cols if c not in categorical_columns]
    
    from sklearn.preprocessing import StandardScaler          #  ← novo import (já no topo))
    # ⇣⇣  NOVO  ⇣⇣  — normaliza atributos contínuos
    scaler = StandardScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    # ⇡⇡  NOVO  ⇡⇡

    # ===============================================================
    # 1) Dataset para PRÉ‑TREINAMENTO ― **somente features** (sem label)
    # ===============================================================
    df_pretrain = df.drop(columns=[label_column]).copy()

    train_idx, val_idx = train_test_split(
        np.arange(len(df)),                 # índices das linhas
        test_size=0.10,
        random_state=GLOBAL_SEED,
        stratify=df[label_column]           # garante a mesma proporção de classes
    )

    df_train_orig = df_pretrain.iloc[train_idx].reset_index(drop=True)
    df_val_orig   = df_pretrain.iloc[val_idx].reset_index(drop=True)

    # ===============================================================
    # 2) Modelo de pré‑treino (mesmo que antes)
    # ===============================================================
    embedder = TabularEmbedder(
        df=df_pretrain,
        categorical_columns=categorical_columns,
        numerical_columns=numerical_columns,
        dimensao=64,
        hidden_dim=128
    )

    transformer = TabularTransformerEncoder(
        d_model=embedder.dimensao,
        nhead=8,
        num_layers=8,
        dim_feedforward=64,
        dropout=0.3
    )

    pretrain_model = TabularBertPretrainer(embedder, transformer).to(device)

    def _init(m):
        if isinstance(m, (nn.Embedding, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)

    pretrain_model.apply(_init)

    optimizer = optim.AdamW(
    pretrain_model.parameters(), lr=3e-4, weight_decay=1e-2)   # lr menor
    num_epochs = 50
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs)

    batch_size = 512

    train_losses_history = []
    val_losses_history = []

    # ===============================================================
    # 3) LOOP de PRÉ‑TREINAMENTO com MÁSCARAS DINÂMICAS             #
    # ===============================================================
    print("\n=== Iniciando Pré‑Treinamento (máscaras novas a cada época) ===")
    for epoch in range(num_epochs):
        # ---------- (re)gera novas máscaras para TODA a base ----------
        df_train_masked = preprocess_table(                 # <<< MODIFICADO
            df_train_orig.copy(), p_base=0.5, fine_tunning=False)
        df_val_masked = preprocess_table(                   # <<< MODIFICADO
            df_val_orig.copy(), p_base=0.5, fine_tunning=False)

        df_train_size = len(df_train_masked)                # <<< MODIFICADO
        df_val_size = len(df_val_masked)                    # <<< MODIFICADO

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
            torch.nn.utils.clip_grad_norm_(pretrain_model.parameters(), 0.5)
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

    # ---- plot das curvas de perda (idem) ----
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses_history, label='Train Loss')
    plt.plot(val_losses_history, label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Pré‑Treino MLM')
    plt.legend(); plt.show()

    # =================================================
    # 4) FINE TUNING (Classificação) - EXEMPLO
    # =================================================
    print("\n=== Iniciando Fine-Tuning (Exemplo de Classificação) ===")
    # Separamos novamente o dataset completo (df) em train/val/test com a label
    df_trainval, df_test = train_test_split(
    df,
    test_size=0.10,
    random_state=GLOBAL_SEED,
    stratify=df[label_column]
    )

    df_train, df_val = train_test_split(
        df_trainval,
        test_size=0.1111,                      # 0.1111 × 0.9 ≈ 0.10
        random_state=GLOBAL_SEED,
        stratify=df_trainval[label_column]
    )

    df_train = df_train.reset_index(drop=True)
    df_val   = df_val.reset_index(drop=True)
    df_test  = df_test.reset_index(drop=True)

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
        num_labels=2,
        class_weights=class_weights
    ).to(device)

    # Otimizador para fine-tuning
    optimizer_ft = optim.AdamW(classifier_model.parameters(), lr=1e-4,weight_decay=1e-2)
    num_epochs_ft = 50
    batch_size_ft = 256
    scheduler_ft = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_ft, T_max=num_epochs_ft)
    train_losses_history = []
    val_losses_history = []

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
            torch.nn.utils.clip_grad_norm_(classifier_model.parameters(), max_norm=0.5)
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

    # Avaliação final no Test
    classifier_model.eval()
    with torch.no_grad():
        logits_test, test_loss = classifier_model(processed_test, labels=y_test)
        preds_test = torch.argmax(logits_test, dim=1)

    preds_test_np = preds_test.cpu().numpy()
    y_test_np = y_test.cpu().numpy()

    f1_test_per_class = f1_score(y_test_np, preds_test_np, average=None)
    f1_test_micro = f1_score(y_test_np, preds_test_np, average='micro')
    f1_test_macro = f1_score(y_test_np, preds_test_np, average='macro')

    print("\n=== Resultados no Conjunto de Teste ===")
    print(f"F1 micro: {f1_test_micro:.4f}")
    print(f"F1 macro: {f1_test_macro:.4f}")
    print(f"F1 por classe: {f1_test_per_class}")

    # Plot final (Fine-Tuning)
    plt.figure(figsize=(8,6))
    plt.plot(train_losses_history, label='Train Loss')
    plt.plot(val_losses_history,   label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Curvas de Perda (Fine-Tuning)')
    plt.legend()
    plt.show()
