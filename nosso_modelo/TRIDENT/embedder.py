import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder

from utils import split_numeric_and_special

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