import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from embedder import TabularEmbedder
from transformer import TabularTransformerEncoder

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

        # 1) Embeddings "corrompidos" (entrada do Transformer)
        emb_in = self.embedder(df_masked)              # (B, L, d)
        # 2) Embeddings "puros" (alvo) — detach p/ não retro‑propagar neles
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