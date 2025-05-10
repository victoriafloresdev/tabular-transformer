import torch
import torch.nn as nn

def initialize_weight(x):
    """
    Inicializa o peso de uma camada usando Xavier uniform para pesos e zeros para bias
    """
    nn.init.xavier_uniform_(x.weight)
    if x.bias is not None:
        nn.init.constant_(x.bias, 0)


class FeedForwardNetwork(nn.Module):
    """
    Rede feed-forward simples com duas camadas lineares e ativação ReLU
    """
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
    """
    Camada de atenção multi-cabeça para transformers
    """
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
    """
    Camada do encoder do transformer com self-attention e feed-forward
    """
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
    """
    Encoder completo do transformer para dados tabulares
    """
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