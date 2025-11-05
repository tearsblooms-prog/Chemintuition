
import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class ConditionalTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.GELU()

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # --- Causal Self-Attention ---
        tgt2 = self.norm1(tgt)
        q = k = v = tgt2
        tgt2, _ = self.self_attn(q, k, v, attn_mask=tgt_mask, need_weights=False)
        tgt = tgt + self.dropout1(tgt2)

        # --- Cross-Attention ---
        tgt2 = self.norm2(tgt)
        q = tgt2
        k = v = memory  # Memory is the condition from MPNN
        tgt2, _ = self.multihead_attn(q, k, v, attn_mask=memory_mask, need_weights=False)
        tgt = tgt + self.dropout2(tgt2)

        # --- Feed Forward ---
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)

        return tgt


class AutoregressiveModel(nn.Module):
    def __init__(self, vocab_size, model_dim=256, num_layers=6, num_heads=8,
                 condition_dim=1024, dropout=0.1, max_seq_len=150):
        super().__init__()
        self.model_dim = model_dim

        self.token_embedding = nn.Embedding(vocab_size, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim, dropout, max_len=max_seq_len)

        # Project condition vector to model dimension
        self.condition_projection = nn.Linear(condition_dim, model_dim)

        decoder_layer = ConditionalTransformerDecoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout
        )
        self.transformer_decoder = nn.ModuleList([
            ConditionalTransformerDecoderLayer(
                d_model=model_dim, nhead=num_heads, dim_feedforward=model_dim * 4, dropout=dropout
            ) for _ in range(num_layers)
        ])

        self.output_norm = nn.LayerNorm(model_dim)
        self.to_logits = nn.Linear(model_dim, vocab_size)

        self._init_weights()

    def _init_weights(self):
        # Initialize weights for better training stability
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.zeros_(self.to_logits.bias)

    def _generate_square_subsequent_mask(self, sz, device):
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, token_ids, condition_vector):
        device = token_ids.device
        seq_len = token_ids.size(1)

        # 1. Embed tokens and add positional encoding
        token_embeds = self.token_embedding(token_ids) * math.sqrt(self.model_dim)
        token_embeds = self.pos_encoder(token_embeds)

        # 2. Project condition and expand for cross-attention
        # The condition is the same for every token in a sequence
        condition_embeds = self.condition_projection(condition_vector).unsqueeze(1)

        # 3. Create causal mask for self-attention
        tgt_mask = self._generate_square_subsequent_mask(seq_len, device)

        # 4. Pass through Transformer decoder layers
        output = token_embeds
        for layer in self.transformer_decoder:
            output = layer(output, condition_embeds, tgt_mask=tgt_mask)

        # 5. Final normalization and projection to vocab size
        output = self.output_norm(output)
        logits = self.to_logits(output)

        return logits