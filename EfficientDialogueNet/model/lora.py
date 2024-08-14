import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=32, lora_dropout=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = alpha / rank
        self.lora_dropout = nn.Dropout(p=lora_dropout)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x):
        x_lora = self.lora_dropout(x)
        return (F.linear(x_lora, self.lora_A) @ self.lora_B.T) * self.scaling

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=32, lora_dropout=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        
        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.lora = LoRALayer(in_features, out_features, rank, alpha, lora_dropout)
    
    def forward(self, x):
        return self.linear(x) + self.lora(x)

    def merge_weights(self):
        self.linear.weight.data += (self.lora.lora_B @ self.lora.lora_A) * self.lora.scaling
        self.lora = None

    def unmerge_weights(self):
        if self.lora is None:
            raise RuntimeError("LoRA weights have been merged and discarded. Cannot unmerge.")
        self.linear.weight.data -= (self.lora.lora_B @ self.lora.lora_A) * self.lora.scaling

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_linear = LoRALinear(d_model, d_model)
        self.k_linear = LoRALinear(d_model, d_model)
        self.v_linear = LoRALinear(d_model, d_model)
        self.out_linear = LoRALinear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)

        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        output = self.out_linear(context)
        return output

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = LoRALinear(d_model, d_ff)
        self.fc2 = LoRALinear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class LoRALanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = nn.Embedding(max_seq_length, d_model)
        
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.fc_out = LoRALinear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        seq_length = x.size(1)
        pos = torch.arange(0, seq_length).unsqueeze(0).repeat(x.size(0), 1).to(x.device)
        
        x = self.dropout(self.embedding(x) + self.positional_encoding(pos))
        
        for layer in self.layers:
            x = layer(x, mask)
        
        output = self.fc_out(x)
        return output

    def merge_weights(self):
        for module in self.modules():
            if isinstance(module, LoRALinear):
                module.merge_weights()

    def unmerge_weights(self):
        for module in self.modules():
            if isinstance(module, LoRALinear):
                module.unmerge_weights()

