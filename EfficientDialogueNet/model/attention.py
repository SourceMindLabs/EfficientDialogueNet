import torch
import torch.nn as nn
import torch.nn.functional as F

class ShallowDeepAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # Shallow layer
        self.shallow_query = nn.Linear(d_model, d_model)
        self.shallow_key = nn.Linear(d_model, d_model)
        self.shallow_value = nn.Linear(d_model, d_model)
        
        # Deep layer
        self.deep_query = nn.Linear(d_model, d_model)
        self.deep_key = nn.Linear(d_model, d_model)
        self.deep_value = nn.Linear(d_model, d_model)
        
        self.fc_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        # Shallow attention
        shallow_Q = self.shallow_query(query)
        shallow_K = self.shallow_key(key)
        shallow_V = self.shallow_value(value)
        
        shallow_Q = shallow_Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        shallow_K = shallow_K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        shallow_V = shallow_V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
        shallow_energy = torch.matmul(shallow_Q, shallow_K.permute(0, 1, 3, 2)) / self.head_dim**0.5
        
        if mask is not None:
            shallow_energy = shallow_energy.masked_fill(mask == 0, float("-1e20"))
        
        shallow_attention = torch.softmax(shallow_energy, dim=-1)
        shallow_out = torch.matmul(self.dropout(shallow_attention), shallow_V)
        shallow_out = shallow_out.permute(0, 2, 1, 3).contiguous()
        shallow_out = shallow_out.view(batch_size, -1, self.d_model)
        
        # Deep attention
        deep_Q = self.deep_query(query)
        deep_K = self.deep_key(key)
        deep_V = self.deep_value(value)
        
        deep_Q = deep_Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        deep_K = deep_K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        deep_V = deep_V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
        deep_energy = torch.matmul(deep_Q, deep_K.permute(0, 1, 3, 2)) / self.head_dim**0.5
        
        if mask is not None:
            deep_energy = deep_energy.masked_fill(mask == 0, float("-1e20"))
        
        deep_attention = torch.softmax(deep_energy, dim=-1)
        deep_out = torch.matmul(self.dropout(deep_attention), deep_V)
        deep_out = deep_out.permute(0, 2, 1, 3).contiguous()
        deep_out = deep_out.view(batch_size, -1, self.d_model)
        
        # Combine shallow and deep outputs
        out = self.fc_out(shallow_out + deep_out)
        return out