import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5 # preventing divided by zero
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
        
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=True)
        norm_x = (x-mean)/torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4*cfg["emb_dim"]), # 4: hyper parameter
            nn.GELU(),
            nn.Linear(4*cfg["emb_dim"], cfg["emb_dim"]),
        )
    def forward(self, x):
        return self.layers(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, num_heads, dropout, bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=bias)
        self.W_key = nn.Linear(d_in, d_out, bias=bias)
        self.W_value = nn.Linear(d_in, d_out, bias=bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length,context_length), diagonal=1))

    def forward(self, x):
        batch, num_tokens, d_in = x.shape

        # [batch, num_tokens, d_in]
        # --> [batch, num_tokens, d_out] d_out = num_heads * head_dim
        # --> [batch, num_tokens, num_heads, head_dim]
        queries = self.W_query(x).view(batch, num_tokens, self.num_heads, self.head_dim)
        keys = self.W_key(x).view(batch, num_tokens, self.num_heads, self.head_dim)
        values = self.W_value(x).view(batch, num_tokens, self.num_heads, self.head_dim)

        # 1) compute attention weight: queires @ keys.T
        ## [batch, num_tokens, num_heads, head_dim]
        ## --> [batch, num_heads, num_tokens, head_dim]
        queries = queries.transpose(1,2)
        keys = keys.transpose(1,2)
        values = values.transpose(1,2)

        attn_values = queries @ keys.transpose(2,3) # [batch, num_heads, num_tokens, num_tokens]

        ## masking the weights
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_values.masked_fill_(mask_bool, -torch.inf) # [batch, num_heads, num_tokens, num_tokens]
        
        # 2) compute attention weight (normalization)
        attn_weights = F.softmax(attn_values / keys.shape[-1]**0.5, dim=-1) # [batch, num_heads, num_tokens, num_tokens]
        attn_weights = self.dropout(attn_weights)
        
        # 3) compute context_vector: attn_weights @ value
        ## [batch, num_heads, num_tokens, num_tokens] @ [batch, num_heads, num_tokens, head_dim]
        # --> [batch, num_heads, num_tokens, head_dim]
        context_vector = (attn_weights @ values).transpose(1,2)
        context_vector = context_vector.reshape(batch, num_tokens, self.d_out)
        context_vector = self.out_proj(context_vector)

        return context_vector # [batch, num_tokens, d_out]
        
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in = cfg["emb_dim"], 
            d_out = cfg["emb_dim"],
            context_length = cfg["context_length"], 
            num_heads = cfg["n_heads"], 
            dropout = cfg["drop_rate"],
            bias = cfg["qkv_bias"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x # saving the shortcut connection
        x = self.norm1(x)
        x = self.att(x) # [batch, num_tokens, emb_dim]
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x