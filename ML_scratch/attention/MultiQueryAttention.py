import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiQueryAttention(nn.Module):
	def __init__(self, d_in, d_out, context_length, num_head, droprate, bias, kv_dim):
		super().__init__()
		assert d_out % num_head == 0, "d_out should be divisible by num_head"
		self.d_out = d_out
		self.head_dim = d_out // num_head
		self.num_head = num_head
		self.kv_dim = kv_dim if kv_dim else self.head_dim
		
		self.Q = nn.Linear(d_in, d_out, bias=bias) #[B, T, H*D]
		self.K = nn.Linear(d_in, self.kv_dim, bias=bias) # [B, T, D_kv] (H=1)
		self.V = nn.Linear(d_in, self.kv_dim, bias=bias) # [B, T, D_kv]
		
		if self.head_dim != self.kv_dim:
			self.q_to_kv = nn.Linear(self.head_dim, self.kv_dim, bias=False)
			self.kv_to_head = nn.Linear(self.kv_dim, self.head_dim, bias=True)
		else:
			self.q_to_kv = nn.Identity()
			self.kv_to_head = nn.Identity()
		
		self.dropout = nn.Dropout(droprate)
		self.out_proj = nn.Linear(d_out, d_out)

def forward(self, x):
	B, T, _ = x.shape
	device = x.device
	
	# get weights of QKV
	# [B, T, D_out] -> [B, T, H, D] -> [B, H, T, D]
	q = self.Q(x).view(B, T, self.num_head, self.head_dim).transpose(1,2)
	q = self.q_to_kv(q) # [B, H, T, D_kv]
	
	# add dimension in 1st position
	k = self.K(x).unsqueeze(1) # [B, 1, T, D_kv]
	v = self.V(x).unsqueeze(1)
	
	# broadcast KV to all heads without parameter
	k = k.expand(B, self.num_head, T, self.kv_dim) # [B, H, T, D_kv]
	v = v.expand(B, self.num_head, T, self.kv_dim)
	
	# get attention values Q @ K.T
	attn_values = q @ k.transpose(2,3) # [B, H, T, T]
	
	# attn weight
	attn_weight = F.softmax(attn_values/self.kv_dim **0.5, dim=-1)
	attn_weight = self.droupout(attn_weight)
	
	# context vector [B, H, T, D_kv]
	ctx_vec = attn_weight @ v
	
	# if kv_dim != head_dim -> match output dim (projection)
	ctx_vec = self.kv_to_head(ctx_vec)
	ctx_vec = ctx_vec.transpose(1,2).contiguous().view(B, -1, self.num_head * self.head_dim)
	
	return self.out_proj(ctx_vec) # [B, T, D_out]