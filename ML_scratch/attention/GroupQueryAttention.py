import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupQueryAttention(nn.Module):
	def __init__(self, d_in, d_out, context_length, num_head, droprate, bias,
								num_kv_head, kv_dim):
		super().__init__()
		assert d_out % num_head == 0, "d_out should be divisible by num_head"
		assert num_head % num_kv_head == 0
		
		self.d_out = d_out
		self.head_dim = d_out // num_head
		self.num_head = num_head
		
		self.num_kv_head = num_kv_head
		self.group_size = num_head // num_kv_head
		self.kv_dim = kv_dim if kv_dim else self.head_dim
		
		self.Q = nn.Linear(d_in, d_out, bias=bias)
		self.K = nn.Linear(d_in, self.num_kv_head*self.kv_dim, bias=bias)
		self.V = nn.Linear(d_in, self.num_kv_head*self.kv_dim, bias=bias)
		self.dropout = nn.Dropout(droprate)
		self.out_proj = nn.Linear(d_out, d_out)
		
		if self.head_dim != self.kv_dim:
			self.q_to_kv = nn.Linear(self.head_dim, self.kv_dim, bias=False)
			self.kv_to_head = nn.Linear(self.kv_dim, self.head_dim, bias=True)
		else:
			self.q_to_kv = nn.Identity()
			self.kv_to_head = nn.Identity()
			
	
def forward(self, x):
	B, T, _ = x.shape
	device = x.device
	
	# get weights of QKV --> [B, T, D_out] --> [B, H, T, D]
	q = self.Q(x).view(B, T, self.num_head, self.head_dim).transpose(1,2)
	q = self.q_to_kv(q) # [B, H, T, D_kv]
	
	# GroupKV: [B, T, H_kv * D_kv] --> [B, H_kv, T, D_kv]
	k = self.K(x).view(B, T, self.num_kv_head, self.kv_dim).transpose(1,2)
	v = self.V(x).view(B, T, self.num_kv_head, self.kv_dim).transpose(1,2)
	
	# expand KV to match H via group repeat [H_kv -> H]
	k = k.repeat_interleave(self.group_size, dim=1) # [B, H, T, D_kv]
	v = v.repeat_interleave(self.group_size, dim=1)
	
	# get attention values Q @ K.T --> [B, H, T, T]
	attn_values = q @ k.transpose(2,3) 
	
	# attn weight
	attn_weight = F.softmax(attn_values/self.kv_dim **0.5, dim=-1)
	attn_weight = self.dropout(attn_weight)
	
	# context vector
	ctx_vec = attn_weight @ v # [B, H, T, D_kv]
	ctx_vec = self.kv_to_head(ctx_vec) # [B, H, T, D]
	ctx_vec = ctx_vec.transpose(1,2).contiguous().view(B, T, self.num_head*self.head_dim)
	
	return self.out_proj(ctx_vec) # [B, T, D_out]