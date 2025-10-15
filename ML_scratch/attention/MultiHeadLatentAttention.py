import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadLatentAttention(nn.Module):
	def __init__(self, d_in, d_out, context_length, num_head, droprate, bias,
							latent_dim, num_kv_head, kv_dim):
		super().__init__()
		assert d_out % num_head == 0, "d_out should be divisible by num_head"
		assert num_head % num_kv_head == 0
		
		self.d_out = d_out
		self.head_dim = d_out // num_head
		self.num_head = num_head
		
		self.num_kv_head = num_kv_head
		self.group_size = num_head // num_kv_head
		self.kv_dim = kv_dim if kv_dim else self.head_dim
		
		self.latent_dim = latent_dim if latent_dim else (d_in//2)
		
		self.Q = nn.Linear(d_in, d_out, bias=bias)
		self.to_latent = nn.Sequential(nn.Linear(d_in, self.latent_dim, bias=bias),
																		nn.SiLU())
		self.K = nn.Linear(self.latent_dim, self.num_kv_head*self.kv_dim, bias=bias)
		self.V = nn.Linear(self.latent_dim, self.num_kv_head*self.kv_dim, bias=bias)
		
		self.dropout = nn.Dropout(droprate)
		self.out_proj = nn.Linear(d_out, d_out)
		
		if self.kv_dim != self.head_dim:
			self.q_to_kv = nn.Linear(self.head_dim, self.kv_dim, bias=False)
			self.kv_to_head = nn.Linear(self.kv_dim, self.head_dim, bias=True)
		else:
			self.q_to_kv = nn.Identity()
			self.kv_to_head = nn.Identity()
	
def forward(self, x):
	B, T, _ = x.shape
	
	# [B, T, D_out] --> [B, T, H, D] --> [B, H, T, D]
	q = self.Q(x).view(B, T, self.num_head, self.head_dim).transpose(1,2)
	q = self.q_to_kv(q) # [B, H, T, D_kv]
	
	# latent H
	H = self.to_latent(x) # [B, T, L]
	
	# KV from Latent: [B, T, H_kv*D_kv] --> [B, H_kv, T, D_kv]
	k = self.K(H).view(B, T, self.num_kv_head, self.kv_dim).transpose(1,2)
	v = self.V(H).view(B, T, self.num_kv_head, self.kv_dim).transpose(1,2)
	
	# group expend: [B, H, T, D_kv]
	if self.num_kv_head != self.num_head:
		k = k.repeat_interleave(self.group_size, dim=1)
		v = v.repeat_interleave(self.group_size, dim=1)
	
	# get attention values Q @ K.T 
	attn_values = q @ k.transpose(2,3) # [B, H, T, T]
	
	# attn weight
	attn_weight = F.softmax(attn_values/self.kv_dim **0.5, dim=-1)
	attn_weight = self.dropout(attn_weight)
	
	# context vector
	ctx_vec = attn_weight @ v # [B, H, T, D_kv]
	ctx_vec = self.kv_to_head(ctx_vec) # [B, H, T, D]
	ctx_vec = ctx_vec.transpose(1,2).contiguous().view(B, T, self.num_head*self.head_dim)
	
	return  # [B, T, D_out]