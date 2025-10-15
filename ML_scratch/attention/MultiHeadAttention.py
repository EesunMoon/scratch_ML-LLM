import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
	def __init__(self, d_in, d_out, context_length, num_head, droprate, bias):
		super().__init__()
		assert d_out % num_head == 0, "d_out should be divisible by num_head"
		self.d_out = d_out
		self.head_dim = d_out // num_head
		self.num_head = num_head
		
		self.Q = nn.Linear(d_in, d_out, bias=bias)
		self.K = nn.Linear(d_in, d_out, bias=bias)
		self.V = nn.Linear(d_in, d_out, bias=bias)
		self.dropout = nn.Dropout(droprate)
		self.out_proj = nn.Linear(d_out, d_out)
		self.register_buffer("mask", 
												torch.triu(torch.ones(context_length, context_length), diagonal=1))
def softmax_naive(self, attn):
	# [ batch, head_dim, num_tokens, num_tokens ]
	attn_max = attn_max.max(dim=-1, keepdim=True).values
	attn_shift = attn = attn_max
	exp = torch.exp(attn_shift)
	return exp/(exp.sum(dim=-1, keepdim=True)+1e-12)
	
def forward(self, x):
	batch, num_tokens, d_in = x.shape
	
	# get weights of QKV --> [batch, num_tokens, d_out]
	# d_out = num_head * head_dim  --> [batch, num_tokens, num_head, head_dim]
	queries = self.Q(x).view(batch, num_tokens, self.num_head, self.head_dim)
	keys = self.K(x).view(batch, num_tokens, self.num_head, self.head_dim)
	values = self.V(x).view(batch, num_tokens, self.num_head, self.head_dim)
	
	# transpose to calculate attn in batch and number_tokens-wise
	# [batch, num_head, num_tokens, head_dim]
	queries = queries.transpose(1,2)
	keys = keys.transpose(1,2)
	values = values.transpose(1,2)
	
	# get attention values Q @ K.T
	attn_values = queries @ keys.transpose(2,3) # [batch, num_head, num_tokens, num_tokens]
	
	# (if masked) masking
	mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
	attn_values.masked_fill_(mask_bool, -torch.inf)
	
	# attn weight
	attn_weight = F.softmax(attn_values/keys.shape[-1] **0.5, dim=-1)
	# attn_weight = self.softmax_naive(attn_weight)
	attn_weight = self.dropout(attn_weight)
	
	# context vector
	# [batch, num_head, num_tokens, head_dim] --> [batch, num_tokens, num_head, head_dim]
	context_vector = (attn_weight @ values).transpose(1,2)
	context_vector = context_vector.reshape(batch, num_tokens, self.d_out)
	context_vector = self.out_proj(context_vector)
	
	return context_vector # [batch, num_tokens, d_out]