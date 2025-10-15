import torch
import torch.nn as nn

class LayerNorm(nn.Module):
	def __init__(self, eps, emb):
		super().__init__()
		self.eps = eps
		self.scale = nn.Parameter(torch.ones(emb))
		self.shift = nn.Parameter(torch.zeros(emb))
	def forward(self,x):
		# mean of 0, std of 1
		x_mean = x.mean(dim=-1, keepdim=True)
		x_var = x.var(dim=-1, keepdim=True, unbiased=False)
		x_norm = (x-x_mean) / torch.sqrt(x_var+self.eps)
		return x_norm*self.scale + self.shift
	
	def layer_norm(x: torch.Tensor, gamma:torch.Tensor, beta:torch.Tensor, 
									eps:float=1e-15):
		# x: [B, seq, emb]
		x_mean = x.mean(dim=-1, keepdim=True)
		x_var = x.var(dim=-1, keepdim=True, unbiased=False)
		x_norm = (x-x_mean) / torch.sqrt(x_var+eps)
		
		# reshape gamma, beta for broadcasting
		# gamma [emb, ], beta [emb, ] --> [B, seq, emb]
		shape = [1] * x.dim() # 입력의 차원 수만큼 1로 채움
		shape[-1] = x.size(-1) # [B, seq, emb]
		gamma = gamma.view(*shape)
		beta = beta.view(*shape)
		return x_norm * gamma + beta
	
class BatchNorm2D:
	def __init__(self, num_features, momentum=0.9, eps=1e-5):
		self.gamma = torch.ones(num_features, 1, 1)
		self.beta = torch.zeros(num_features, 1, 1)
		self.running_mean = torch.zeros(num_features, 1, 1)
		self.running_var = torch.ones(num_features, 1, 1)
		self.momentum = momentum
		self.eps=eps
	def __call__(self, x, training=True):
		# [B, C, H, W] -> N, H, W로 계산 (channel-wise)
		if training:
			mean = x.mean(dim=(0,2,3), keepdim=True)
			var = x.var(dim=(0,2,3), keepdim=True, unbiased=False)
			self.running_mean = self.momentum * self.running_mean + (1-self.momentum)*mean
			self.running_var = self.momenum * self.running_mean + (1-self.momentum)*var
		else:
			mean = self.running_mean
			var = self.running_var
		
		x_norm = (x-mean)/torch.sqrt(var + self.eps)
		return x_norm*self.gamma + self.beta
			