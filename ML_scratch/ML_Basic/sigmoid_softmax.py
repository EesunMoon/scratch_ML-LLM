import torch 

# sigmoid = 1/(1+exp(-x))
def sigmoid(x):
	return 1/(1+torch.exp(-x))

# softmax = exp(x) / sum(exp(x))
def softmax_lastdim(x: torch.Tensor) -> torch.Tensor:
	# x: [..., D]
	x_max = x.max(dim=-1, keepdim=True).values
	x_shift = x-x_max
	exp = torch.exp(x_shift)
	demonitor = exp.mean(dim=-1, keepdim=True) + 1e-12
	return exp/demonitor	