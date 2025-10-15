import torch
# cross_entropy = -sum(y*log(p(y)))
# binary_cross_entropy = -sum( y*log(p(y)) + (1-y)*log(1-p(y)) )
def cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
	# logits: [N, C], target: [N,]
	x_max = logits.max(dim=-1, keepdim=True).values
	x_shift = logits-x_max
	
	inner = torch.exp(x_shift).sum(dim=-1, keepdim=True) + 1e-12
	logsum_exp = torch.log(inner)
	log_probs = x_shift - logsum_exp # [N, C]
	
	# loss = - (sum(log(p)) / N
	n = logits.size(0)
	loss = -log_probs[torch.arange(n), target].mean()
	return loss