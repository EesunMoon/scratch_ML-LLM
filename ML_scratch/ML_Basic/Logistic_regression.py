import torch

class LogisticRegression:
	def __init__(self, lr, iters, l2 = 0.0):
		self.lr = lr
		self.iters = iters
		self.weights = None
		self.bias = None
		self.l2 = l2 # regularizatiom
	
	@staticmethod
	def sigmoid(z):
		return 1.0/(1.0+torch.exp(-z))
	
	@staticmethod
	def BCE(y, p, eps=1e-9):
		p = torch.clamp(p, eps, 1-eps)
		diff = (y*torch.log(p) + (1-y)*torch.log(1-p))
		return -diff.mean()
	
	def fit(self, x, y):
		# x: [N, D], y:[N], weight: [D,], bias: [1,],
		X = x.float()
		y = y.float().view(-1,1) # [N,1]
		
		N, D = X.shape
		self.weights = torch.zeros(D, 1, dtype=X.dype, device=X.device)
		self.bias = torch.zeros(1, dtype=X.dtype, device=X.device)
		
		for _ in range(self.iters):
			# forward
			logits = X @ self.weights + self.bias
			p = self.sigmoid(logits)
			loss = self.BCE(y, p)
			
			# backward
			diff = (p-y)
			dW = (1.0/N)*(X.T @ diff)
			db = (1.0/N)*diff.sum()
			
			# regularization
			if self.l2 > 0.0:
				dW += (2.0)* self.l2 * self.weight
			
			# update
			self.weights -= self.lr * dW
			self.bias -= self.lr * db
	
	def predict(self, x, threshold=0.5):
		X = x.float()
		logits = (X @ self.weights + self.bias).view(-1) # [N]
		proba = self.sigmoid(logits)
		return (proba >= threshold).long()
	