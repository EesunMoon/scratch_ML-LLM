import torch

class LinearRegression(nn.Module):
	def __init__(self, lr, iters, reg_type, reg_lambda=0.0):
		super().__init__()
		self.lr = lr
		self.iters = iters
		self.weights = None
		self.bias = None
		
		self.reg_type = reg_type # 'L1', or 'L2'
		self.reg_lambda = reg_lambda
	
	def fit(self, x, y):
		# X: [N, D], y: [N,]
		N, D = x.shape
		x = x.float()
		y = y.float().view(-1,1) # [N,1]
		self.weights = torch.zeros(D,1) # [D,1]
		self.bias = torch.zeros(1) # scalar
		
		for _ in range(self.iters):
			y_pred = self.predict(x) # [N,]
			# MSE
			diff = y_pred - y
			loss = (diff.pow(2).mean()).item()
			
			# backward
			dW = (2.0/N) * (x.T @ diff) # [D,]
			db = (2.0/N) * diff.sum()
			
			# update
			if self.reg_type == "l2":
				dw += 2 * self.reg_lambda * self.weights
			elif self.reg_type == "l1":
				dw += self.reg_lambda * torch.sign(self.weights)
				
			self.weights -= self.lr * dW
			self.bias -= self.lr * db
	def predict(self, x):
		return x @ self.weights + self.bias