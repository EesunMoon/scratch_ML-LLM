import numpy as np

def gini(y):
	# y: [N,]
	_, counts = np.unique(y, return_counts=True)
	p = counts/counts.sum()
	return 1.0 - np.sum(p**2)


def entropy(y):
	_, counts = np.unique(y, return_counts=True)
	p = counts/counts.sum()
	eps=1e-15
	return -np.sum(p* np.log2(p+eps))

def information_gain(y_parent, y_left, y_right, criterion='gini'):
	if criterion == 'gini':
		impurity = gini
	else:
		impurity = entropy
	
	n = len(y_parent)
	n_left = len(y_left)
	n_right = len(y_right)
	
	# sample 수 고려
	weighted_impurity = (n_left/n)*impurity(y_left) + (n_right/n)*impurity(y_right)
	IG = impurity(y_parent) - weighted_impurity
	return IG