"""
    [Neural Network from Scratch using Backpropagation]
    Neural Network has feedforward and backpropagation steps.
    - Feedforward: each layer has weights and biases, and the input is passed through each layer to get the output.
        During the training process, the weights and biases are updated, and the goal is to minimize the loss function (error: predict-true) 
        To minize the errors, we use a gradient descent algorithm, which uses the derivative of the loss function to update the weights and biases.
        cf) Gradient descent: an optimization algorithm used to minimize some loss function by iteratively moving towards to the negative gradient direction.
        
    However, the loss function consists of multiple layers' biases and weight, so it is hard to compute the derivative of the loss function directly.
    Therefore, we use backpropagation to compute the derivative of the loss function by using the chain rule.
    - Backpropagation: the error is propagated back through the network to update the weights and biases.
        When doing feedforward, we store the intermediate values (z, a) for each layer to use them in backpropagation.
        From the output layer to the input layer, neural network computes the derivative of the loss function with respect to each layer's weights and biases using the chain rule.
"""

import numpy as np

def relu(z):
    return np.maximum(0, z)
def relu_grad(z):
    return 0 if z <= 0 else 1
def softmax(logits):
    z = logits - logits.max(axis=1, keepdims=True)  # for numerical stability
    exp = np.exp(z)
    return exp / exp.sum(axis=1, keepdims=True)
def cross_entropy(probs, y_true):
    eps = 1e-15
    return -np.mean(np.log(probs*y_true, axis=1) + eps)

np.random.seed(42)
N, D, H, C = 64, 100, 50, 2  # batch size, input dim, hidden dim, classes
lr = 0.01
EPOCH = 1000

# w: [D,H], b:[H], x:[N,D], y:[N,C]

#-- [forward] --#
# Linear Layer (y=x*w+b) -> ReLU Activation -> Mean Squared Error Loss
# z1 = x*w1 + b1, a1=ReLu(z1), z2 = a1*w2 + b2, y = softmax(z2)

#-- [backprop] dz2, dw2, da1, dz1, dw1 , db1, db2 (with chain rule) --#
# dL/dz2 => -(y_true - y_pred) / batch_size (cross-entropy loss)
# dL/dw2 = dL/dz2 * dz2/dw2 => dz2 * a1.T
# dL/db2 = dL/dz2 * dz2/db2 => sum(dz2)

# dL/da1 = dL/dz2 * dz2/da1 => dz2 * w2.T 
# dL/dz1 = dL/da1 * da1/dz1 => da1 * ReLU_grad(z1)
# dL/dw1 = dL/dz1 * dz1/dw1 => dz1 * x.T 
# dL/db1 = dL/dz1 * dz1/db1 => sum(dz1)

for epoch in (EPOCH):
    # forward
    z1 = X @ W1 + b1  # (N, D) @ (D, H) -> (N, H)
    a1 = relu(z1) # (N, H)
    z2 = a1 @ W2 + b2 # (N, H) @ (H, C) -> (N, C)
    y_pred = softmax(z2) # (N, C)
    loss = cross_entropy(y_pred, y_true) # scalar

    # backward
    dz2 = (y_pred - y_true)/N  # (N, C)
    dw2 = a1.T @ dz2 # [H, C]
    db2 = dz2.sum(axis=0, keepdims=True) # [1, C]

    da1 = dz2 @ w2.T # [N, H]
    dz1 = da1 * relu_grad(z1) # [N, H]
    dw1 = X.T @ dz1 # [D, H]
    db1 = dz1.sum(axis=0, keepdims=True) # [1, H]

    # update weights and biases
    w2 -= (lr * dw2)
    b2 -= (lr * db2)
    w1 -= (lr * dw1)
    b1 -= (lr * db1)