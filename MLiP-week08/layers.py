import numpy as np


def affine_forward(x, W, b):
    """
    A linear mapping from inputs to scores.
    
    Inputs:
        - x: input matrix (N, d_1, ..., d_k)
        - W: weigh matrix (D, C)
        - b: bias vector  (C, )
    
    Outputs:
        - out: output of linear layer (N, C)
    """
    x2d = np.reshape(x, (x.shape[0], -1))  # convert 4D input matrix to 2D    
    out = np.dot(x2d, W) + b               # linear transformation
    cache = (x, W, b)                      # keep for backward step (stay with us)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
        - dout: Upstream derivative, of shape (N, C)
        - cache: Tuple of:
            - x: Input data, of shape (N, d_1, ... d_k)
            - w: Weights, of shape (D, C)
            - b: biases, of shape (C,)

    Outputs:
        - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
        - dw: Gradient with respect to w, of shape (D, C)
        - db: Gradient with respect to b, of shape (C,)
    """
    x, w, b = cache
    x2d = np.reshape(x, (x.shape[0], -1))

    # compute gradients
    db = np.sum(dout, axis=0)
    dw = np.dot(x2d.T, dout)
    dx = np.dot(dout, w.T)

    # reshape dx to match the size of x
    dx = dx.reshape(x.shape)
    
    return dx, dw, db

def relu_forward(x):
    """Forward pass for a layer of rectified linear units.

    Inputs:
        - x: a numpy array of any shape

    Outputs:
        - out: output of relu, same shape as x
        - cache: x
    """
    cache = x
    out = np.maximum(0, x)
    return out, cache

def relu_backward(dout, cache):
    """Backward pass for a layer of rectified linear units.

    Inputs:
        - dout: upstream derevatives, of any shape
        - cache: x, same shape as dout

    Outputs:
        - dx: gradient of loss w.r.t x
    """
    x = cache
    dx = dout * (x > 0)
    return dx

def svm_loss(scores, y):
    """
    Fully-vectorized implementation of SVM loss function.

    Inputs:
        - scores: scores for all training data (N, C)
        - y: correct labels for the training data of shape (N,)

    Outputs:
       - loss: data loss plus L2 regularization loss
       - grads: graidents of loss w.r.t scores
    """

    N = scores.shape[0]

    # Compute svm data loss
    correct_class_scores = scores[range(N), y]
    margins = np.maximum(0.0, scores - correct_class_scores[:, None] + 1.0)
    margins[range(N), y] = 0.0
    loss = np.sum(margins) / N

    # Compute gradient off loss function w.r.t. scores
    num_pos = np.sum(margins > 0, axis=1)
    dscores = np.zeros(scores.shape)
    dscores[margins > 0] = 1
    dscores[range(N), y] -= num_pos
    dscores /= N

    return loss, dscores


def softmax_loss(scores, y):
    """
    Softmax loss function, fully vectorized implementation.

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
        - scores: A numpy array of shape (N, C).
        - y: A numpy array of shape (N,) containing training labels;

    Outputs:
        - loss as single float
        - gradient with respect to scores
    """
    N = scores.shape[0]  # number of input data

    # compute data loss
    shifted_logits = scores - np.max(scores, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    loss = -np.sum(log_probs[range(N), y]) / N

    # Compute gradient of loss function w.r.t. scores
    dscores = probs.copy()
    dscores[range(N), y] -= 1
    dscores /= N
    
    return loss, dscores        
