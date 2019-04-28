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

def svm_loss_naive(scores, y, W, reg=1e-3):
    """
    Naive implementation of SVM loss function.

    Inputs:
        - scores: scores for all training data (N, C)
        - y: correct labels for the training data
        - reg: regularization strength (lambd)

    Outputs:
       - loss: data loss plus L2 regularization loss
       - grads: graidents of loss wrt scores
    """

    N, C = scores.shape

    # Compute svm data loss
    loss = 0.0
    for i in range(N):
        s = scores[i]  # scores for the ith data
        correct_class = y[i]  # correct class score

        for j in range(C):
            if j == y[i]:
                continue
            else:
                # loss += max(0, s[j] - s[correct_class] + 1.0)
                margin = s[j] - s[correct_class] + 1.0
                if margin > 0:
                    loss += margin
    loss /= N

    # Adding L2-regularization loss
    loss += 0.5 * reg * np.sum(W * W)

    # Compute gradient off loss function w.r.t. scores
    # We will write this part later
    grads = {} 

    return loss, grads

def svm_loss_half_vectorized(scores, y, W, reg=1e-3):
    """
    Half-vectorized implementation of SVM loss function.

    Inputs:
        - scores: scores for all training data (N, C)
        - y: correct labels for the training data
        - reg: regularization strength (lambd)

    Outputs:
       - loss: data loss plus L2 regularization loss
       - grads: graidents of loss wrt scores
    """

    N, C = scores.shape

    # Compute svm data loss
    loss = 0.0
    for i in range(N):
        s = scores[i]  # scores for the ith data
        correct_class = y[i]  # correct class score

        margins = np.maximum(0.0, s - s[correct_class] + 1.0)
        margins[correct_class] = 0.0
        loss += np.sum(margins)

    loss /= N

    # Adding L2-regularization loss
    loss += 0.5 * reg * np.sum(W * W)

    # Compute gradient off loss function w.r.t. scores
    # We will write this part later
    grads = {} 

    return loss, grads


def svm_loss(scores, y, W, reg=1e-3):
    """
    Fully-vectorized implementation of SVM loss function.

    Inputs:
        - scores: scores for all training data (N, C)
        - y: correct labels for the training data
        - reg: regularization strength (lambd)

    Outputs:
       - loss: data loss plus L2 regularization loss
       - grads: graidents of loss wrt scores
    """

    N = scores.shape[0]

    # Compute svm data loss
    correct_class_scores = scores[range(N), y]
    margins = np.maximum(0.0, scores - correct_class_scores[:, None] + 1.0)
    margins[range(N), y] = 0.0
    loss = np.sum(margins) / N

    # Adding L2-regularization loss
    loss += 0.5 * reg * np.sum(W * W)

    # Compute gradient off loss function w.r.t. scores
    # We will write this part later
    grads = {} 

    return loss, grads


def softmax_loss_naive(scores, y, W, reg=1e-3):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
        - scores: A numpy array of shape (N, C).
        - y: A numpy array of shape (N,) containing training labels;
        - W: A numpy array of shape (D, C) containing weights.
        - reg: (float) regularization strength

    Outputs:
        - loss as single float
        - gradient with respect to weights W; an array of same shape as W
    """
    N, C = scores.shape

    # compute data loss
    loss = 0.0
    for i in range(N):
        correct_class = y[i]
        score = scores[i]
        score -= np.max(scores)
        exp_score = np.exp(score)
        probs = exp_score / np.sum(exp_score)
        loss += -np.log(probs[correct_class])

    loss /= N

    # compute regularization loss
    loss += 0.5 * reg * np.sum(W * W)

    # Compute gradient off loss function w.r.t. scores
    # We will write this part later
    grads = {}  

    return loss, grads


def softmax_loss(scores, y, W, reg=1e-3):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
        - scores: A numpy array of shape (N, C).
        - y: A numpy array of shape (N,) containing training labels;
        - W: A numpy array of shape (D, C) containing weights.
        - reg: (float) regularization strength

    Outputs:
        - loss as single float
        - gradient with respect to weights W; an array of same shape as W
    """
    N = scores.shape[0]  # number of input data

    # compute data loss
    scores -= np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    loss = -np.sum(np.log(probs[range(N), y])) / N

    # compute regularization loss
    loss += 0.5 * reg * np.sum(W * W)

    # Compute gradient off loss function w.r.t. scores
    # We will write this part later
    grads = {}  

    return loss, grads        
