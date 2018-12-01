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


def affine_relu_forward(x, w, b):
    out, cache_a = affine_forward(x, w, b)
    out, cache_r = relu_forward(out)
    return out, (cache_a, cache_r)


def affine_relu_backward(dout, cache):
    cache_a, cache_r = cache
    dout = relu_backward(dout, cache_r)
    dx, dw, db = affine_backward(dout, cache_a)
    return dx, dw, db


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We drop each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None

    if mode == 'train':
        mask = (np.random.rand(*x.shape) < (1 - p)) / (1 - p)
        out = x * mask
    elif mode == 'test':
        out = x

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    if mode == 'train':
        dx = dout * mask
    elif mode == 'test':
        dx = dout
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    N, D = x.shape

    # get parameters
    mode = bn_param['mode']                   # mode is train or test
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var  = bn_param.get('running_var',  np.zeros(D, dtype=x.dtype))
    cache = None

    if mode == 'train':
        
        # Normalize
        mu = np.mean(x, axis=0)
        xc = x - mu
        var = np.mean(xc ** 2, axis=0)
        std = (var + eps) ** 0.5
        xn = xc / std
        
        # Scale and Shift
        out = gamma * xn + beta

        cache = (x, xc, var, std, xn, gamma, eps)

        # update running mean and running average
        running_mean = momentum * running_mean + (1 - momentum) * mu
        running_var  = momentum * running_var  + (1 - momentum) * var
        
        bn_param['running_mean'] = running_mean
        bn_param['running_var' ] = running_var
    
    else:
        xn = (x - running_mean) / (np.sqrt(running_var + eps))
        out = gamma * xn + beta
        
    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None

    x, xc, var, std, xn, gamma, eps = cache
    N = x.shape[0]

    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * xn, axis=0)
    dxn = dout * gamma

    dxc = dxn / std
    dstd = np.sum(-(xc * dxn) / (std * std), axis=0)
    dvar = 0.5 * dstd / std

    dxc += (2.0 / N) * xc * dvar
    dmu = -np.sum(dxc, axis=0)
    dx = dxc + dmu / N

    return dx, dgamma, dbeta


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
