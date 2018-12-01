import numpy as np
from layers import *


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learn-able parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy data-type object; all computations will be performed using
          this data-type. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deterministic so we can gradient check the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        dims = [input_dim] + hidden_dims + [num_classes]
        for i in range(1, self.num_layers + 1):
            self.params['W%d' %i] = weight_scale * np.random.randn(dims[i - 1], dims[i])
            self.params['b%d' %i] = np.zeros(dims[i])
            if i < self.num_layers and self.use_batchnorm:
                self.params['gamma%d' %i] = np.ones(dims[i])
                self.params['beta%d' %i] = np.zeros(dims[i])

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct data-type
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        scores = None

        cache = {}
        a_cache, relu_cache, bn_cache, d_cache = {}, {}, {}, {}
        h = X
        for i in range(1, self.num_layers + 1):
            W, b = self.params['W%d' % i], self.params['b%d' % i]
            if i < self.num_layers:
                if self.use_batchnorm:
                    gamma, beta = self.params['gamma%d' % i], self.params['beta%d' % i]
                    h, a_cache[i] = affine_forward(h, W, b)
                    h, bn_cache[i] = batchnorm_forward(h, gamma, beta, self.bn_params[i - 1])
                    h, relu_cache[i] = relu_forward(h)
                else:
                    h, cache[i] = affine_relu_forward(h, W, b)
                if self.use_dropout:
                    h, d_cache[i] = dropout_forward(h, self.dropout_param)
            else:
                scores, cache[i] = affine_forward(h, W, b)

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}

        loss, dscores = softmax_loss(scores, y)

        # backward pass
        dout = dscores
        for i in reversed(range(1, self.num_layers + 1)):
            if i < self.num_layers:
                if self.use_dropout:
                    dout = dropout_backward(dout, d_cache[i])
                if self.use_batchnorm:
                    dout = relu_backward(dout, relu_cache[i])
                    dout, grads['gamma%d' % i], grads['beta%d' % i] = batchnorm_backward(dout, bn_cache[i])
                    dout, grads['W%d' % i], grads['b%d' % i] = affine_backward(dout, a_cache[i])
                else:
                    dout, grads['W%d' % i], grads['b%d' % i] = affine_relu_backward(dout, cache[i])
            else:
                dout, grads['W%d' % i], grads['b%d' %i] = affine_backward(dout, cache[i])

        for i in range(1, self.num_layers):
            W = self.params['W%d' % i]
            loss += 0.5 * self.reg * np.sum(W * W)
            grads['W%d' % i] += self.reg * W

        return loss, grads
