import numpy as np
from models.layer_utils import affine_relu_affine_forward, affine_relu_affine_backward
from models.layers import affine_forward, affine_backward


class Layer:
    '''API for layer classes. They need a self.params variable for their parameters.'''

    def forward(*args, **kwargs):
        '''Returns the output and a cache variable to be used in the backwards pass.'''
        raise NotImplementedError()

    def backward(*args, **kwargs):
        '''Returns dx, grads, where grads are the gradients for this layer's parameters.'''
        raise NotImplementedError()


class AffineReluAffine:

    def __init__(self, inputDim, outputDim, weightScale=None):
        weightScale = weightScale or np.sqrt(2.0 / inputDim)

        self.params = {}
        self.params['W1'] = np.random.normal(scale=weightScale, size=(inputDim, outputDim))
        self.params['b1'] = np.zeros(outputDim)
        self.params['W2'] = np.random.normal(scale=weightScale, size=(outputDim, outputDim))
        self.params['b2'] = np.zeros(outputDim)
        self.isRegularised = dict(W1=True, W2=True, b1=False, b2=False)

    def forward(self, x):
        # Clear cache and gradients.
        self.cache, self.grads = None, None
        W1 = self.params['W1']
        b1 = self.params['b1']
        W2 = self.params['W2']
        b2 = self.params['b2']
        out, cache = affine_relu_affine_forward(x, W1, b1, W2, b2)
        self.cache = cache
        return out

    def backward(self, dout):
        assert self.cache is not None
        dx, dW1, db1, dW2, db2 = affine_relu_affine_backward(dout, self.cache)
        self.grads = dict(W1=dW1, b1=db1, W2=dW2, b2=db2)
        return dx


class Affine:

    def __init__(self, inputDim, outputDim, weightScale=None):
        weightScale = weightScale or np.sqrt(2.0 / inputDim)
        self.params = {}
        self.params['W'] = np.random.normal(scale=weightScale, size=(inputDim, outputDim))
        self.params['b'] = np.zeros(outputDim)
        self.isRegularised = dict(W=True, b=False)

    def forward(self, x):
        # Clear cache and grads.
        self.cache, self.grads = None, None
        W = self.params['W']
        b = self.params['b']
        out, cache = affine_forward(x, W, b)
        self.cache = cache
        return out

    def backward(self, dout):
        assert self.cache is not None
        dx, dW, db = affine_backward(dout, self.cache)
        self.grads = dict(W=dW, b=db)
        return dx
