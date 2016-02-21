import numpy as np
from layer_utils import affine_relu_affine_forward, affine_relu_affine_backward


class Layer:
    '''API for layer classes. They need a self.params variable for their parameters.'''

    def forward(*args, **kwargs):
        raise NotImplementedError()

    def backward(*args, **kwargs):
        raise NotImplementedError()


class AffineReluAffine:

    def __init__(self, inputDim, outputDim, weightScale=None):
        weightScale = weightScale or np.sqrt(2.0 / inputDim)

        self.params = {}
        self.params['W1'] = np.random.normal(scale=weightScale, size=(inputDim, outputDim))
        self.params['b1'] = np.zeros(outputDim)
        self.params['W2'] = np.random.normal(scale=weightScale, size=(outputDim, outputDim))
        self.params['b2'] = np.zeros(outputDim)

    def forward(self, x):
        W1 = self.params['W1']
        b1 = self.params['b1']
        W2 = self.params['W2']
        b2 = self.params['b2']
        return affine_relu_affine_forward(x, W1, b1, W2, b2)

    def backward(self, dout, cache):
        return affine_relu_affine_backward(dout, cache)
