import numpy as np
from models.backprop import (
    affine_forward, affine_backward,
    relu_backward, relu_forward,
    sigmoid_forward, sigmoid_backward,
    scale_shift_forward, scale_shift_backward
)


class Layer:
    '''API for layer classes.'''
    def __init__(self, *args, **kwargs):
        self.params = {}
        self.regularizedParams = []
        raise NotImplementedError()

    def forward(self, x):
        '''Returns the output of this layer and a caches variables to be used in the backwards pass.'''
        self.cache = tuple()
        raise NotImplementedError()

    def backward(self, dout):
        ''' Calculates the backward pass for the layer.
        Inputs:
        - dout: Upstream derivative.

        Returns a tuple of
        - dx: Gradients with respect to inputs.
        - grads: Gradients with respect to parameters. Dictionary {name: dw} where dw is the gradient
          with respect to self.params[name].
        '''
        raise NotImplementedError()

    def __str__(self):
        '''Return name and perhaps dimensions.'''
        raise NotImplementedError()


class AffineReluAffine(Layer):
    '''A composite layer consisting of an Affine-ReLU-Affine transform.'''

    def __init__(self, inputDim, outputDim, weightScale=None):
        weightScale = weightScale or np.sqrt(2.0 / inputDim)

        self.params = {}
        self.params['W1'] = np.random.normal(scale=weightScale, size=(inputDim, outputDim))
        self.params['b1'] = np.zeros(outputDim)
        self.params['W2'] = np.random.normal(scale=weightScale, size=(outputDim, outputDim))
        self.params['b2'] = np.zeros(outputDim)
        self.regularizedParams = ['W1', 'W2']

    def forward(self, x):
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        a, fc_cache1 = affine_forward(x, W1, b1)
        a2, relu_cache = relu_forward(a)
        out, fc_cache2 = affine_forward(a2, W2, b2)
        self.cache = (fc_cache1, relu_cache, fc_cache2)
        return out

    def backward(self, dout):
        fc_cache1, relu_cache, fc_cache2 = self.cache
        da2, dW2, db2 = affine_backward(dout, fc_cache2)
        da = relu_backward(da2, relu_cache)
        dx, dW1, db1 = affine_backward(da, fc_cache1)

        grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}
        return dx, grads


class AffineSigmoidScale(Layer):
    '''A composite layer consisting of an Affine-Sigmoid-Scale transform.'''

    def __init__(self, inputDim, outputDim, weightScale=None):
        weightScale = weightScale or np.sqrt(2.0 / inputDim)

        self.params = {}
        self.params['W'] = np.random.normal(scale=weightScale, size=(inputDim, outputDim))
        self.params['b'] = np.zeros(outputDim)
        self.params['sigma'] = np.ones(outputDim)
        self.params['mu'] = np.zeros(outputDim)
        self.regularizedParams = ['W']

    def forward(self, x):
        W, b = self.params['W'], self.params['b']
        sigma, mu = self.params['sigma'], self.params['mu']
        a, fc_cache = affine_forward(x, W, b)
        a2, sigmoid_cache = sigmoid_forward(a)
        out, scale_cache = scale_shift_forward(a2, sigma, mu)

        self.cache = (fc_cache, sigmoid_cache, scale_cache)
        return out

    def backward(self, dout):
        fc_cache, sigmoid_cache, scale_cache = self.cache
        da2, dsigma, dmu = scale_shift_backward(dout, scale_cache)
        da = sigmoid_backward(da2, sigmoid_cache)
        dx, dW, db = affine_backward(da, fc_cache)

        grads = {'W': dW, 'b': db, 'sigma': dsigma, 'mu': dmu}
        return dx, grads


class Affine(Layer):

    def __init__(self, inputDim, outputDim, weightScale=None):
        weightScale = weightScale or np.sqrt(2.0 / inputDim)
        self.params = {}
        self.params['W'] = np.random.normal(scale=weightScale, size=(inputDim, outputDim))
        self.params['b'] = np.zeros(outputDim)
        self.regularizedParams = ['W']

    def forward(self, x):
        W = self.params['W']
        b = self.params['b']
        out, self.cache = affine_forward(x, W, b)
        return out

    def backward(self, dout):
        dx, dW, db = affine_backward(dout, self.cache)
        grads = {'W': dW, 'b': db}
        return dx, grads
