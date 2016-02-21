import numpy as np

from models.layers import *
from models.layer_utils import *


class FullyConnectedNet:
  """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. For a network with L layers,
  the architecture will be

  {affine - relu - affine} x (L - 1) - affine - softmax

  where batch normalization and dropout are optional, and the {...} block is
  repeated L - 1 times.

  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """

  def __init__(self, hidden_dims, input_dim=28*28, num_classes=10,
               reg=0.0, weight_scale=1e-2, dtype=np.float32):
    """
    Initialize a new FullyConnectedNet.

    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer. The second affine
      unit in each layer has same input/output dimensions as the output dimension of the previous affine
      unit.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    """
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.dtype = dtype
    self.params = {}

    # Initialize parameters.
    inputDim = input_dim
    for layer in range(self.num_layers):
      if layer < len(hidden_dims):
        outputDim = hidden_dims[layer]
        self.params['W' + str(layer + 1) + 'f'] = np.random.normal(scale=weight_scale, size=(inputDim, outputDim))
        self.params['b' + str(layer + 1) + 'f'] = np.zeros(outputDim)
        self.params['W' + str(layer + 1) + 's'] = np.random.normal(scale=weight_scale, size=(outputDim, outputDim))
        self.params['b' + str(layer + 1) + 's'] = np.zeros(outputDim)
      else:
        outputDim = num_classes
        self.params['W' + str(layer + 1)] = np.random.normal(scale=weight_scale, size=(inputDim, outputDim))
        self.params['b' + str(layer + 1)] = np.zeros(outputDim)
      inputDim = outputDim

    # Cast all parameters to the correct datatype
    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)

  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.

    Input / output: Same as TwoLayerNet above.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    # Forward pass.
    caches = []
    out = X
    for layer in range(1, self.num_layers):
      W1 = self.params['W' + str(layer) + 'f']
      b1 = self.params['b' + str(layer) + 'f']
      W2 = self.params['W' + str(layer) + 's']
      b2 = self.params['b' + str(layer) + 's']

      out, cache = affine_relu_affine_forward(out, W1, b1, W2, b2)
      caches.append(cache)

    W_out = self.params['W' + str(self.num_layers)]
    b_out = self.params['b' + str(self.num_layers)]
    scores, cache_out = affine_forward(out, W_out, b_out)

    # If test mode return early.
    if mode == 'test':
      return scores

    # Backward pass.
    grads = {}
    loss, dout = softmax_loss(scores, y)
    dout, dW, db = affine_backward(dout, cache_out)
    grads['W' + str(self.num_layers)] = dW
    grads['b' + str(self.num_layers)] = db

    for layer in reversed(range(1, self.num_layers)):
      cache = caches[layer - 1]
      dout, dW1, db1, dW2, db2 = affine_relu_affine_backward(dout, cache)
      grads['W' + str(layer) + 'f'] = dW1
      grads['b' + str(layer) + 'f'] = db1
      grads['W' + str(layer) + 's'] = dW2
      grads['b' + str(layer) + 's'] = db2

    # Regularization terms.
    regLoss = 0
    for layer in range(1, self.num_layers):
      W1 = self.params['W' + str(layer) + 'f']
      W2 = self.params['W' + str(layer) + 's']
      grads['W' + str(layer) + 'f'] += self.reg * W1
      grads['W' + str(layer) + 's'] += self.reg * W2
      regLoss += np.sum(W1 * W1) + np.sum(W2 * W2)
    W_out = self.params['W' + str(self.num_layers)]
    grads['W' + str(self.num_layers)] += self.reg * W_out
    regLoss += np.sum(W_out * W_out)

    loss += 0.5 * self.reg * regLoss

    return loss, grads
