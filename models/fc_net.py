import numpy as np

from models.layers import *
from models.layer_utils import *
from models.composite import *


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
    self.layers = []

    # Initialize layers and parameters.
    inputDim = input_dim
    for layer in range(len(hidden_dims)):
      outputDim = hidden_dims[layer]
      self.layers.append(AffineReluAffine(inputDim, outputDim, weightScale=weight_scale))
      inputDim = outputDim

    outputDim = num_classes
    self.layers.append(Affine(inputDim, outputDim, weightScale=weight_scale))

    # Cast all parameters to the correct datatype
    for layer in self.layers:
      for k, v in layer.params.items():
        layer.params[k] = v.astype(dtype)

  @property
  def params(self):
    d = {}
    for i, layer in enumerate(self.layers):
      for p, v in layer.params.items():
        d[(i, p)] = v
    return d

  @property
  def grads(self):
    d = {}
    for i, layer in enumerate(self.layers):
      for p, v in layer.grads.items():
        d[(i, p)] = v
    return d


  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.

    Input / output: Same as TwoLayerNet above.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    # Forward pass.
    out = X
    for layer in self.layers:
      out = layer.forward(out)
    scores = out

    # If test mode return early.
    if mode == 'test':
      return scores

    # Backward pass.
    grads = {}
    loss, dout = softmax_loss(scores, y)
    for i in reversed(range(self.num_layers)):
      layer = self.layers[i]
      dout, grads_i = layer.backward(dout)
      grads.update({(i, p): w for p, w in grads_i.items()})

    # Regularization terms.
    regLoss = 0
    for i, layer in enumerate(self.layers):
      for paramName in layer.regularizedParams:
        W = layer.params[paramName]
        regLoss += np.sum(W * W)
        grads[(i, paramName)] += self.reg * W
    loss += 0.5 * self.reg * regLoss

    return loss, grads

  def __getitem__(self, key):
    '''Returns the submodel layer[key] with a squared error loss attached to the head.'''
    return NotImplementedError()


class Net:
  """
  A custom neural net.
  """

  def __init__(self, layers, loss_function, reg=0.0, dtype=np.float32):
    """
    Initialize a new FullyConnectedNet.

    Inputs:
    - layers: A list of layer objects that specify the architecture. The
      inputs and outputs of each layer needs to match.
    - reg: Scalar giving L2 regularization strength.
    - loss: A loss function.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    """
    self.layers = layers
    self.num_layers = len(layers)
    self.loss_function = loss_function
    self.reg = reg
    self.dtype = dtype

    # Cast all parameters to the correct datatype
    for layer in self.layers:
      for k, v in layer.params.items():
        layer.params[k] = v.astype(dtype)

  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    # Forward pass.
    out = X
    for layer in self.layers:
      out = layer.forward(out)
    scores = out

    # If test mode return early.
    if mode == 'test':
      return scores

    # Backward pass.
    grads = []
    loss, dout = self.loss_function(scores, y)
    for layer in reversed(self.layers):
      dout, layer_grads = layer.backward(dout)
      grads.append(layer_grads)

    # Regularization terms.
    regLoss = 0
    for layer in self.layers:
      for paramName in layer.params:
        if layer.isRegularised[paramName]:
          W = layer.params[paramName]
          regLoss += np.sum(W * W)
          layer.grads[paramName] += self.reg * W
    loss += 0.5 * self.reg * regLoss

    return loss, grads
