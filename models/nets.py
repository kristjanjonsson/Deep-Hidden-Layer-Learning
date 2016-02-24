import numpy as np

from models.layers import AffineReluAffine, Affine
from models.losses import softmax_loss, squared_loss


class Net:
  """
  A custom neural net.
  """

  def __init__(self, layers, loss_function, reg=0.0, dtype=np.float32):
    """
    Wrap a net around layers with the given loss function.

    Inputs:
    - layers: A list of layer objects that specify the architecture. The dimensions of
      consequtive layers needs to match.
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
    grads = {}
    loss, dout = self.loss_function(scores, y)
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

  @property
  def params(self):
    '''Returns a dictionary of all the parameters (numLayer, paramName): param.
    This is just so gradient checking works.'''
    d = {}
    for i, layer in enumerate(self.layers):
      for p, v in layer.params.items():
        d[(i, p)] = v
    return d


class Subnet(Net):
  """
  A net that was obtained as a part of a parent net and inherits its properties.
  """

  def __init__(self, parent_model, layers, loss_function):
    self.parent_model = parent_model
    super().__init__(layers, loss_function, reg=parent_model.reg, dtype=parent_model.dtype)


class AffineReluAffineNet(Net):
  """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. For a network with L layers,
  the architecture will be

  {affine - relu - affine} x (L - 1) - affine - softmax
  """
  def __init__(self, hidden_dims, input_dim=28*28, num_classes=10,
               reg=0.0, weight_scale=1e-2, dtype=np.float32):

    layers = []

    # Initialize layers and parameters.
    inputDim = input_dim
    for layer in range(len(hidden_dims)):
      outputDim = hidden_dims[layer]
      layers.append(AffineReluAffine(inputDim, outputDim, weightScale=weight_scale))
      inputDim = outputDim

    outputDim = num_classes
    layers.append(Affine(inputDim, outputDim, weightScale=weight_scale))

    super().__init__(layers, softmax_loss, reg=reg, dtype=dtype)

  def __getitem__(self, key):
    '''Returns the submodel layers[key] with a squared error loss attached to the head.'''
    sublayers = self.layers[key]
    if type(sublayers) is not list:
      sublayers = [sublayers]

    return Subnet(self, sublayers, squared_loss)


