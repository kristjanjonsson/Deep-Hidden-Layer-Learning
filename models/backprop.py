from __future__ import division
import numpy as np

"""
Functions for calculating forward and backprop for simple single units.
"""


def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
  examples, where each example x[i] has shape (d_1, ..., d_k). We will
  reshape each input into a vector of dimension D = d_1 * ... * d_k, and
  then transform it to an output vector of dimension M.

  Inputs:
  - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
  - w: A numpy array of weights, of shape (D, M)
  - b: A numpy array of biases, of shape (M,)

  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  N = x.shape[0]
  D = np.prod(x.shape[1:])
  xtrans = x.reshape((N, D))
  out = xtrans.dot(w) + b
  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  N = dout.shape[0]
  D, M = w.shape
  x_shaped = x.reshape((N, D))
  dx = dout.dot(w.T).reshape(x.shape)
  dw = x_shaped.T.dot(dout)
  db = dout.sum(axis=0)

  return dx, dw, db


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = x * (x > 0)
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  x = cache
  dx = dout * (x > 0)
  return dx


def sigmoid_forward(x):
  """
  Computes the forward pass for a layer of sigmoid units.

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: out
  """
  out = sigmoid(x)
  cache = out
  return out, cache


def sigmoid_backward(dout, cache):
  """
  Computes the backward pass for a layer of sigmoid units.

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Output out, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  out = cache
  dx = 1 - out
  dx *= out
  dx *= dout
  return dx


def scale_shift_forward(x, sigma, mu):
  """
  Computes element wise: out = sigma * x + mu

  Inputs:
  - x: A numpy array containing input data, of shape (N, D)
  - sigma: A numpy array of weights, of shape (D,)
  - mu: A numpy array of biases, of shape (d,)

  Returns a tuple of:
  - out: output, of shape (N, D)
  - cache: (x, sigma)
  """
  out = sigma * x + mu
  cache = (x, sigma)
  return out, cache


def scale_shift_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, D)
  - cache: Tuple of:
    - out: output, of shape (N, D)
    - cache: (x, sigma)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, D)
  - dsigma: Gradient with respect to sigma, of shape (D, )
  - dmu: Gradient with respect to mu, of shape (D,)
  """
  x, sigma = cache
  dx = dout * sigma
  dsigma = (dout * x).sum(axis=0)
  dmu = dout.sum(axis=0)
  return dx, dsigma, dmu


def sigmoid(x):
  '''Elementwise sigmoid.'''
  out = np.empty_like(x)
  pos_idx = x > 0
  out[pos_idx] = 1 / (1 + np.exp(-x[pos_idx]))

  exp = np.exp(x[~pos_idx])
  out[~pos_idx] = exp / (exp + 1)
  return out
