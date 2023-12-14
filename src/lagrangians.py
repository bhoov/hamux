"""Default lagrangian functions that correspond to commonly used non-linearities in Neural networks.

1. Lagrangians return a scalar.
2. Lagrangians are convex
3. The derivative of a lagrangian w.r.t. its input is the activation function typically used in Neural Networks.

Feel free to use these as inspiration for building your own lagrangians. They're simple enough
"""

import jax.numpy as jnp
import jax
from typing import *

## LAGRANGIANS
def lagr_identity(x): 
    """The Lagrangian whose activation function is simply the identity."""
    return 0.5 * jnp.power(x, 2).sum()

def _repu(x, n):
    return jnp.maximum(x, 0) ** n

def lagr_repu(x, 
              n): # Degree of the polynomial in the power unit
    """Rectified Power Unit of degree `n`"""
    return 1 / n * jnp.power(jnp.maximum(x, 0), n).sum()

def lagr_relu(x):
    """Rectified Linear Unit. Same as repu of degree 2"""
    return lagr_repu(x, 2)

def lagr_softmax(x,
                 beta:float=1.0, # Inverse temperature
                 axis:int=-1): # Dimension over which to apply logsumexp
    """The lagrangian of the softmax -- the logsumexp"""
    return (1/beta * jax.nn.logsumexp(beta * x, axis=axis, keepdims=False))

def lagr_exp(x, 
             beta:float=1.0): # Inverse temperature
    """Exponential activation function, as in [Demicirgil et al.](https://arxiv.org/abs/1702.01929). Operates elementwise"""
    return 1 / beta * jnp.exp(beta * x).sum()

def _rexp(x, 
          beta:float=1.0 # Inverse temperature
          ): 
    """Rectified exponential activation function"""
    xclipped = jnp.maximum(x, 0)
    return jnp.exp(beta * xclipped) - 1

def lagr_rexp(x, 
             beta:float=1.0): # Inverse temperature
    """Lagrangian of the Rectified exponential activation function"""
    xclipped = jnp.maximum(x, 0)
    return (jnp.exp(beta * xclipped)/beta-xclipped).sum()


@jax.custom_jvp
def _lagr_tanh(x, beta=1.0):
    return 1 / beta * jnp.log(jnp.cosh(beta * x))

@_lagr_tanh.defjvp
def _lagr_tanh_defjvp(primals, tangents):
    x, beta = primals
    x_dot, beta_dot = tangents
    primal_out = _lagr_tanh(x, beta)
    tangent_out = jnp.tanh(beta * x) * x_dot
    return primal_out, tangent_out

def lagr_tanh(x, 
              beta=1.0): # Inverse temperature
    """Lagrangian of the tanh activation function"""
    return _lagr_tanh(x, beta).sum()

@jax.custom_jvp
def _lagr_sigmoid(x, 
                  beta=1.0, # Inverse temperature
                  scale=1.0): # Amount to stretch the range of the sigmoid's lagrangian
    """The lagrangian of a sigmoid that we can define custom JVPs of"""
    return scale / beta * jnp.log(jnp.exp(beta * x) + 1)

def _tempered_sigmoid(x, 
                     beta=1.0, # Inverse temperature
                     scale=1.0): # Amount to stretch the range of the sigmoid
    """The basic sigmoid, but with a scaling factor"""
    return scale / (1 + jnp.exp(-beta * x))

@_lagr_sigmoid.defjvp
def _lagr_sigmoid_jvp(primals, tangents):
    x, beta, scale = primals
    x_dot, beta_dot, scale_dot = tangents
    primal_out = _lagr_sigmoid(x, beta, scale)
    tangent_out = _tempered_sigmoid(x, beta=beta, scale=scale) * x_dot # Manually defined sigmoid
    return primal_out, tangent_out

def lagr_sigmoid(x, 
                 beta=1.0, # Inverse temperature
                 scale=1.0): # Amount to stretch the range of the sigmoid's lagrangian
    """The lagrangian of the sigmoid activation function"""
    return _lagr_sigmoid(x, beta=beta, scale=scale).sum()


def _simple_layernorm(x:jnp.ndarray, 
                   gamma:float=1.0, # Scale the stdev
                   delta:Union[float, jnp.ndarray]=0., # Shift the mean
                   axis=-1, # Which axis to normalize
                   eps=1e-5, # Prevent division by 0
                  ): 
    """Layer norm activation function"""
    xmean = x.mean(axis, keepdims=True)
    xmeaned = x - xmean
    denominator = jnp.sqrt(jnp.power(xmeaned, 2).mean(axis, keepdims=True) + eps)
    return gamma * xmeaned / denominator + delta

def lagr_layernorm(x:jnp.ndarray, 
                   gamma:float=1.0, # Scale the stdev
                   delta:Union[float, jnp.ndarray]=0., # Shift the mean
                   axis=-1, # Which axis to normalize
                   eps=1e-5, # Prevent division by 0
                  ): 
    """Lagrangian of the layer norm activation function"""
    D = x.shape[axis] if axis is not None else x.size
    xmean = x.mean(axis, keepdims=True)
    xmeaned = x - xmean
    y = jnp.sqrt(jnp.power(xmeaned, 2).mean(axis, keepdims=True) + eps)
    return (D * gamma * y + (delta * x).sum()).sum()

def _simple_spherical_norm(x:jnp.ndarray, 
                   gamma:float=1.0, # Scale the stdev
                   delta:Union[float, jnp.ndarray]=0., # Shift the mean
                   axis=-1, # Which axis to normalize
                   eps=1e-5, # Prevent division by 0
                  ): 
    """Spherical norm activation function"""
    xnorm = jnp.sqrt(jnp.power(x, 2).sum(axis, keepdims=True) + eps)
    return gamma * x / xnorm + delta

def lagr_spherical_norm(x:jnp.ndarray, 
                   gamma:float=1.0, # Scale the stdev
                   delta:Union[float, jnp.ndarray]=0., # Shift the mean
                   axis=-1, # Which axis to normalize
                   eps=1e-5, # Prevent division by 0
                  ): 
    """Lagrangian of the spherical norm activation function"""
    y = jnp.sqrt(jnp.power(x, 2).sum(axis, keepdims=True) + eps)
    return (gamma * y + (delta * x).sum()).sum()