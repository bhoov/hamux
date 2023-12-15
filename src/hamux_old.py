"""A minimal implementation of HAMs in JAX. Unlike pytorch, this implementation operates on individual samples."""

import jax.numpy as jnp
import jax
import numpy as np
import functools as ft
from typing import *
from dataclasses import dataclass
import equinox as eqx
import jax.tree_util as jtu

## LAGRANGIANS
def lagr_identity(x): 
    """The Lagrangian whose activation function is simply the identity."""
    return 0.5 * jnp.power(x, 2).sum()

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

def lagr_rexp(x, 
             beta:float=1.0): # Inverse temperature
    """Rectified exponential activation function"""
    xclipped = jnp.maximum(x, 0)
    return 1 / beta * (jnp.exp(beta * xclipped)-xclipped).sum()

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
    return _lagr_tanh(x, beta)

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
    return _lagr_sigmoid(x, beta=beta, scale=scale)

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
                   axis=-1, # Which axis to normalize
                  ): 
    """Spherical norm activation function"""
    xmean = x.mean(axis, keepdims=True)
    xmeaned = x - xmean
    denominator = jnp.sqrt(jnp.power(xmeaned, 2).mean(axis, keepdims=True) + eps)
    return gamma * xmeaned / denominator + delta

def lagr_spherical_norm(x:jnp.ndarray, 
                   gamma:float=1.0, # Scale the stdev
                   delta:Union[float, jnp.ndarray]=0., # Shift the mean
                   axis=-1, # Which axis to normalize
                   eps=1e-5, # Prevent division by 0
                  ): 
    """Lagrangian of the spherical norm activation function"""
    y = jnp.sqrt(jnp.power(x, 2).sum(axis, keepdims=True) + eps)
    return (gamma * y + (delta * x).sum()).sum()

## Neurons
class Neurons(eqx.Module):
    lagrangian: Callable
    shape: Tuple[int]
    def __init__(self, 
                 lagrangian:Union[Callable, eqx.Module], 
                 shape:Union[int, Tuple[int]]
                ):
        super().__init__()
        self.lagrangian = lagrangian
        if isinstance(shape, int):
            shape = (shape,)
        self.shape = shape
        
    def activations(self, x):
        return jax.grad(self.lagrangian)(x)
    
    def g(self, x):
        return self.activations(x)
    
    def energy(self, g, x):
        """Assume vectorized"""
        return jnp.multiply(g, x).sum() - self.lagrangian(x)
    
    def init(self, bs:Optional[int]=None):
        """Return an empty state of the correct shape"""
        if bs is None or bs == 0:
            return jnp.zeros(*self.shape)
        return jnp.zeros((bs, *self.shape))
    
    def __repr__(self):
        return f"Neurons(lagrangian={self.lagrangian}, shape={self.shape})"
    

## HAM
class HAM(eqx.Module):
    neurons: Dict[str, Neurons]
    synapses: Dict[str, eqx.Module]
    connections: List[Tuple[Tuple, str]]
    
    def __init__(self, neurons, synapses, connections):
        self.neurons = neurons
        self.synapses = synapses
        self.connections = connections
        
    @property
    def n_neurons(self): return len(self.neurons)
    @property
    def n_synapses(self): return len(self.synapses)
    @property
    def n_connections(self): return len(self.connections)

    def activations(self, xs):
        """Convert hidden states to activations"""
        gs = {k: v.g(xs[k]) for k,v in self.neurons.items()}
        return gs
    
    def init_states(self, bs:Optional[int]=None):
        """Initialize states"""
        xs = {k: v.init(bs) for k,v in self.neurons.items()}
        return xs
    
    def neuron_energy(self, gs, xs):
        """The sum of all neuron energies"""
        energies = [self.neurons[k].energy(gs[k], xs[k]) for k in self.neurons.keys()]
        return jnp.sum(jnp.stack(energies))
    
    def synapse_energy(self, gs):
        """The sum of all synapse energies"""
        def get_energy(neuron_set, s):
            mygs = [gs[k] for k in neuron_set]
            return self.synapses[s](*mygs)
        energies = [get_energy(neuron_set, s) for neuron_set, s in self.connections]
        return jnp.sum(jnp.stack(energies))

    def energy(self, gs, xs):
        """The complete energy of the HAM"""
        return self.neuron_energy(gs, xs) + self.synapse_energy(gs)
    
    def dEdg(self, gs, xs, return_energy=False):
        """Calculate gradient of system energy wrt activations using cute trick"""
        if return_energy:
            return jax.value_and_grad(self.energy)(gs, xs)
        return jax.grad(self.energy)(gs, xs)
    
    def dEdg_manual(self, gs, xs, return_energy=False):
        """Calculate gradient of system energy wrt activations using cute trick"""
        dEdg = jtu.tree_map(lambda x, s: x + s, xs, jax.grad(self.synapse_energy)(gs))
        if return_energy:
            return dEdg, self.energy(gs, xs)
        return dEdg
    
    def vectorize(self):
        """Compute new HAM with same API, except all methods expect a batch dimension"""
        return VectorizedHAM(self)
    
    def unvectorize(self):
        return self
    
class VectorizedHAM(eqx.Module):
    """Re-expose HAM API with vectorized inputs"""
    _ham: eqx.Module
    
    def __init__(self, ham):
        self._ham = ham
        
    @property
    def neurons(self): return self._ham.neurons
    @property
    def synapses(self): return self._ham.synapses
    @property
    def connections(self): return self._ham.connections
    @property
    def n_neurons(self): return self._ham.n_neurons
    @property
    def n_synapses(self): return self._ham.n_synapses
    @property
    def n_connections(self): return self._ham.n_connections
    @property
    def _batch_axes(self:HAM):
        """A helper function to tell vmap to batch along the 0'th dimension of each state in the HAM."""
        return {k: 0 for k in self._ham.neurons.keys()}
    
    def init_states(self, bs=None):
        return self._ham.init_states(bs)
    
    def activations(self, xs):
        return jax.vmap(self._ham.activations, in_axes=(self._batch_axes,))(xs)

    def synapse_energy(self, gs):
        return jax.vmap(self._ham.synapse_energy, in_axes=(self._batch_axes,))(gs)
    
    def neuron_energy(self, gs, xs):
        return jax.vmap(self._ham.neuron_energy, in_axes=(self._batch_axes, self._batch_axes))(gs, xs)
    
    def energy(self, gs, xs):
        return jax.vmap(self._ham.energy, in_axes=(self._batch_axes, self._batch_axes))(gs, xs)
    
    def dEdg(self, gs, xs, return_energy=False):
        return jax.vmap(self._ham.dEdg, in_axes=(self._batch_axes, self._batch_axes, None))(gs, xs, return_energy)
        
    def dEdg_manual(self, gs, xs, return_energy=False):
        return jax.vmap(self._ham.dEdg, in_axes=(self._batch_axes, self._batch_axes, None))(gs, xs, return_energy)
    
    def unvectorize(self):
        return self._ham
    
    def vectorize(self):
        return self