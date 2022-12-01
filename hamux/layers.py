# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/01_layers.ipynb.

# %% auto 0
__all__ = ['IdentityLayer', 'RepuLayer', 'ReluLayer', 'SoftmaxLayer', 'SigmoidLayer', 'TanhLayer', 'ExpLayer', 'RexpLayer',
           'LayerNormLayer', 'Layer', 'MakeLayer']

# %% ../nbs/01_layers.ipynb 9
import jax
import jax.numpy as jnp
from typing import *
import treex as tx
from abc import ABC, abstractmethod
from flax import linen as nn
from .lagrangians import *
import functools as ft
from fastcore.meta import delegates
from fastcore.utils import *
from fastcore.basics import *

# %% ../nbs/01_layers.ipynb 10
class Layer(tx.Module):
    """The energy building block of any activation in our network that we want to hold state over time"""
    lagrangian: tx.Module 
    shape: Tuple
    tau: float
    use_bias: bool
    bias: jnp.ndarray = tx.Parameter.node(default=None)

    def __init__(self, 
                 lagrangian:tx.Module, # Factory function creating lagrangian module describing
                 # lagrangian:Union[Callable, tx.Module], Either a factory function or the module itself depending on `init_lagr's` value
                 shape:Tuple[int], # Number and shape of neuron assembly
                 tau:float=1.0, # Time constant
                 use_bias:bool=False, # Add bias?
                 init_lagrangian=False, # Initialize the lagrangian with kwargs?
                 name:str=None, # Overwrite default class name, if provided
                 **kwargs, # Passed to lagranigan factory
                ): 
        self.lagrangian = lagrangian(**kwargs) if init_lagrangian else lagrangian
        self.shape = shape
        assert tau > 0.0, "Tau must be positive and non-zero"
        self.tau = tau
        self.use_bias = use_bias

        if name is not None:
            self.name = name
        
    def energy(self, x):
        """The predefined energy of a layer, defined for any lagrangian"""
        if self.initializing():
            if self.use_bias:
                self.bias = nn.initializers.normal(0.02)(tx.next_key(), self.shape)
        x2 = x - self.bias if self.use_bias else x # Is this an issue?

        # When jitted, this is no slower than the optimized `@` vector multiplication
        return jnp.multiply(self.g(x), x2).sum() - self.lagrangian(x2)
        
    def __call__(self, x):
        """Alias for `self.energy`. Helps simplify treex's `.init` method"""
        return self.energy(x)
            
    def activation(self, x):
        """The derivative of the lagrangian is our activation or Gain function `g`. 
        
        Defined to operate over input states `x` of shape `self.shape`
        """
        if self.initializing():
            if self.use_bias:
                self.bias = nn.initializers.normal(0.02)(tx.next_key(), self.shape)
        x2 = x - self.bias if self.use_bias else x
        return jax.grad(self.lagrangian)(x2)
    
    def g(self, x):
        """Alias for `self.activation`"""
        return self.activation(x)

    def init_state(self, 
                   bs: int = None, # Batch size
                   rng=None): # If given, initialize states from a normal distribution with this key
        """Initialize the states of this layer, with correct shape.
        
        If `bs` is provided, return tensor of shape (bs, *self.shape), otherwise return self.shape
        By default, initialize layer state to all 0.
        """
        layer_shape = self.shape if bs is None else (bs, *self.shape)
        if rng is not None:
            return jax.random.normal(rng, layer_shape)
        return jnp.zeros(layer_shape)

# %% ../nbs/01_layers.ipynb 20
def MakeLayer(lagrangian_factory:Callable, 
              name:Optional[str]=None): # Name of the new class
    """Hack to make it easy to create new layers from `Layer` utility class.
    
    `delegates` modifies the signature for all Layers. We want a different signature for each type of layer.

    So we redefine a local version of layer and delegate that for type inference.
    """
    global Layer

    @delegates(lagrangian_factory, keep=True)
    class Layer(Layer):
        __doc__ = Layer.__doc__
        
    out = partialler(Layer, lagrangian_factory, init_lagrangian=True, name=name)
    out.__doc__ = Layer.__doc__

    return out

# %% ../nbs/01_layers.ipynb 21
# Some reason, docstrings are not showing the new kwargs, and the docs for these are broken. 
IdentityLayer = MakeLayer(LIdentity, "identity_layer")
RepuLayer = MakeLayer(LRepu, "repu_layer")
ReluLayer = MakeLayer(LRelu, "relu_layer")
SoftmaxLayer = MakeLayer(LSoftmax, "softmax_layer")
SigmoidLayer = MakeLayer(LSigmoid, "sigmoid_layer")
TanhLayer = MakeLayer(LTanh, "tanh_layer")
ExpLayer = MakeLayer(LExp, "exp_layer")
RexpLayer = MakeLayer(LRexp, "rexp_layer")
LayerNormLayer = MakeLayer(LLayerNorm, "layernorm_layer")
