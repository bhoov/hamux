# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/03_ham.ipynb.

# %% auto 0
__all__ = ['HAM']

# %% ../nbs/03_ham.ipynb 6
import jax
import jax.numpy as jnp
from typing import *
import treex as tx
from .layers import Layer
from .synapses import Synapse
import jax.tree_util as jtu
from .utils import pytree_save, pytree_load, to_pickleable, align_with_state_dict
import pickle
import functools as ft
from fastcore.utils import *
from abc import ABC, abstractmethod, abstractproperty
import hypernetx as hnetx

# %% ../nbs/03_ham.ipynb 8
class HAM(tx.Module):
    layers: List[Layer]
    synapses: List[Synapse]
    connections: List[Tuple[Tuple, int]]

    def __init__(self, layers, synapses, connections):
        self.layers = layers
        self.synapses = synapses
        self.connections = connections

    @property
    def n_layers(self): return len(self.layers)
    @property
    def n_synapses(self): return len(self.synapses)
    @property
    def n_connections(self): return len(self.connections)
    @property
    def layer_taus(self): return [layer.tau for layer in self.layers]
    def alphas(self, dt): return [dt / tau for tau in self.layer_taus]

# %% ../nbs/03_ham.ipynb 9
@patch
def activations(self:HAM, 
                xs:jnp.ndarray): # Collection of states for each layer
    """Turn a collection of states into a collection of activations"""
    gs = [self.layers[i].g(xs[i]) for i in range(len(xs))]
    return gs

# %% ../nbs/03_ham.ipynb 10
@patch
def layer_energy(self:HAM,
                 xs:jnp.ndarray): # Collection of states for each layer
    """The total contribution of the layers' contribution to the energy of the HAM"""
    energies = jnp.stack([self.layers[i].energy(x) for i, x in enumerate(xs)])
    return jnp.sum(energies)

@patch
def synapse_energy(self:HAM,
                   gs:jnp.ndarray): # Collection of activations of each layer
    """The total contribution of the synapses' contribution to the energy of the HAM.
    
    A function of the activations `gs` rather than the states `xs`
    """
    def get_energy(lset, k):
        mygs = [gs[i] for i in lset]
        synapse = self.synapses[k]
        return synapse.energy(*mygs)
    energies = jnp.stack([get_energy(lset, k) for lset, k in self.connections])
    return jnp.sum(energies)

@patch
def energy(self:HAM,
           xs:jnp.ndarray): # Collection of states for each layer
    """The complete energy of the HAM"""
    gs = self.activations(xs)
    energy = self.layer_energy(xs) + self.synapse_energy(gs)
    return energy

# %% ../nbs/03_ham.ipynb 12
@patch
def init_states(self:HAM, 
                bs=None, # Batch size of the states to initialize, if needed
                rng=None): # RNG seed for random initialization of the states, if non-zero initializations are desired
    """Initialize the states of every layer in the network"""
    if rng is not None:
        keys = jax.random.split(rng, self.n_layers)
        return [layer.init_state(bs, rng=key) for layer, key in zip(self.layers, keys)]
    return [layer.init_state(bs) for layer in self.layers]

@patch
def init_states_and_params(self:HAM, 
                           param_key, # RNG seed for random initialization of the parameters
                           bs=None, # Batch size of the states to initialize, if needed
                           state_key=None): # RNG seed for random initialization of the states, if non-zero initializations are desired
    """Initialize the states and parameters of every layer and synapse in the network"""
    # params don't need a batch size to initialize
    params = self.init(param_key, self.init_states(), call_method="energy")
    states = self.init_states(bs, rng=state_key)
    return states, params

# %% ../nbs/03_ham.ipynb 23
@patch
def dEdg(self:HAM, 
         xs:jnp.ndarray):
    """Calculate the gradient of system energy wrt. the activations

    Notice that we use an important mathematical property of the Legendre transform to take a mathematical shortcut,
    where dE_layer / dg = x
    """
    gs = self.activations(xs)
    return jtu.tree_map(
        lambda x, s: x + s, xs, jax.grad(self.synapse_energy)(gs)
    )

@patch
def updates(self:HAM,
            xs:jnp.ndarray): # Collection of states for each layer
    """The negative of our dEdg, computing the update direction each layer should descend"""
    return jtu.tree_map(lambda dg: -dg, self.dEdg(xs))

# %% ../nbs/03_ham.ipynb 29
@patch
def step(self:HAM,
    xs: List[jnp.ndarray], # Collection of current states for each layer
    updates: List[jnp.ndarray], # Collection of update directions for each state
    dt: float = 0.1, # Stepsize to take in direction of updates
    masks: Optional[List[jnp.ndarray]] = None, # Boolean mask, 0 if clamped neuron, and 1 elsewhere. A pytree identical to `xs`. Optional.
):
    """A discrete step down the energy using step size `dt` scaled by the `tau` of each layer"""
    taus = self.layer_taus
    alphas = [dt / tau for tau in taus] # Scaling factor of the update size of each layer
    if masks is not None:
        next_xs = jtu.tree_map(lambda x, u, m, alpha: x + alpha * u * m, xs, updates, masks, alphas)
    else:
        next_xs = jtu.tree_map(lambda x, u, alpha: x + alpha * u, xs, updates, alphas)
    return next_xs

# %% ../nbs/03_ham.ipynb 31
@patch
def _statelist_batch_axes(self:HAM):
    """A helper function to tell vmap to batch along the 0'th dimension of each state in the HAM."""
    return ([0 for _ in range(self.n_layers)],)
    
@patch
def vactivations(self:HAM, 
                 xs: List[jnp.ndarray]): # Collection of states for each layer
    """A vectorized version of `activations`"""
    return jax.vmap(self.activations, in_axes=self._statelist_batch_axes())(xs)

@patch
def venergy(self:HAM, 
            xs: List[jnp.ndarray]): # Collection of states for each layer
    """A vectorized version of `energy`"""
    return jax.vmap(self.energy, in_axes=self._statelist_batch_axes())(xs)

@patch
def vdEdg(self:HAM, 
          xs: List[jnp.ndarray]): # Collection of states for each layer
    """A vectorized version of `dEdg`"""
    return jax.vmap(self.dEdg, in_axes=self._statelist_batch_axes())(xs)

@patch
def vupdates(self:HAM,
             xs: List[jnp.ndarray]): # Collection of states for each layer
    """A vectorized version of `updates`"""
    return jax.vmap(self.updates, in_axes=self._statelist_batch_axes())(xs)

# %% ../nbs/03_ham.ipynb 38
@patch
def load_state_dict(self:HAM, 
                    state_dict:Any): # The dictionary of all parameters, saved by `save_state_dict`
    """Load the state dictionary for a HAM"""
    if not self.initialized:
        _, self = self.init_states_and_params(jax.random.PRNGKey(0), 1)
    self.connections = state_dict["connections"]
    self.layers = align_with_state_dict(self.layers, state_dict["layers"])
    self.synapses = align_with_state_dict(self.synapses, state_dict["synapses"])
    return self

@patch
def to_state_dict(self:HAM):
    """Convert HAM to state dictionary of parameters and connections"""
    return jtu.tree_map(to_pickleable, self.to_dict())

@patch
def save_state_dict(self:HAM, 
                    fname:Union[str, Path], # Filename of checkpoint to save
                    overwrite:bool=True): # Overwrite an existing file of the same name?
    """Save the state dictionary for a HAM"""
    to_save = self.to_state_dict()
    pytree_save(to_save, fname, overwrite=overwrite)

@patch
def load_ckpt(self:HAM, 
              ckpt_f:Union[str, Path]): # Filename of checkpoint to load
    """Load from file name"""
    with open(ckpt_f, "rb") as fp:
        state_dict = pickle.load(fp)
    return self.load_state_dict(state_dict)
