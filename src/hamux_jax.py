"""A minimal implementation of HAMs in JAX. Unlike pytorch, this implementation operates on individual samples."""

import jax.numpy as jnp
import jax
import numpy as np
import functools as ft
from typing import *
from dataclasses import dataclass
import equinox as eqx
import jax.tree_util as jtu

from lagrangians import *

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

    def connection_energies(self, gs):
        """Get the energy for each connection"""
        def get_energy(neuron_set, s):
            mygs = [gs[k] for k in neuron_set]
            return self.synapses[s](*mygs)

        return [get_energy(neuron_set, s) for neuron_set, s in self.connections]

    def energy_tree(self, gs, xs):
        """Return energies for each individual component"""

        neuron_energies = jtu.tree_map(lambda neuron, g, x: neuron.energy(g, x), self.neurons, gs, xs)
        connection_energies = self.connection_energies(gs)

        return {
            "neurons": neuron_energies,
            "connections": connection_energies
        }
    
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

    def energy_tree(self, gs, xs):
        """Return energies for each individual component"""
        neuron_energies = {k: self.neurons[k].energy(gs[k], xs[k]) for k in self.neurons.keys()}
        connection_energies = self.connection_energies(gs)
    
        return {
            "neurons": neuron_energies,
            "connections": connection_energies
        }

    def scaled_energy_f(self, gs, xs, energy_scales):
        """Compute energy after scaling each component down by energy scales, a pytree of the same structure as the output of `energy_tree`"""
        etree = self.energy_tree(gs, xs)
        etree = jtu.tree_map(lambda E, s: E * s, etree, energy_scales)
        return energy_from_tree(etree)

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

    def connection_energies(self, gs):
        return jax.vmap(self._ham.connection_energies, in_axes=(self._batch_axes,))(gs)
        
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

    def energy_tree(self, gs, xs):
        return jax.vmap(self._ham.energy_tree, in_axes=(self._batch_axes, self._batch_axes))(gs, xs)

    def scaled_energy_f(self, gs, xs, etree):
        return jax.vmap(self._ham.scaled_energy_f, in_axes=(self._batch_axes, self._batch_axes, None))(gs, xs, etree)
        
    def unvectorize(self):
        return self._ham
    
    def vectorize(self):
        return self

