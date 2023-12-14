import jax
import jax.numpy as jnp
import equinox as eqx
from typing import *

class Neurons(eqx.Module):
  """Neurons represent dynamic variables in the HAM that are evolved during inference (i.e., memory retrieval/error correction)

  They have an evolving state (created using the `.init` function) that is stored outside the neuron layer itself
  """
  lagrangian: Callable
  shape: Tuple[int]

  def __init__(
    self, lagrangian: Union[Callable, eqx.Module], shape: Union[int, Tuple[int]]
  ):
    super().__init__()
    self.lagrangian = lagrangian
    if isinstance(shape, int):
      shape = (shape,)
    self.shape = shape

  def activations(self, x: jax.Array) -> jax.Array:
    return jax.grad(self.lagrangian)(x)

  def g(self, x: jax.Array) -> jax.Array:
    return self.activations(x)

  def energy(self, g: jax.Array, x: jax.Array) -> jax.Array:
    """Assume vectorized"""
    return jnp.multiply(g, x).sum() - self.lagrangian(x)

  def init(self, bs: Optional[int] = None) -> jax.Array:
    """Return an empty state of the correct shape"""
    if bs is None or bs == 0:
      return jnp.zeros(self.shape)
    return jnp.zeros((bs, *self.shape))

  def __repr__(self: jax.Array):
    return f"Neurons(lagrangian={self.lagrangian}, shape={self.shape})"
