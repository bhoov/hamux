import pytest

import equinox as eqx
import jax
import jax.numpy as jnp
from typing import *
from hamux import Neurons, HAM
import jax.random as jr
from src.lagrangians import lagr_identity, lagr_softmax

class SimpleSynapse(eqx.Module):
  W: jax.Array
  def __init__(self, key:jax.Array, shape:Tuple[int, int]):
    self.W = 0.1 * jr.normal(key, shape)

  def __call__(self, g1, g2):
    return -jnp.einsum("...d,de,...e->...", g1, self.W, g2)

@pytest.fixture
def simple_ham():
  d1, d2 = (5,7)
  neurons = {
    "image": Neurons(lagr_identity, (d1,)),
    "hidden": Neurons(lagr_softmax, (d2,))
  }
  synpases = {
    "s1": SimpleSynapse(jr.PRNGKey(0), (d1, d2))
  }
  connections = [
    # (vertices, hyperedge)
    (("image", "hidden"), "s1")
  ]
  ham = HAM(neurons, synpases, connections)
  return ham