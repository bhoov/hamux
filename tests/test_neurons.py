from neurons import Neurons
from lagrangians import lagr_softmax
import jax.numpy as jnp
import jax

neuron_shape = (5,)
beta = 3.
neuron = Neurons(lagrangian=lambda x: lagr_softmax(x, beta=beta), shape=neuron_shape)
act_fn = lambda x: jax.nn.softmax(beta * x)

def test_init():
  assert neuron.init().shape == neuron_shape
  assert neuron.init(bs=3).shape == (3, *neuron_shape)

def test_activations():
  x = neuron.init() 
  assert jnp.all(neuron.activations(x) == neuron.g(x))
  assert jnp.allclose(act_fn(x), neuron.g(x))