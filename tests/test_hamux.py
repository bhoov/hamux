from hamux import Neurons, HAM, VectorizedHAM
import pytest
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from lagrangians import lagr_softmax
import jax.numpy as jnp
import jax
import jax.random as jr

neuron_shape = (5,)
beta = 3.0
neuron = Neurons(lagrangian=lambda x: lagr_softmax(x, beta=beta), shape=neuron_shape)
act_fn = lambda x: jax.nn.softmax(beta * x)


def test_init():
  assert neuron.init().shape == neuron_shape
  assert neuron.init(bs=3).shape == (3, *neuron_shape)


def test_activations():
  x = neuron.init()
  assert jnp.all(neuron.activations(x) == neuron.g(x))
  assert jnp.allclose(act_fn(x), neuron.g(x))


def test_ham_lengths(simple_ham: HAM):
  assert simple_ham.n_neurons == 2
  assert simple_ham.n_synapses == 1
  assert simple_ham.n_connections == 1


def test_vham_lengths(simple_ham: HAM):
  vham = simple_ham.vectorize()
  assert vham.n_neurons == 2
  assert vham.n_synapses == 1
  assert vham.n_connections == 1


def test_ham_dEdg(simple_ham: HAM):
  xs = simple_ham.init_states()
  gs = simple_ham.activations(xs)
  auto_E, auto_dEdg = jax.value_and_grad(simple_ham.energy)(gs, xs)
  man_E, man_dEdg = simple_ham.dEdg(gs, xs, return_energy=True)

  assert jnp.allclose(auto_E, man_E)
  assert jnp.allclose(auto_dEdg["image"], man_dEdg["image"])
  assert jnp.allclose(auto_dEdg["hidden"], man_dEdg["hidden"])


def test_vectorize_unvectorize(simple_ham: HAM):
  vham = simple_ham.vectorize()
  assert isinstance(vham, VectorizedHAM)
  assert isinstance(vham.unvectorize(), HAM)


@pytest.mark.slow
@pytest.mark.parametrize("stepsize", [0.001, 0.01, 0.1])
def test_ham_energies(simple_ham: HAM, stepsize, nsteps=10):
  energies = []
  xs = simple_ham.init_states()
  xs["image"] = jr.normal(jr.PRNGKey(1), xs["image"].shape)
  for i in range(nsteps):
    gs = simple_ham.activations(xs)
    E, dEdg = simple_ham.dEdg(gs, xs, return_energy=True)
    energies.append(E)
    xs = jax.tree_map(lambda x, dx: x - stepsize * dx, xs, dEdg)

  Estacked = jnp.stack(energies)
  assert Estacked.shape == (nsteps,)
  assert jnp.all(jnp.diff(jnp.array(energies)) <= 0)


@pytest.mark.slow
@pytest.mark.parametrize("stepsize", [0.001, 0.01, 0.1])
def test_vham_energies(simple_ham: HAM, stepsize, nsteps=10):
  bs = 3
  energies = []
  xs = simple_ham.init_states(bs=bs)
  vham = simple_ham.vectorize()
  xs["image"] = jr.normal(jr.PRNGKey(1), xs["image"].shape)
  for i in range(nsteps):
    gs = vham.activations(xs)
    E, dEdg = vham.dEdg(gs, xs, return_energy=True)
    energies.append(E)
    xs = jax.tree_map(lambda x, dx: x - stepsize * dx, xs, dEdg)

  Estacked = jnp.stack(energies).T
  assert Estacked.shape == (bs, nsteps)
  assert jnp.all(jnp.diff(Estacked, axis=-1) <= 0)
