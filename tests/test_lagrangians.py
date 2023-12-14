import pytest
from lagrangians import (
    lagr_identity,
    _repu,
    lagr_repu,
    lagr_softmax,
    lagr_exp,
    _rexp,
    lagr_rexp,
    lagr_tanh,
    _tempered_sigmoid,
    lagr_sigmoid,
    _simple_layernorm,
    lagr_layernorm,
    _simple_spherical_norm,
    lagr_spherical_norm,
)
import jax
import jax.numpy as jnp
import jax.random as jr

key = jr.PRNGKey(0)
xtest = jr.normal(key, (16,)) * 5 - 3


def test_identity():
    assert jnp.allclose(xtest, jax.grad(lagr_identity)(xtest))


def test_repu():
    # Same as ReLU
    assert jnp.allclose(_repu(xtest, 1), jax.grad(lambda x: lagr_repu(x, 2))(xtest))
    # Higher degrees
    assert jnp.allclose(
        jnp.maximum(xtest, 0) ** 3, jax.grad(lambda x: lagr_repu(x, 4))(xtest)
    )


@pytest.mark.parametrize(
    "beta",
    [0.1, 1.0, 5.0],
)
def test_softmax(beta: float):
    assert jnp.allclose(
        jax.nn.softmax(beta * xtest), jax.grad(lambda x: lagr_softmax(x, beta))(xtest)
    )


@pytest.mark.parametrize(
    "beta",
    [0.1, 1.0, 5.0],
)
def test_exp(beta: float):
    assert jnp.allclose(
        jnp.exp(beta * xtest), jax.grad(lambda x: lagr_exp(x, beta))(xtest)
    )


@pytest.mark.parametrize(
    "beta",
    [0.1, 1.0, 5.0],
)
def test_rexp(beta: float):
    assert jnp.allclose(
        _rexp(xtest, beta), jax.grad(lambda x: lagr_rexp(x, beta))(xtest)
    )


@pytest.mark.parametrize(
    "beta",
    [0.1, 1.0, 5.0],
)
def test_tanh(beta: float):
    assert jnp.allclose(
        jnp.tanh(beta * xtest), jax.grad(lambda x: lagr_tanh(x, beta))(xtest)
    )


@pytest.mark.parametrize("beta", [0.1, 1.0, 5.0])
@pytest.mark.parametrize("scale", [0.1, 1.0, 5.0])
def test_tempered_sigmoid(beta: float, scale: float):
    assert jnp.allclose(
        _tempered_sigmoid(xtest, beta=beta, scale=scale),
        jax.grad(lambda x: lagr_sigmoid(x, beta=beta, scale=scale))(xtest),
    )


@pytest.mark.parametrize("gamma", [0.1, 1.0, 5.0])
@pytest.mark.parametrize("delta", [0.1, 1.0, 5.0])
def test_layernorm(gamma: float, delta: float):
    delta = jnp.ones_like(xtest) * delta
    assert jnp.allclose(
        _simple_layernorm(xtest, gamma=gamma, delta=delta),
        jax.grad(lambda x: lagr_layernorm(x, gamma=gamma, delta=delta))(xtest),
        rtol=1e-3,
    )


@pytest.mark.parametrize("gamma", [0.1, 1.0, 5.0])
@pytest.mark.parametrize("delta", [0.1, 1.0, 5.0])
def test_spherical_norm(gamma: float, delta: float):
    delta = jnp.ones_like(xtest) * delta
    assert jnp.allclose(
        _simple_spherical_norm(xtest, gamma=gamma, delta=delta),
        jax.grad(lambda x: lagr_spherical_norm(x, gamma=gamma, delta=delta))(xtest),
        rtol=1e-3,
    )
