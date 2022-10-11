Introduction to HAMUX
================

<!-- WARNING: THIS FILE WAS AUTOGENERATED! DO NOT EDIT! -->

With HAMUX, we can build **arbitrarily deep** networks that obey the
energy rules of [Hopfield
Networks](https://en.wikipedia.org/wiki/Hopfield_network). That is,
*every* system built using HAMUX is a *dynamical system* guaranteed to
have a tractable energy function that converges to a fixed point. Our
deep Hierarchical Associative Memories (HAMs) have several additional
advantages over traditional Hopfield Networks (HNs):

| Hopfield Networks                                                                                         | HAMUX                                                                                                                                                                                                       |
|-----------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| HNs connect one visible layer to one hidden layer                                                         | HAMs can connect **arbitrary numbers** of layers, as deep as you want                                                                                                                                       |
| HNs exclusively model layer relationships (*synapses*) as dense matrix multiplications                    | HAMs can be composed of **arbitrary operations**, e.g., convolutions, pooling, attention, $\ldots$                                                                                                          |
| HNs are typically shallow, consisting of a single visible neuron layer connected to a single hidden layer | HAMs can be **deep**! E.g., Perform convolutions and pooling on an image layer, connect these patches to a sequence of text tokens via energy-attention, throw in labels that also evolve in time, $\ldots$ |

Additionally, HNs are typically implemented using the computation graph
of their update step, e.g., an energy version of attention (Krotov and
Hopfield 2021) (placeholder citation):

$$-\frac{\partial E}{\partial g_{iA}} = \sum \limits_{C \neq A} \sum\limits_{\alpha}  W^Q_{\alpha i}\; K_{\alpha C} \; \underset{C}{\text{softmax}}\Big( \beta \sum\limits_\gamma K_{\gamma C} \; Q_{\gamma A}\Big) + W^K_{\alpha i} \; Q_{\alpha C}\; \underset{A}{\text{softmax}}\Big( \beta \sum\limits_\gamma K_{\gamma A} \; Q_{\gamma C}\Big)$$

where

Our HAMs are alternatively described by their energy function and JAX’s
autograd computes the computation graph. Instead of the complex equation
above, we only need to define:

$$  E = -\frac{1}{\beta}\sum\limits_h\sum\limits_C \textrm{log} \left(\sum\limits_{B \neq C} \textrm{exp}\left(\beta \sum\limits_{\alpha} K_{ \alpha h B} \; Q_{\alpha h C}\right) \right)\label{energy attention}
 $$

HAMs are a generalization of HNs and can be thought of as “Hierarchical
Hopfield Networks”. Every layer and synapse is defined by the layer and
synapse energies of the original Hopfield Network. The contributions
HAMUX, then, is primarily in uniting two seemingly incompatible
approaches to AI: the deep computational circuit approach that is modern
Deep Learning, and the shallow energy-based differential equation
approach that is the traditional HN. HAMUX brings much of the language
and capabilities of modern Deep Learning to the energy-based regime of
Hopfield Networks and associative memories.

This library is part proof of concept, part functional framework for
building deep, energy-based associative memories.

## Install

    pip install hamux

If you are using accelerators beyond the CPU you will need to install
the corresponding `jax` and `jaxlib` versions following [their
documentation](https://github.com/google/jax#installation). E.g.,

    pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

You can install the requirements for datasets with

    pip install -r requirements-dev.txt

## How to Use

### Building a HAM

Every layer in a HAM can be thought of as an activation that tracks
state through time. HAMUX calculates how these states evolve over time
given the other states and the values of the learned parameters. Below
is an example of assembling a HAM using the building blocks provided by
HAMUX, assuming MNIST as input data.

``` python
import hamux as hmx
import jax.numpy as jnp
import jax
```

``` python
layers = [
    hmx.TanhLayer((32,32,3)), # CIFAR Images
    hmx.SigmoidLayer((11,11,1000)), # CIFAR patches
    hmx.SoftmaxLayer((10,)), # CIFAR Labels
    hmx.ReluLayer((1000,)), # Hidden Memory Layer
]

synapses = [
    hmx.ConvSynapse((3,3), strides=3),
    hmx.DenseSynapse(),
    hmx.DenseSynapse(),
]

connections = [
    ([0,1], 0),
    ([1,3], 1),
    ([2,3], 2),
]

states, ham = hmx.HAM(layers, synapses, connections).init_states_and_params(jax.random.PRNGKey(0));
```

Notice that we did not specify any output channel shapes in the
synapses. The desired output shape is computed from the layers connected
each synapse during `hmx.HAM.init_states_and_params`.

We had to know the shape of the convolution outputs to track them over
time. We provide helper functions in the `hmx.ConvSynapse` to calculate
the shapes below:

``` python
syn = hmx.ConvSynapse((3,3), strides=3)
g1 = jnp.ones((32,32,3))
print(syn.example_output(g1).shape)
```

    (11, 11, 1)

### HAMs are Energy Based

Read [our documentation](https://bhoov.github.io/hamux) for exploration
of the energy components of our system

### Training HAMs

See the examples in `/examples`.

<div id="refs" class="references csl-bib-body hanging-indent">

<div id="ref-krotov2021large" class="csl-entry">

Krotov, Dmitry, and John J. Hopfield. 2021. “Large Associative Memory
Problem in Neurobiology and Machine Learning.” In *International
Conference on Learning Representations*.
<https://openreview.net/forum?id=X4y_10OX-hX>.

</div>

</div>
