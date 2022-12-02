# How to contribute

## Getting Started

HAMUX uses the [`nbdev`](https://github.com/fastai/nbdev) to build software. It comes with the following advantages:

- All major code is written, documented, and tested entirely within Jupyter Notebooks. This makes it easy to ensure these interdependent components of a library are in sync
- `nbdev` comes with nice out-of-the-box support for distributing software, e.g., publishing to `pip`/`conda` and serving your documentation site on github pages.


However, it also increases the overhead for potential contributors. After cloning and setting up the environment according to the README, do the following:

1. install the git hooks that run notebook cleaning and tests on every commit:
    ```
    nbdev_install_hooks
    ```
2. If your edits touch any `*.ipynb` file (and they should, since that's where the logic, documentation, and tests live), run the following command before making your commit:
    ```
    nbdev_prepare
    ```
3. After pushing, check that all tests pass on your fork of the github before submitting a PR.


## Did you find a bug?

Submit an issue! We can solve it together and improve documentation+tests at the same time. Please include a description on how you expected the code to behave and error messages, if relevant.


#### Did you write a patch that fixes a bug?

* Open a new GitHub pull request with the patch.
* Ensure that your PR includes a test that fails without your patch, and pass with it.

## On Style

HAMUX is a rapidly evolving library -- there is no other deep learning library that operates like it. As such, the API is highly subject to change and a consistent coding style is not particularly important.

That said, we do follow `nbdev` [best practices](https://www.youtube.com/watch?v=67FdzLSt4aA). For us coders, it requires shifting our thinking from building insular, minimally documented python modules to thinking in terms of [Literate Programming](https://www-cs-faculty.stanford.edu/~knuth/lp.html) where the functional code is interspersed with explanation and tests for expected behavior. 

Submit a PR and we can work out the functional style together.

## Roadmap

- JAX + Treex is a powerful and intuitive combo... for me; it is not a combination that many DL researchers use. Thus we need to port HAMUX to, at the minimum, [`pytorch`](https://pytorch.org/) and [`flax`](https://github.com/google/flax). These are long term goals, and help on them would be appreciated.
- HAMUX right now is a very powerful theoretical abstraction for memory. We demonstrate equivalent performance to the single layer HN. However, we don't have the architecture+training paradigm to prove that its hierarchical form is substantially better. We need to build better architectures that solve real tasks.

