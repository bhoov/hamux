"""Training script for classification on MNIST

We recreate the famous 2-layer Hopfield Network in our framework and compute best performance on MNIST classification.

To speed up training, MNIST data is loaded with select (and simplified) augmentation strategies from the popular timm library.

```
python classification.py vectorized_chn_cifar ./_log_chn_cifar --device 1 --dataset cifar10
python classification.py vectorized_chn_mnist ./_log_tchn_mnist --device 1 --dataset mnist
```
"""

## SETUP
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "model", type=str, help="Which model to load from default model registry"
)
parser.add_argument(
    "--outdir",
    type=str,
    default=None,
    help="Where to saved checkpoints and model information. If not provided use the current working directory",
)
parser.add_argument(
    "--dataset", type=str, default="MNIST", help="One of [MNIST, CIFAR10]."
)
parser.add_argument(
    "--device", type=int, default=0, help="GPU device on which to conduct the training"
)
parser.add_argument(
    "--num_epochs",
    type=int,
    default=1000,
    help="The number of epochs on which to run the training",
)
parser.add_argument("--batch_size", type=int, default=400, help="Batch Size")
parser.add_argument("--lr", type=float, default=0.001, help="Default learning rate")
parser.add_argument(
    "--seed",
    type=int,
    default=34,
    help="Seed the randomness of training and model initialization",
)
parser.add_argument(
    "--filter_betas",
    action="store_true",
    help="Do not train betas in the softmax layers of the model",
)
parser.add_argument(
    "--pct_dev_memory",
    type=float,
    default=None,
    help="How much available memory to preallocate to jax on provided device",
)
parser.add_argument(
    "--normal_init",
    action="store_true",
    help="If true, initialize all layer states as normal distributions rather than all 0s",
)
args = parser.parse_args()

assert args.dataset.lower() in set(["mnist", "cifar10"])
import os

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
os.environ[
    "XLA_FLAGS"
] = "--xla_gpu_force_compilation_parallelism=1"  # This is necessary for our environment's JAX installation
if args.pct_dev_memory is not None:
    assert args.pct_dev_memory <= 1 and args.pct_dev_memory > 0.0
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(args.pct_dev_memory)

from loguru import logger
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import torch
import hamux as hmx
import treex as tx
from flax import linen as nn  # For initializers
import optax
import jax.tree_util as jtu
from typing import *
from hamux.datasets import *
from tqdm import trange, tqdm
from dataclasses import dataclass
from hamux.utils import pytree_save, pytree_load, to_pickleable
import json
import sys


jax_key = jax.random.PRNGKey(args.seed)
torch.manual_seed(args.seed + 1)
np.random.seed(args.seed + 2)

logdir = Path().absolute() if args.outdir is None else Path(args.outdir)
logdir.mkdir(parents=True, exist_ok=True)
logger.add(str(logdir / "stdlogs.log"), colorize=False)
logger.add(sys.stdout)

# ===========================================
## Training code
# ===========================================


class TrainState(tx.Module):
    model: tx.Module
    optimizer: tx.Optimizer
    apply_fn: Callable
    filter_betas: bool
    rng: jnp.ndarray = tx.Rng.node()
    eval_rng: jnp.ndarray = tx.Rng.node()

    def __init__(
        self, model, optimizer, apply_fn, rng, filter_betas=False, do_normal_init=False
    ):
        self.filter_betas = filter_betas
        self.model = model
        self.optimizer = tx.Optimizer(optimizer).init(self.params)
        self.apply_fn = apply_fn
        self.rng, self.eval_rng = jax.random.split(rng)
        self.do_normal_init = do_normal_init

    @property
    def params(self):
        if self.filter_betas:
            return self.model.filter(lambda x: "beta" not in x.name)
        return self.model.filter(tx.Parameter)

    def apply_updates(self, grads):
        new_params = self.optimizer.update(grads, self.params)
        self.model = self.model.merge(new_params)
        return self


def cross_entropy_loss(*, probs, labels):
    n_classes = probs.shape[-1]
    labels_onehot = jax.nn.one_hot(labels, num_classes=n_classes)
    smoothed_labels = (0.1 / n_classes + labels_onehot)
    smoothed_labels = smoothed_labels / jnp.abs(smoothed_labels).sum(-1, keepdims=True)

    stable_probs = (probs + 1e-6) / (1+(1e-6)*n_classes)
    loss = -jnp.sum(smoothed_labels * jnp.log(stable_probs), axis=-1).mean()
    return loss


def compute_metrics(*, probs, labels):
    loss = cross_entropy_loss(probs=probs, labels=labels)
    accuracy = jnp.mean(jnp.argmax(probs, -1) == labels)
    metrics = {
      "probs_min": probs.min(),
      'probs_max': probs.max(),
      'loss': loss,
      'accuracy': accuracy,
    }
    return metrics


@jax.jit
def train_step(state, batch):
    if state.do_normal_init:
        rng, state.rng = jax.random.split(state.rng)
    else:
        rng = None

    def loss_fn(params):
        state.model = state.model.merge(params)
        x = batch["image"]
        probs = state.apply_fn(state.model, x, rng=rng)
        loss = cross_entropy_loss(probs=probs, labels=batch["label"])
        return loss, (probs, state)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, (probs, state)), grads = grad_fn(state.params)

    state = state.apply_updates(grads)
    metrics = compute_metrics(probs=probs, labels=batch["label"])
    return state, metrics

@jax.jit
def eval_step(state, batch):
    x = batch["image"]
    if state.do_normal_init:
        rng = state.eval_rng
    else:
        rng = None
    probs = state.apply_fn(state.model, x, rng=rng)
    return compute_metrics(probs=probs, labels=batch['label'])

def train_epoch(state, train_dl, epoch):
    """Train for a single epoch."""
    batch_metrics = []
    bs = train_dl.batch_size
    for i, batch in enumerate(tqdm(train_dl, leave=False)):
        batch = {"image": jnp.array(batch[0]), "label": jnp.array(batch[1])}
        state, metrics = train_step(state, batch)
        batch_metrics.append(metrics)

    # compute mean of metrics across each batch in epoch.
    batch_metrics_np = jax.device_get(batch_metrics)
    epoch_metrics_np = {
        k: np.mean([metrics[k] for metrics in batch_metrics_np])
        for k in batch_metrics_np[0]
    }
    return state, epoch_metrics_np["loss"], epoch_metrics_np["accuracy"]


def eval_model(state, test_dl):
    batch_metrics = []

    for i, batch in enumerate(test_dl):
        batch = {"image": jnp.array(batch[0]), "label": jnp.array(batch[1])}

        metrics = eval_step(state, batch)
        batch_metrics.append(metrics)
    batch_metrics_np = jax.device_get(batch_metrics)
    summary = {
        k: np.mean([metrics[k] for metrics in batch_metrics_np])
        for k in batch_metrics_np[0]
    }

    return summary["loss"], summary["accuracy"]


# ===========================================
## Dataloaders
# ===========================================
if args.dataset.lower() == "mnist":
    dl_args = DataloadingArgs(
        dataset="torch/MNIST",
        aa=None,
        reprob=0.1,
        vflip=0.0,
        hflip=0.0,
        scale=(0.9, 1.0),
        batch_size=args.batch_size,
        color_jitter=0.4,
        validation_batch_size=2 * args.batch_size,
    )
    data_config = DataConfigMNIST(input_size=(1, 28, 28))
elif args.dataset.lower() == "cifar10":
    dl_args = DataloadingArgs(
        dataset="torch/CIFAR10",
        # aa="rand",
        aa=None,
        reprob=0.2,
        vflip=0.0,
        hflip=0.5,
        scale=(0.2, 1.0),
        batch_size=args.batch_size,
        color_jitter=0.5,
        validation_batch_size=2 * args.batch_size,
    )
    data_config = DataConfigCIFAR10(input_size=(3, 32, 32))

train_dl, eval_dl = create_dataloaders(dl_args, data_config)

# ===========================================
## Model Configuration
# ===========================================
k1, jax_key = jax.random.split(jax_key)
ham, forward_classification = hmx.create_model(args.model)
states, ham = ham.init_states_and_params(k1, bs=1)

optimizer = optax.adamw(args.lr)
state = TrainState(
    ham,
    optimizer,
    forward_classification,
    rng=jax_key,
    filter_betas=args.filter_betas,
    do_normal_init=args.normal_init,
)


def get_nparams(model):
    params, meta = jtu.tree_flatten(model.parameters())

    def get_nel(x):
        try:
            return x.size
        except AttributeError:  # float
            return 1

    return sum([get_nel(p) for p in params])


def escape_ansi(line):
    import re

    ansi_escape = re.compile(r"(?:\x1B[@-_]|[\x80-\x9F])[0-?]*[ -/]*[@-~]")
    return ansi_escape.sub("", line)


logger.info(f"NParams={get_nparams(state.model)}")
logger.info(escape_ansi(state.model.tabulate()))

# ===========================================
## Training
# ===========================================
@dataclass
class CkptTracker:
    base_name: str
    model: tx.Module = None
    epoch: int = -1
    best_acc: float = -1

    def get_save_name(self):
        return f"{self.base_name}_epoch-{self.epoch}_acc-{100*self.best_acc:.3f}.pckl"


train_acc_list = []
test_acc_list = []
ckpt_tracker = CkptTracker(args.model)


with trange(1, args.num_epochs + 1, unit="epochs") as pbar:
    for epoch in pbar:
        state, train_loss, train_acc = train_epoch(state, train_dl, epoch)
        test_loss, test_acc = eval_model(state, eval_dl)
        if test_acc > ckpt_tracker.best_acc:
            old_ckpt_name = str(logdir / ckpt_tracker.get_save_name())
            try:
                os.remove(old_ckpt_name)
            except Exception as e:
                logger.debug(f"Couldn't remove {old_ckpt_name}, {e}")
                pass
            ckpt_tracker.model = state.model
            ckpt_tracker.epoch = epoch
            ckpt_tracker.best_acc = test_acc
            to_save = jtu.tree_map(to_pickleable, ckpt_tracker.model.to_dict())
            ckpt_name = str(logdir / ckpt_tracker.get_save_name())
            pytree_save(to_save, ckpt_name, overwrite=True)
        desc = f"[{epoch}/{args.num_epochs}] | BestAcc: {100*ckpt_tracker.best_acc:.3f} | TrainLoss: {train_loss:.2f} | TrainAcc: {100*train_acc:.2f}"
        addition = f"curr_val_acc={100*test_acc:0.2f}"
        logger.info(desc + " | " + addition)
        pbar.set_description(desc)
        pbar.set_postfix(
            train_acc=f"{100*train_acc:0.2f}", val_acc=f"{100*test_acc:0.2f}"
        )