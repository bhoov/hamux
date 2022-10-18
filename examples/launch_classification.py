from pathlib import Path
import subprocess
import multiprocessing as mp
from fastcore.script import *
from typing import *


gpus = [1,2,3,4,5,6]

pct_mem_per_gpu = 0.6
test_only = True
max_epochs = 600
logdir = Path().absolute() / "_dummylogs" / "log01"
do_normal_init = False
logdir.mkdir(parents=True, exist_ok=True)

mnames_flags = [
    ("hn_relu_mnist", "--dataset mnist"),
    ("hn_repu5_mnist", "--dataset mnist"),
    ("hn_softmax_mnist", "--dataset mnist"),

    ("conv_ham_avgpool_mnist", "--dataset mnist"),
    ("conv_ham_maxpool_mnist", "--dataset mnist"),
    ("energy_attn_mnist", "--dataset mnist"),

    ("hn_relu_cifar", "--dataset cifar10"),
    ("hn_repu5_cifar", "--dataset cifar10"),
    ("hn_softmax_cifar", "--dataset cifar10"),
    ("conv_ham_avgpool_cifar", "--dataset cifar10"),
    ("conv_ham_maxpool_cifar", "--dataset cifar10"),
    ("energy_attn_cifar", "--dataset cifar10"),
]

assert len(mnames_flags) >= len(gpus)

grouped_runs = [[] for _ in gpus]
for i, info in enumerate(mnames_flags):
    gpui = i % len(gpus)
    gpu = gpus[gpui]
    grouped_runs[gpui].append(info + (gpu,))

def submit_run(mname, flags, device):
    outdir = logdir / mname
    #  outdir.mkdir(parents=True, exist_ok=False)
    cmd = f"python classification.py {mname} {flags} --outdir {outdir} --device {device}"
    if test_only:
        cmd += " --num_epochs=2"
    else:
        cmd += f" --num_epochs={max_epochs}"

    if do_normal_init:
        cmd += " --normal_init"
    print(cmd)
    subprocess.run(cmd, shell=True)

def submit_runs(runs):
    """Where `runs` is a list of (mname, flags). Submit all commands to run sequentially on specified device"""
    for mname, flags, device in runs:
        submit_run(mname, flags, device)


with mp.Pool(len(gpus)) as p:
    p.map(submit_runs, grouped_runs)
