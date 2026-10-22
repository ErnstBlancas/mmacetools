#!/home/ernesto/install/macefork/bin/python3
# Rebuild a MACE .model from a training checkpoint (.pt).
# The checkpoint only stores weights, so the model is built first, either:
#  - from the training options: the same ones given to mace_run_train
#    (--model, --config, ...). The training data is not read: the values
#    computed from it (elements, E0s, scale/shift, avg_num_neighbors) are
#    stored in the checkpoint. Checkpoints written before avg_num_neighbors
#    was stored need --avg_num_neighbors (printed in the training log).
#  - from a template model with the same architecture (--template); for
#    fine-tuning runs this is the foundation_model.
# Checkpoints from runs with enable_cueq need cuequivariance installed.
# Usage: python pt2model.py checkpoint.pt --model MACE --config config.yaml
#        python pt2model.py checkpoint.pt --template foundation.model
import sys
import os
import ast
import argparse
import functools
import torch

# Models saved on a GPU: e3nn's TorchScript submodules ignore map_location,
# so force torch.jit.load onto the CPU before importing MACE.
_orig_jit_load = torch.jit.load
@functools.wraps(_orig_jit_load)
def _jit_load_cpu(f, map_location=None, *args, **kwargs):
    return _orig_jit_load(f, map_location="cpu", *args, **kwargs)

torch.jit.load = _jit_load_cpu

from macefork import tools
from macefork.tools.model_script_utils import configure_model
from macefork.cli.convert_e3nn_cueq import run as run_e3nn_to_cueq, CUEQQ_AVAILABLE
from macefork.cli.convert_cueq_e3nn import run as run_cueq_to_e3nn


def build_model(mace_args, state):
    """Build the model from mace_run_train options and the checkpoint state."""
    parser = tools.build_default_arg_parser()
    args = parser.parse_args(["--name", "pt2model"] + mace_args)
    args, _ = tools.check_args(args)
    tools.set_default_dtype(args.default_dtype)

    stored = any(key.endswith("avg_num_neighbors") for key in state)
    if not stored and args.avg_num_neighbors == parser.get_default("avg_num_neighbors"):
        sys.exit("This checkpoint does not store avg_num_neighbors: pass "
                 "--avg_num_neighbors (from the training log)")

    # Values that run_train computes from the training data
    heads = list(ast.literal_eval(args.heads)) if args.heads else ["Default"]
    z_table = tools.AtomicNumberTable(state["atomic_numbers"].tolist())
    atomic_energies = state["atomic_energies_fn.atomic_energies"].numpy()
    args.mean = state["scale_shift.shift"].numpy()
    args.std = state["scale_shift.scale"].numpy()
    # Set by run_train; configure_model only logs them
    args.compute_energy = True
    args.compute_dipole = False

    model, _ = configure_model(args, None, atomic_energies, heads=heads, z_table=z_table)
    return model


parser = argparse.ArgumentParser(
    description="Rebuild a MACE .model from a checkpoint (.pt). Options not "
                "listed here are passed to mace_run_train (--model, --config, ...).")
parser.add_argument("checkpoint", help="checkpoint file (.pt)")
parser.add_argument("--template", help="model with the same architecture")
parser.add_argument("--output", help="output model (default: checkpoint name with .model)")
args, mace_args = parser.parse_known_args()

output = args.output
if output is None:
    output = os.path.basename(args.checkpoint).replace(".pt", ".model")

state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["model"]
# Runs with enable_cueq save the weights in the cuequivariance layout
cueq = "products.0.symmetric_contractions.weight" in state
if cueq and not CUEQQ_AVAILABLE:
    sys.exit("This checkpoint is from a run with enable_cueq: "
             "cuequivariance must be installed to convert it")

if args.template:
    model = torch.load(args.template, map_location="cpu", weights_only=False)
else:
    model = build_model(mace_args, state)

if cueq:
    model = run_e3nn_to_cueq(model)
    # Without the CUDA kernels, cuequivariance stores constant buffers
    # (graph.c*) that are not in the checkpoint; they come with the model.
    result = model.load_state_dict(state, strict=False)
    missing = [key for key in result.missing_keys if ".graph.c" not in key]
    if missing or result.unexpected_keys:
        sys.exit(f"Checkpoint does not match the model.\nMissing: {missing}\n"
                 f"Unexpected: {result.unexpected_keys}")
    model = run_cueq_to_e3nn(model)
else:
    model.load_state_dict(state)
torch.save(model, output)

print(f"Checkpoint: {args.checkpoint}")
if args.template:
    print(f"Template:   {args.template}")
else:
    print(f"Built from: {' '.join(mace_args)}")
print(f"Written:    {output}")
