#!/home/ernesto/install/macefork/bin/python3
# Rebuild a MACE .model from a training checkpoint (.pt).
# The checkpoint only stores weights, so the model is built first, either:
#  - from the training options: the same ones given to mace_run_train
#    (--model, --config, ...), using MACE's own run_train. The training data
#    must be reachable, since avg_num_neighbors is computed from it.
#  - from a template model with the same architecture (--template); for
#    fine-tuning runs this is the foundation_model.
# Usage: python pt2model.py checkpoint.pt --model MACE --config config.yaml
#        python pt2model.py checkpoint.pt --template foundation.model
import sys
import os
import argparse
import tempfile
import functools
import torch

# Models saved on a GPU: e3nn's TorchScript submodules ignore map_location,
# so force torch.jit.load onto the CPU before importing MACE.
_orig_jit_load = torch.jit.load
@functools.wraps(_orig_jit_load)
def _jit_load_cpu(f, map_location=None, *args, **kwargs):
    return _orig_jit_load(f, map_location="cpu", *args, **kwargs)

torch.jit.load = _jit_load_cpu

import macefork.cli.run_train as run_train


class ModelBuilt(Exception):
    pass


def build_model(mace_args):
    """Run mace_run_train with mace_args and stop once the model is built."""
    original_configure_model = run_train.configure_model

    def configure_and_stop(*args, **kwargs):
        model, _ = original_configure_model(*args, **kwargs)
        raise ModelBuilt(model)

    run_train.configure_model = configure_and_stop
    with tempfile.TemporaryDirectory() as log_dir:
        sys.argv = ["mace_run_train", "--name", "pt2model"] + mace_args + [
            "--device", "cpu", "--log_dir", log_dir]
        try:
            run_train.main()
        except ModelBuilt as built:
            return built.args[0]
    sys.exit("mace_run_train finished without building a model")


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

if args.template:
    model = torch.load(args.template, map_location="cpu", weights_only=False)
else:
    model = build_model(mace_args)

state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
model.load_state_dict(state["model"])
torch.save(model, output)

print(f"Checkpoint: {args.checkpoint}")
if args.template:
    print(f"Template:   {args.template}")
else:
    print(f"Built from: {' '.join(mace_args)}")
print(f"Written:    {output}")
