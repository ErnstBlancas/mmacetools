#!/home/ernesto/install/macefork/bin/python3
import sys
import functools
import torch

# Models saved on a GPU: e3nn's TorchScript submodules ignore map_location,
# so force torch.jit.load onto the CPU before importing MACE.
_orig_jit_load = torch.jit.load
@functools.wraps(_orig_jit_load)
def _jit_load_cpu(f, map_location=None, *args, **kwargs):
    return _orig_jit_load(f, map_location="cpu", *args, **kwargs)

torch.jit.load = _jit_load_cpu

from ase.data import chemical_symbols
from macefork.tools.scripts_utils import extract_config_mace_model

if len(sys.argv) > 1:
    mace_model = sys.argv[1]
else:
    mace_model = input("Model file: ").strip()

model = torch.load(mace_model, map_location="cpu", weights_only=False)
config = extract_config_mace_model(model)

n_params = sum(p.numel() for p in model.parameters())
n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
elements = [chemical_symbols[z] for z in config["atomic_numbers"].tolist()]

print(f"Model:               {mace_model}")
print(f"Model type:          {type(model).__name__}")
print(f"Precision:           {next(model.parameters()).dtype}")
print(f"Parameters:          {n_params} ({n_trainable} trainable)")
print(f"Heads:               {config['heads']}")
print(f"Elements:            {len(elements)} ({' '.join(elements)})")
print()
print(f"Cutoff r_max:        {config['r_max']} Ang")
print(f"Interaction layers:  {config['num_interactions']}")
print(f"First interaction:   {config['interaction_cls_first'].__name__}")
print(f"Interaction:         {config['interaction_cls'].__name__}")
print(f"Hidden irreps:       {config['hidden_irreps']}")
print(f"Max L (sph. harm.):  {config['max_ell']}")
print(f"Correlation order:   {config['correlation']}")
print(f"Radial basis:        {config['radial_type']} ({config['num_bessel']} functions)")
print(f"Polynomial cutoff p: {config['num_polynomial_cutoff']}")
print(f"Distance transform:  {config['distance_transform']}")
print(f"Radial MLP:          {config['radial_MLP']}")
print(f"Readout MLP irreps:  {config['MLP_irreps']}")
print(f"Gate:                {config['gate'].__name__}")
print(f"Pair repulsion:      {config['pair_repulsion']}")
print(f"Avg num neighbors:   {config['avg_num_neighbors']:.4f}")
print(f"Scale / shift:       {config['atomic_inter_scale']} / {config['atomic_inter_shift']}")
print()
print("Parameters per block:")
for name, block in model.named_children():
    n = sum(p.numel() for p in block.parameters())
    if n > 0:
        print(f"  {name:22s} {n}")
