# Model was trained/saved on a GPU: e3nn's TorchScript submodules ignore
# map_location, so force torch.jit.load onto the CPU before importing MACE.
import functools
import torch


## only for gpu to cpu 
_orig_jit_load = torch.jit.load
@functools.wraps(_orig_jit_load)
def _jit_load_cpu(f, map_location=None, *args, **kwargs):
    return _orig_jit_load(f, map_location="cpu", *args, **kwargs)

torch.jit.load = _jit_load_cpu

from mace.calculators import MACECalculator
from subprocess import check_output
from ase.io import read, write
import numpy as np
import glob
from ase.optimize import BFGS
import time




#################### Input #################### 
mace_models = ['cell_01-iter0.model']

targets = '*/*.in'
device = 'cuda'
###############################################

phases = ['formI', 'formII', 'formIII', 'formIII-o', 'formVI', 'formVII', ]
for mace_model in mace_models:
    label = mace_model[:-6]+'.dat'
    macepot = MACECalculator(model_paths=[mace_model], device=device)
    print(mace_model)
    for phase in phases:
        print(phase)
        fout = open(phase+'/'+label, 'w')
        for f in sorted(glob.glob(phase+ '/' + targets)):
            print(f)
            cell = read(f)
            cell.calc = macepot
            a = time.time()
            dyn = BFGS(cell, logfile=None, maxstep=250)
            dyn.run(fmax=0.05, steps=250)
            flag = '# conv' if dyn.converged() else '# not conv'
            e_rel = cell.get_potential_energy()
            v = cell.get_volume()
            b = time.time()
            fout.write(f"{v:16.12f}  {e_rel:16.12f} {b-a:10.4f}  {flag}\n")
            print(f"{v:16.12f}  {e_rel:16.12f}  {flag}\n ")
        fout.close()
