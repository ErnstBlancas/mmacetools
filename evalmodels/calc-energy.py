from mace.calculators import MACECalculator
from subprocess import check_output
from ase.io import read, write
import numpy as np
import glob
from ase.optimize import BFGS
import time




#################### Input #################### 
#mace_model = 'models/mace-omat-0-small.model'
#targets = 'al*/0-relax/*.out'
#device = 'cpu'
#fname = 'iter0.out'
#
mace_model = 'iter0/models/finetune.model.model'
targets = '*/fps*/*.out'
device = 'cpu'
fname = 'iter0.out'
###############################################

macepot = MACECalculator(model_paths=[mace_model], device=device)
fout = 'iter0.out'
fout = open(fout, 'w')
for f in glob.glob(targets):
    ffull = "/".join(f.split('/')[0:2])+'/'+ fname
    line = f"grep -ri 'Unit cell volume' {f} " + "| tail -1  | awk '{printf \"%.10f\", $6}'"
    v = np.float64(check_output(line, shell=True).decode('utf8'))
    cell = read(f)
    true_dft = cell.get_potential_energy()
    e_unrel = cell.get_potential_energy()
    cell.calc = macepot
    dyn = BFGS(cell)
    a = time.time()
    dyn.run(fmax=0.001)
    b = time.time()
    e_rel = cell.get_potential_energy()
    fout.write(f"{v:16.12f} {true_dft:16.12f} {cell.get_volume():16.12f} {e_unrel:16.12f} {e_rel:16.12f} {b-a:8.4f} # {ffull}")
    print(f"{v:16.12f} {true_dft:16.12f} {cell.get_volume():16.12f} {e_unrel:16.12f} {e_rel:16.12f} {b-a:8.4f}# {ffull}")
fout.close()
