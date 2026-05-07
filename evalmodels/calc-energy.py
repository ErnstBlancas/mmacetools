from mace.calculators import MACECalculator
from subprocess import check_output
from ase.io import read, write
import numpy as np
import glob


#################### Input #################### 
mace_model = 'models/mace-omat-0-small.model'
targets = 'al*/0-relax/*.out'
device = 'cpu'
fname = 'iter0.out'
###############################################

macepot = MACECalculator(model_paths=[mace_model], device=device)
for f in glob.glob(targets):
    fout= "/".join(f.split('/')[0:2])+'/'+ fname
    line = f"grep -ri 'Unit cell volume' {f} " + "| tail -1  | awk '{printf \"%.10f\", $6}'"
    v = np.float64(check_output(line, shell=True).decode('utf8'))
    cell = read(f)
    cell.calc = macepot
    e = cell.get_potential_energy()
    fout = open(fout, 'w')
    print(f"{v:12.8f} {e:12.8f}", file=fout)
    fout.close()
