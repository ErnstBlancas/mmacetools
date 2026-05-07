from mace.calculators import MACECalculator
from ase.io import read, write
from glob import glob 
import numpy as np
import time
ang2bohr = 1.8897261
eV2ryd = 0.038893808

################################################################################
scells = '*/9-*/qha/supercell.in'
dft_to_calc = '9-eval/qha/harmonic*/geometry.in'
mace_model = 'models/mace-omat-0-small.model'
device = 'cpu'
dfset= "DFSET_HAR_ITER0"
################################################################################

def read_aims_in(fin):
    fin = open(fin, 'r')
    cell = []
    pos = []
    for line in fin:
        if "lattice_vector" in line:
            cell.append(np.array([np.float64(zz) for zz in line.split()[1:]]))
        elif "atom_frac" in line:
            pos.append(np.array([np.float64(zz) for zz in line.split()[1:4]]))
    fin.close()
    cell = np.array(cell)
    pos = np.array(pos)
    return cell, pos

def refold(x):
    if x >= 0.5:
        return x - 1.0
    elif x < -0.5:
        return x + 1.0
    else:
        return x

def calc_disp_forces(fname, macepot, x0, lavec0):
    cell = read(fname)
    x = cell.get_scaled_positions()
    disp = np.vectorize(refold)(x -x0)
    for i in range(disp.shape[0]):
        disp[i] = lavec0 @ disp[i] * ang2bohr
    cell.calc = macepot
    a = time.time()
    forces = cell.get_forces() * eV2ryd /ang2bohr
    b = time.time()
    return disp, forces, b-a

macepot = MACECalculator(model_paths=[mace_model], device=device)
path = "/".join(dft_to_calc.split('/')[0:2])
for scell in glob(scells):
    lavec0, x0 = read_aims_in(scell)
    ttime = 0
    cpath=scell.split('/')[0]
    fpath = cpath+'/'+path+'/'+dfset
    fout = open(fpath, 'w')
    for f in glob(cpath+'/'+dft_to_calc):
        print(f)
        print("Calculating forces on: ", f)
        disp, forces, etime = calc_disp_forces(f, macepot, x0, lavec0)
        print(f'# Filename: {f}, eval_time: {etime:10.4f} s',file=fout, flush=True)
        for i in range(disp.shape[0]):
            print(f'{disp[i][0]:15.10f} {disp[i][1]:15.10f} {disp[i][2]:15.10f} {forces[i][0]:15.10f} {forces[i][1]:15.10f} {forces[i][2]:15.10f}',file=fout, flush=True)
        ttime += etime

    print(f"# Total evaluation time: {ttime:10.4f} s", file=fout)
    fout.close()
print(f"# Total evaluation time: {ttime:10.4f} s")
