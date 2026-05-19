from mace.calculators import MACECalculator
from ase.io import read, write
from glob import glob 
import numpy as np
import time
ang2bohr = 1.8897261
eV2ryd = 0.038893808

################################################################################
scells = '*/9-*/supercell.in'
dft_to_calc = '9-eval/harmonic*/geometry.in'
mace_model = 'finetune/iter2/1-supercell-2x/best_forces/models/finetune.model.model'
device = 'cpu'
dfset= "9-eval/qha-2x-prim/DFSET_HAR_ITER2"
## 
################################################################################

def get_LR(born,cell, pos, symbols, scell_matrix):
    from phonopy.structure.atoms import PhonopyAtoms
    from phonopy.file_IO import parse_BORN
    from phonopy import Phonopy
    if len(scell_matrix) == 3:
        scell_matrix = np.diag(scell_matrix)
    elif len(scell_matrix) == 9:
        scell_matrix = np.array(scell_matrix).reshape((3,3))
    else:
        print('Something is wrong with the supercell_matrix')
        exit()
    ucell = PhonopyAtoms(symbols=symbols, scaled_positions=pos, cell=cell)
    phonon = Phonopy(ucell, scell_matrix, primitive_matrix='auto')
    phonon.force_constants = np.zeros((len(pos), len(pos), 3,3))
    phonon.nac_params = parse_BORN(phonon.primitive, filename=born) 
    dynmat = phonon.dynamical_matrix
    dynmat.make_Gonze_nac_dataset()
    fc2_LR = -dynmat.Gonze_nac_dataset[0]
    return fc2_LR

def read_aims_in(fin):
    fin = open(fin, 'r')
    cell = []
    pos = []
    symbols = []
    for line in fin:
        if "lattice_vector" in line:
            cell.append(np.array([np.float64(zz) for zz in line.split()[1:]]))
        elif "atom_frac" in line:
            pos.append(np.array([np.float64(zz) for zz in line.split()[1:4]]))
            symbols.append(line.split()[4])
    fin.close()
    cell = np.array(cell)
    pos = np.array(pos)
    return cell, pos, symbols

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
    e = cell.get_potential_energy()
    return disp, forces, e, b-a

macepot = MACECalculator(model_paths=[mace_model], device=device)
path = "/".join(dft_to_calc.split('/')[0:2])

#borns = glob(born)

cc = 0
for scell in glob(scells):
    lavec0, x0, symbols = read_aims_in(scell)
#    if use_born:
#        fc2_lr = get_LR(borns[cc],lavec0, x0, symbols, scell_matrix)
    ttime = 0
    cpath=scell.split('/')[0]
    print(cpath)
    fpath = cpath+'/'+dfset
    print(fpath)
    fout = open(fpath, 'w')
    for f in glob(cpath+'/'+dft_to_calc):
        print("Calculating forces on: ", f)
        disp, forces, e, etime = calc_disp_forces(f, macepot, x0, lavec0)
        print(f'# Filename: {f}, eval_time: {etime:10.4f} s, energy = {e:10.8f} eV',file=fout, flush=True)
        for i in range(disp.shape[0]):
            print(f'{disp[i][0]:15.10f} {disp[i][1]:15.10f} {disp[i][2]:15.10f} {forces[i][0]:15.10f} {forces[i][1]:15.10f} {forces[i][2]:15.10f}',file=fout, flush=True)
        ttime += etime
        print(etime)
    print(f"# Total evaluation time: {ttime:10.4f} s", file=fout)
    fout.close()
##     cc += 1
print(f"# Total evaluation time: {ttime:10.4f} s")
## 
