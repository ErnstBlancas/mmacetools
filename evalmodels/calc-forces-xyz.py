from mace.calculators import MACECalculator
from ase.io import read
from glob import glob
import time

################################################################################
scells = '*/9-*/qha/supercell.in'
dft_to_calc = '9-eval/qha/harmonic*/geometry.in'
mace_model = 'models/mace-omat-0-small.model'
device = 'cpu'
xyzout = "mace_forces.xyz"
################################################################################

def calc_forces(fname, macepot):
    cell = read(fname)
    cell.calc = macepot
    a = time.time()
    energy = cell.get_potential_energy()
    forces = cell.get_forces()
    stress = cell.get_stress(voigt=False)
    b = time.time()
    return cell, energy, forces, stress, b-a

def write_xyz(fout, cell, energy, forces, stress, label, etime):
    lat = " ".join([f'{zz:12.8f}' for zz in cell.get_cell().flatten()])
    lstress = " ".join([f'{zz:12.8f}' for zz in stress.flatten()])
    print(len(cell), file=fout)
    print(f'Lattice="{lat}" Properties=species:S:1:pos:R:3:forces_mace:R:3 pbc="T T T" energy_mace={energy:12.8f} stress_mace="{lstress}" name="{label}-{time:.4f}"', file=fout)
    for atom, pos, force in zip(cell.get_chemical_symbols(), cell.get_positions(), forces):
        pp = " ".join([f'{zz:12.8f}' for zz in pos])
        ff = " ".join([f'{zz:12.8f}' for zz in force])
        print(f'{atom:4} {pp} {ff}', file=fout, flush=True)

macepot = MACECalculator(model_paths=[mace_model], device=device)
path = "/".join(dft_to_calc.split('/')[0:2])
for scell in glob(scells):
    ttime = 0
    cpath = scell.split('/')[0]
    fpath = cpath+'/'+path+'/'+xyzout
    fout = open(fpath, 'w')
    for f in glob(cpath+'/'+dft_to_calc):
        print("Calculating forces on: ", f)
        cell, energy, forces, stress, etime = calc_forces(f, macepot)
        write_xyz(fout, cell, energy, forces, stress, f, etime)
        ttime += etime
    fout.close()
    print(f"# Total evaluation time for {cpath}: {ttime:10.4f} s")
