from mace.calculators import MACECalculator
from ase.io import read
from glob import glob
import time

################################################################################
scells = 'formII/formII_*/cell_02_*in'
mace_model = 'models/mace-omat-0-small.model'
device = 'cpu'
xyzout = "mace_forces.xyz"
dtype = "float32" ## or float64
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
    print(f'Lattice="{lat}" Properties=species:S:1:pos:R:3:forces_mace:R:3 pbc="T T T" energy_mace={energy:12.8f} stress_mace="{lstress}" name="{label}-{etime:.4f}"', file=fout)
    for atom, pos, force in zip(cell.get_chemical_symbols(), cell.get_positions(), forces):
        pp = " ".join([f'{zz:12.8f}' for zz in pos])
        ff = " ".join([f'{zz:12.8f}' for zz in force])
        print(f'{atom:4} {pp} {ff}', file=fout, flush=True)

macepot = MACECalculator(model_paths=[mace_model], device=device, default_dtype=dtype)
fout = open(xyzout, 'w')
ttime = 0
for scell in glob(scells):
    print("Calculating forces on: ", scell)
    cell, energy, forces, stress, etime = calc_forces(scell, macepot)
    write_xyz(fout, cell, energy, forces, stress, scell, etime)
    ttime += etime

fout.close()
print(f"# Total evaluation time for {scells}: {ttime:10.4f} s")




