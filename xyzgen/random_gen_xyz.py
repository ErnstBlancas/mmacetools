import glob
import numpy as np
import matplotlib.pyplot as plt
from ase.build import bulk
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
from ase.io import write, read
import time

def write_isolated(fout, atom, label=None):
    first_line = (f'Properties=species:S:1:pos:R:3:forces_dft:R:3 pbc="F F F" '
                  f'name="{label}" config_type=IsolatedAtom')
    print("1", file=fout)
    print(first_line, file=fout)
    print(f"{atom} 0.0 0.0 0.0 0.0 0.0 0.0", file=fout)

## just to check teacher/student
################################################################################
temperatures = [10, 50, 100, 200, 300]  # K
nstructures = 30
fins = 'formII/formII_00/cell_02.in'
fout = 'mace_batch.xyz'
# Displacement = velocity * dt. Taking dt ~ 1/omega (Einstein frequency)
dt = units.fs * 2.5 ## smaller as possible
rng = np.random.default_rng(int(time.time()))
unique = ['H', 'C', 'N', 'O']
################################################################################


displacements = {}
fout = open(fout, 'w')
## first write unique atoms for the student
for atom in unique:
    write_isolated(fout, atom, label='uniqueatoms')

for fin in glob.glob(fins):
    print(fin)
    atoms = read(fin)
    nat = len(atoms)
    name = fin.split('/')[-1][:-3]
    path = './'+ "/".join(fin.split('/')[0:-1])
    lat = " ".join([f'{zz:12.8f}' for zz in atoms.cell.flatten()])
    for temperature in temperatures:
        displacements[temperature] = []
        for i in range(nstructures):
            displaced = atoms.copy()
            MaxwellBoltzmannDistribution(displaced, temperature_K=temperature, rng=rng, force_temp=True)
            displaced.positions += displaced.get_velocities() * dt
            symbols = displaced.symbols
            #write(f'{path}/{name}_{i:03d}_{temperature:03d}.in', displaced)
            u = np.linalg.norm(displaced.positions - atoms.positions, axis=1)
            displacements[temperature].extend(u)
            fout.write(f'{nat}\n')
            fout.write(f'Lattice=\"{lat}\" Properties=species:S:1:pos:R:3 pbc=\"T T T\" name=\"{temperature:02d}_{i:03d}_{fin}\"\n')
            for j in range(len(displaced)):
                pos = " ".join([f'{zz:12.8f}' for zz in displaced.positions[j]])
                fout.write(f'{symbols[j]} {pos}\n') 
fout.close()

fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8))
for temperature in temperatures:
    ax.hist(displacements[temperature], bins=50,label=f'T = {temperature} K')
    ax.set_xlabel('Displacement (Å)')
ax.set_ylabel('Count')
ax.legend()
fig.tight_layout()
fig.savefig('displacements.png')
