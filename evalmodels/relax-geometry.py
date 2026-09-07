from mace.calculators import MACECalculator
from ase.constraints import FixSymmetry
from ase.filters import FrechetCellFilter
from ase.optimize import BFGS, FIRE, LBFGS
from ase.io import read, write
from ase.io.formats import filetype
import time

"""Relax a crystal geometry (atomic positions and, optionally, the cell)
using a MACE machine learning potential through ASE's optimizers.
"""
#################### Input ####################
geom = 'geometry.in'
model = 'models/mace-omat-0-small.model'
device = 'cpu'
relax_cell = True         # also relax the lattice vectors, not just positions
fix_symmetry = True        # constrain the relaxation to the initial spacegroup
hydrostatic_strain = False # if relax_cell: keep the cell shape, only allow isotropic volume change
fmax = 0.001               # eV/ang convergence criterion (max force, or fmax on the cell filter)
steps = 500                # max number of optimizer steps
optimizer = 'BFGS'         # FIRE, BFGS or LBFGS
trajectory = 'relax.traj'
logfile = 'relax.log'
output = 'geometry.in.relaxed'
###############################################

optimizers = {'FIRE': FIRE, 'BFGS': BFGS, 'LBFGS': LBFGS}

def relax(atoms, macepot, relax_cell, fix_symmetry, hydrostatic_strain,
          fmax, steps, optimizer, trajectory, logfile):
    atoms.calc = macepot
    if fix_symmetry:
        atoms.set_constraint(FixSymmetry(atoms))
    to_opt = atoms
    if relax_cell:
        to_opt = FrechetCellFilter(atoms, hydrostatic_strain=hydrostatic_strain)
    dyn = optimizers[optimizer](to_opt, trajectory=trajectory, logfile=logfile)
    a = time.time()
    dyn.run(fmax=fmax, steps=steps)
    b = time.time()
    return atoms, dyn.converged(), b - a

macepot = MACECalculator(model_paths=[model], device=device)
fmt = filetype(geom, read=True)
atoms = read(geom, format=fmt)
atoms, converged, etime = relax(atoms, macepot, relax_cell, fix_symmetry,
                                 hydrostatic_strain, fmax, steps, optimizer,
                                 trajectory, logfile)
if fix_symmetry:
    atoms.set_constraint(None)

e = atoms.get_potential_energy()
write(output, atoms, format=fmt)  # keep the input geometry's format, whatever the output filename
print(f"Relaxation {'converged' if converged else 'did NOT converge'} in {etime:10.4f} s")
print(f"Final energy: {e:12.8f} eV, written to {output}")
