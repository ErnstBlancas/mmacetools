"""Benchmark run-md.py's get_dyn()/runMD() on MgO across all its ensembles.

Builds a small MgO (rocksalt) supercell, attaches the local
mace-omat-0-small.model, and runs a short MD trajectory once for every
ensemble keyword supported by get_dyn() (the 'bussi' NVT thermostat and
every NPT-family integrator implemented in ASE), reporting wall-clock
time, throughput (ms/step), and energy/temperature stability for each.
"""
import importlib.util
import os
import sys
import time
from pathlib import Path

from ase.build import bulk
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution, Stationary, ZeroRotation,
)
from mace.calculators import MACECalculator

# Resolve paths relative to this file, not the cwd: this benchmark lives in
# mmacetools/benchmark/thermostats/, while run-md.py and the model live in
# mmacetools/runmd/.
RUNMD_DIR = Path(__file__).resolve().parent.parent.parent / 'runmd'

# run-md.py isn't a valid module name (hyphen), so load it by file path.
spec = importlib.util.spec_from_file_location("run_md", RUNMD_DIR / "run-md.py")
run_md = importlib.util.module_from_spec(spec)
sys.modules["run_md"] = run_md
spec.loader.exec_module(run_md)

#################### Input ####################
model = str(RUNMD_DIR / 'mace-omat-0-small.model')
device = 'cpu'
supercell = (2, 2, 2)   # 2x2x2 conventional MgO cells -> 64 atoms
a_mgo = 4.212           # MgO lattice constant, Angstrom
t_init = 100            # K
t_md = 300              # K
tstep = 1.0             # fs
steps = 50              # short benchmark run
interval = 10
ensembles = ['bussi', 'npt', 'nptberendsen', 'inhomogeneous_nptberendsen',
             'isotropicmtknpt', 'mtknpt', 'maskedmtknpt']
###############################################


def make_atoms():
    # cubic=True gives the conventional orthogonal cell; NPT/MelchionnaNPT
    # requires a triangular cell matrix, which the primitive rhombohedral
    # rocksalt cell does not satisfy.
    atoms = bulk('MgO', crystalstructure='rocksalt', a=a_mgo, cubic=True) * supercell
    atoms.calc = MACECalculator(model_paths=[model], device=device)
    return atoms


def benchmark(ensemble):
    atoms = make_atoms()
    MaxwellBoltzmannDistribution(atoms, temperature_K=t_init)
    Stationary(atoms)
    ZeroRotation(atoms)

    dyn = run_md.get_dyn(atoms, ensemble, tstep, t_md)
    md_output = f'mgo-{ensemble}.xyz'
    # runMD()'s logger appends to md_output, so clear any stale file from a
    # previous run first (otherwise trajectories/logs accumulate duplicates).
    if os.path.isfile(md_output):
        os.remove(md_output)

    Epot0 = atoms.get_potential_energy()
    dyn, elapsed = run_md.runMD(atoms, dyn, steps, interval, md_output)
    Epot1 = atoms.get_potential_energy()
    Tfinal = atoms.get_temperature()

    return {
        'ensemble': ensemble,
        'natoms': len(atoms),
        'time_s': elapsed,
        'ms_per_step': 1000 * elapsed / steps,
        'Epot0': Epot0,
        'Epot1': Epot1,
        'dEpot': Epot1 - Epot0,
        'Tfinal': Tfinal,
    }


if __name__ == '__main__':
    results = [benchmark(e) for e in ensembles]

    print(f"\n{'ensemble':<8} {'atoms':>6} {'time(s)':>10} {'ms/step':>10} "
          f"{'Epot0(eV)':>12} {'Epot1(eV)':>12} {'dEpot(eV)':>11} {'T_final(K)':>11}")
    for r in results:
        print(f"{r['ensemble']:<8} {r['natoms']:>6} {r['time_s']:>10.3f} "
              f"{r['ms_per_step']:>10.2f} {r['Epot0']:>12.4f} {r['Epot1']:>12.4f} "
              f"{r['dEpot']:>11.4f} {r['Tfinal']:>11.2f}")
