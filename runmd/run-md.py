from ase.md.velocitydistribution import Stationary, ZeroRotation, MaxwellBoltzmannDistribution
from mace.calculators import MACECalculator
from ase.io import read, write
from ase.md.bussi import Bussi
import os, time, torch, path
from ase import units
import numpy as np

"""Bussi stochastic velocity rescaling (NVT) molecular dynamics.

Based on the paper from Bussi et al., J. Chem. Phys. 126, 014101 (2007)
(also available from https://arxiv.org/abs/0803.4060).
"""
#################### Input ####################
geom = 'supercell.in'
models = ['../../models/mace-omat-0-small.model']
device='cpu'
t_init = 300 # K
t_md = 300 # K
tstep = 1.0 ## fs
steps = 10000
interval = 50
md_output = 'al2o3-md.xyz'
###############################################
# Init geometry and calculator
init_conf = read(geom)
init_conf.calc = MACECalculator(model_paths=models, device=device)
# NVT cannonical ensemble with Bussi stocastich rrescaling velocities
## init set up
MaxwellBoltzmannDistribution(init_conf, temperature_K=t_init)
Stationary(init_conf)
ZeroRotation(init_conf)
dyn = Bussi(init_conf, timestep=tstep*units.fs, temperature_K=t_md,
            taut=50*units.fs)

def runMD(init_conf, dyn, steps, interval, md_output):
    ## init_conf with calculator
    ## dyn class ie NPT, NPTBerendsen...
    log = open(md_output+".log", 'w')

    def logger():
        dyn.atoms.write(md_output, append=True)
        pdum = dyn.atoms.get_potential_energy()
        kdum = dyn.atoms.get_kinetic_energy()
        vdum = dyn.atoms.get_volume()
        Tdum = dyn.atoms.get_temperature()
        t = dyn.get_time()/units.fs
        print(f"{t:12.2f} {pdum:12.6f} {kdum:12.6f} {vdum:12.6f} {Tdum:12.4f}", file=log, flush=True)

    print(f'# Time (fs) epot (eV) k (eV) avg_F (eV/ang) v (ang3) T(K)', flush=True, file=log)
    dyn.attach(logger, interval=interval)
    t0 = time.time()
    dyn.run(steps) ## mas o menos estable
    t1 = time.time()
    log.close()
    return dyn, t1-t0

if os.path.isfile(md_output):
    os.remove(md_output)

dyn, t = runMD(init_conf, dyn, steps, interval, md_output)
print("MD finished in {0:.8f} seconds!".format(t))

