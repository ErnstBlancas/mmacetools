from ase.md.velocitydistribution import Stationary, ZeroRotation, MaxwellBoltzmannDistribution
from mace.calculators import MACECalculator
from ase.io import read, write
from ase.md.bussi import Bussi
from ase.md.npt import NPT
from ase.md.nptberendsen import NPTBerendsen, Inhomogeneous_NPTBerendsen
from ase.md.nose_hoover_chain import IsotropicMTKNPT, MTKNPT, MaskedMTKNPT
import os, time, torch, warnings
from ase import units
import numpy as np

"""Bussi stochastic velocity rescaling (NVT) molecular dynamics.

Based on the paper from Bussi et al., J. Chem. Phys. 126, 014101 (2007)
(also available from https://arxiv.org/abs/0803.4060).
"""
def get_dyn(init_conf, ensemble, tstep, t_md, taut=50, ttime=25, ptime=75,
            bulk_modulus=100, pressure=None, taup=1000, tdamp=None, pdamp=None,
            tchain=3, pchain=3, tloop=1, ploop=1, compressibility=None,
            mask=(1, 1, 1)):
    """Return an ASE MD driver, chosen by `ensemble` keyword.

    Supported keywords (matching ASE's own constant-pressure integrators):
      'bussi'                      -> ase.md.bussi.Bussi (NVT, not a barostat)
      'npt'                        -> ase.md.npt.NPT (Melchionna NPT)
      'nptberendsen'               -> ase.md.nptberendsen.NPTBerendsen
      'inhomogeneous_nptberendsen' -> ase.md.nptberendsen.Inhomogeneous_NPTBerendsen
      'isotropicmtknpt'            -> ase.md.nose_hoover_chain.IsotropicMTKNPT
      'mtknpt'                     -> ase.md.nose_hoover_chain.MTKNPT
      'maskedmtknpt'               -> ase.md.nose_hoover_chain.MaskedMTKNPT

    tstep, taut, ttime, ptime, taup, tdamp, pdamp are in fs (tdamp/pdamp
    default to 100/1000 timesteps, ASE's own rule of thumb, when left as
    None); bulk_modulus, pressure are in GPa; compressibility in GPa^-1
    (defaults to 1/bulk_modulus when left as None); mask selects which
    cell axes the barostat may change, as used by
    Inhomogeneous_NPTBerendsen and MaskedMTKNPT.
    """
    ensemble = ensemble.lower()
    npt_ensembles = ('npt', 'nptberendsen', 'inhomogeneous_nptberendsen',
                      'isotropicmtknpt', 'mtknpt', 'maskedmtknpt')
    if ensemble in npt_ensembles and pressure is None:
        warnings.warn(f"ensemble='{ensemble}' but no external pressure was "
                       "given; defaulting to pressure=0.0 GPa.")
        pressure = 0.0
    if tdamp is None:
        tdamp = 100*tstep
    if pdamp is None:
        pdamp = 1000*tstep
    if compressibility is None:
        compressibility = 1.0/bulk_modulus

    if ensemble == 'npt':
        # NPT/MelchionnaNPT already applies the tension/compression sign
        # convention internally for a scalar externalstress (see its
        # docstring), so pass the target pressure directly, unnegated.
        pfactor = (ptime*units.fs)**2 * bulk_modulus*units.GPa
        externalstress = pressure*units.GPa
        dyn = NPT(init_conf, tstep*units.fs, temperature_K=t_md,
                  externalstress=externalstress, ttime=ttime*units.fs,
                  pfactor=pfactor)
    elif ensemble == 'nptberendsen':
        dyn = NPTBerendsen(init_conf, tstep*units.fs, temperature_K=t_md,
                            pressure_au=pressure*units.GPa,
                            taut=taut*units.fs, taup=taup*units.fs,
                            compressibility_au=compressibility/units.GPa)
    elif ensemble == 'inhomogeneous_nptberendsen':
        dyn = Inhomogeneous_NPTBerendsen(init_conf, tstep*units.fs, temperature_K=t_md,
                                          pressure_au=pressure*units.GPa,
                                          taut=taut*units.fs, taup=taup*units.fs,
                                          compressibility_au=compressibility/units.GPa,
                                          mask=mask)
    elif ensemble == 'isotropicmtknpt':
        dyn = IsotropicMTKNPT(init_conf, tstep*units.fs, temperature_K=t_md,
                               pressure_au=pressure*units.GPa,
                               tdamp=tdamp*units.fs, pdamp=pdamp*units.fs,
                               tchain=tchain, pchain=pchain,
                               tloop=tloop, ploop=ploop)
    elif ensemble == 'mtknpt':
        dyn = MTKNPT(init_conf, tstep*units.fs, temperature_K=t_md,
                     pressure_au=pressure*units.GPa,
                     tdamp=tdamp*units.fs, pdamp=pdamp*units.fs,
                     tchain=tchain, pchain=pchain, tloop=tloop, ploop=ploop)
    elif ensemble == 'maskedmtknpt':
        dyn = MaskedMTKNPT(init_conf, tstep*units.fs, temperature_K=t_md,
                            pressure_au=pressure*units.GPa,
                            tdamp=tdamp*units.fs, pdamp=pdamp*units.fs,
                            tchain=tchain, pchain=pchain,
                            tloop=tloop, ploop=ploop,
                            mask=tuple(bool(m) for m in mask))
    elif ensemble == "bussi":
        dyn = Bussi(init_conf, tstep*units.fs, temperature_K=t_md,
                    taut=taut*units.fs)
    else:
        print('Not implmented sorry')
        exit()

    return dyn


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


if __name__ == '__main__':
    #################### Input ####################
    geom = 'relax.in'
    models = ['../../models/mace-omat-0-small.model']
    device='cpu'
    # ensemble keyword: 'bussi', 'npt', 'nptberendsen',
    # 'inhomogeneous_nptberendsen', 'isotropicmtknpt', 'mtknpt' or
    # 'maskedmtknpt' (see get_dyn() docstring)
    ensemble = 'bussi'
    t_init = 100 # K
    t_md = 300 # K
    tstep = 1.0 ## fs
    steps = 50000
    interval = 50
    md_output = 'formII-md.xyz'
    ###############################################
    # Init geometry and calculator
    init_conf = read(geom)
    init_conf.calc = MACECalculator(model_paths=models, device=device)
    # NVT cannonical ensemble with Bussi stocastich rrescaling velocities
    ## init set up
    MaxwellBoltzmannDistribution(init_conf, temperature_K=t_init)
    Stationary(init_conf)
    ZeroRotation(init_conf)

    dyn = get_dyn(init_conf, ensemble, tstep, t_md)

    if os.path.isfile(md_output):
        os.remove(md_output)

    dyn, t = runMD(init_conf, dyn, steps, interval, md_output)
    print("MD finished in {0:.8f} seconds!".format(t))

