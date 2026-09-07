# runmd

Run molecular dynamics with a MACE calculator through ASE integrators.
`run-md.py` sets up an initial configuration, thermalizes it (Maxwell-Boltzmann
+ removal of net translation/rotation), builds an ASE MD driver via
`get_dyn(...)`, and runs it with `runMD(...)` while logging time, energy,
volume and temperature every `interval` steps.

`get_dyn()` picks the driver from a single `ensemble` keyword and exposes all
the underlying ASE classes' parameters through one function signature; not
every parameter applies to every ensemble (see the per-ensemble sections
below). All of these are keyword-convertible: the script accepts them in
convenient physical units and converts them to ASE internal units before
constructing the driver.

## Units used by `get_dyn()`

| Quantity | Argument(s) | Unit accepted | Converted with |
|---|---|---|---|
| Time step / coupling & damping times | `tstep`, `taut`, `ttime`, `ptime`, `taup`, `tdamp`, `pdamp` | femtoseconds (fs) | `units.fs` |
| Pressure / bulk modulus | `pressure`, `bulk_modulus` | GPa | `units.GPa` |
| Compressibility | `compressibility` | GPa⁻¹ | `units.GPa` (inverted) |
| Temperature | `t_md` | Kelvin | passed straight through as `temperature_K` |

Notes that apply across ensembles:

- If an NPT-family ensemble is requested and `pressure` is left `None`,
  `get_dyn()` prints a warning and defaults to `pressure = 0.0` GPa.
- `tdamp`/`pdamp` default to `100*tstep`/`1000*tstep` fs when left `None`
  (ASE's own rule of thumb for Nose-Hoover-chain damping times).
- `compressibility` defaults to `1/bulk_modulus` GPa⁻¹ when left `None`.
- `mask` is always a 3-tuple over the cell's Cartesian axes, but its element
  type differs by driver: integers (0/1) for `inhomogeneous_nptberendsen`,
  booleans for `maskedmtknpt` (`get_dyn()` casts with `bool(m)` before
  passing it on).

## Ensembles

| `ensemble` keyword | ASE class | Type | Summary |
|---|---|---|---|
| `bussi` | `ase.md.bussi.Bussi` | NVT | Stochastic velocity rescaling thermostat |
| `npt` | `ase.md.npt.NPT` (Melchionna) | NPT | Nose-Hoover thermostat + Parrinello-Rahman barostat |
| `nptberendsen` | `ase.md.nptberendsen.NPTBerendsen` | NPT | Berendsen weak-coupling thermostat + isotropic barostat |
| `inhomogeneous_nptberendsen` | `ase.md.nptberendsen.Inhomogeneous_NPTBerendsen` | NPT | Berendsen weak coupling, independent per-axis cell scaling |
| `isotropicmtknpt` | `ase.md.nose_hoover_chain.IsotropicMTKNPT` | NPT | Nose-Hoover chain + Martyna-Tobias-Klein barostat, isotropic volume only |
| `mtknpt` | `ase.md.nose_hoover_chain.MTKNPT` | NPT | Nose-Hoover chain + MTK barostat, full cell (shape + volume) |
| `maskedmtknpt` | `ase.md.nose_hoover_chain.MaskedMTKNPT` | NPT | Nose-Hoover chain + MTK barostat, restricted to chosen axes |

### `bussi` — Bussi stochastic velocity rescaling (NVT)

A canonical-ensemble thermostat that rescales velocities with a stochastic
term so both the average and the fluctuations of the kinetic energy match the
canonical distribution (unlike plain velocity rescaling). Not a barostat —
the cell is fixed.

| Parameter | Meaning |
|---|---|
| `t_md` | Target temperature (K), passed as `temperature_K` |
| `taut` | Thermostat coupling time constant (fs) |

Reference: G. Bussi, D. Donadio, M. Parrinello, *Canonical sampling through
velocity rescaling*, J. Chem. Phys. **126**, 014101 (2007),
https://arxiv.org/abs/0803.4060.

### `npt` — Melchionna NPT (Nose-Hoover + Parrinello-Rahman)

ASE's `NPT` class (an alias of `ase.md.melchionna.MelchionnaNPT`) couples a
Nose-Hoover thermostat to Parrinello-Rahman cell dynamics, integrated with a
centered-difference scheme. The cell is free to change shape and volume.

| Parameter | Meaning |
|---|---|
| `t_md` | Target temperature (K) |
| `ttime` | Thermostat characteristic time (fs) |
| `pressure` | Target external pressure (GPa); becomes a scalar `externalstress = pressure*units.GPa` (`NPT`/`MelchionnaNPT` already applies the sign convention for tension/compression internally, so the target pressure is passed unnegated) |
| `ptime`, `bulk_modulus` | Combined into `pfactor = (ptime*units.fs)**2 * bulk_modulus*units.GPa`, the barostat's characteristic response constant |
| `mask` | Which cell axes the barostat may change (3-tuple, ints) |

References: S. Melchionna, G. Ciccotti, B. L. Holian, *Hoover NPT dynamics
for systems varying in shape and size*, Mol. Phys. **78**, 533 (1993),
https://doi.org/10.1080/00268979300100371; S. Melchionna, *Constrained
systems and statistical distribution*, Phys. Rev. E **61**, 6165 (2000),
https://doi.org/10.1103/PhysRevE.61.6165.

### `nptberendsen` — Berendsen weak-coupling NPT

Rescales velocities and the (isotropic) cell volume towards target
temperature/pressure with first-order relaxation ("weak coupling"). Simple
and robust, but does not sample the exact canonical/isothermal-isobaric
ensemble — mainly useful for equilibration rather than production sampling.

| Parameter | Meaning |
|---|---|
| `t_md` | Target temperature (K) |
| `pressure` | Target pressure (GPa), passed as `pressure_au = pressure*units.GPa` |
| `taut` | Temperature coupling time constant (fs) |
| `taup` | Pressure coupling time constant (fs) |
| `compressibility` / `bulk_modulus` | `compressibility_au = compressibility/units.GPa`, the isothermal compressibility used to relate pressure error to a volume rescaling factor; `compressibility` defaults to `1/bulk_modulus` |

Reference: H. J. C. Berendsen, J. P. M. Postma, W. F. van Gunsteren,
A. DiNola, J. R. Haak, *Molecular dynamics with coupling to an external
bath*, J. Chem. Phys. **81**, 3684 (1984),
https://doi.org/10.1063/1.448118.

### `inhomogeneous_nptberendsen` — Berendsen NPT with per-axis scaling

Same weak-coupling scheme as `nptberendsen`, but the three cell axes are
rescaled independently instead of isotropically — useful for systems under
anisotropic stress or with anisotropic elastic response, while keeping cell
angles fixed.

Same parameters as `nptberendsen`, plus:

| Parameter | Meaning |
|---|---|
| `mask` | 3-tuple of ints (0/1); axes with `0` are excluded from barostat rescaling |

Same reference as `nptberendsen`.

### `isotropicmtknpt` — Nose-Hoover chain + isotropic MTK barostat

Isothermal-isobaric dynamics using a Nose-Hoover *chain* thermostat (a chain
of `tchain` coupled thermostat variables, more robust than a single
Nose-Hoover thermostat) together with a Martyna-Tobias-Klein barostat
restricted to isotropic volume fluctuations (cell shape fixed).

| Parameter | Meaning |
|---|---|
| `t_md` | Target temperature (K) |
| `pressure` | Target pressure (GPa), passed as `pressure_au` |
| `tdamp` | Thermostat characteristic time (fs); default `100*tstep` |
| `pdamp` | Barostat characteristic time (fs); default `1000*tstep` |
| `tchain` | Number of thermostat chain variables (default 3) |
| `pchain` | Number of barostat chain variables (default 3) |
| `tloop` | Thermostat integration sub-steps per MD step (default 1) |
| `ploop` | Barostat integration sub-steps per MD step (default 1) |

References: G. J. Martyna, M. L. Klein, M. E. Tuckerman, *Nosé-Hoover chains:
The canonical ensemble via continuous dynamics*, J. Chem. Phys. **97**, 2635
(1992), https://doi.org/10.1063/1.463940; M. E. Tuckerman, J. Alejandre,
R. López-Rendón, A. L. Jochim, G. J. Martyna, *A Liouville-operator derived
measure-preserving integrator for molecular dynamics simulations in the
isothermal-isobaric ensemble*, J. Phys. A **39**, 5629 (2006),
https://doi.org/10.1088/0305-4470/39/19/S18.

### `mtknpt` — Nose-Hoover chain + full-cell MTK barostat

Same thermostat/barostat machinery as `isotropicmtknpt`, but the barostat
acts on the full cell matrix (9 degrees of freedom: shape and volume both
fluctuate) rather than volume alone.

Parameters identical to `isotropicmtknpt` (same table above); no `mask`.

Same references as `isotropicmtknpt`.

### `maskedmtknpt` — Nose-Hoover chain + axis-restricted MTK barostat

Same as `mtknpt`, but cell fluctuations are restricted to an arbitrary
subset of the three crystallographic axes rather than the full 9-parameter
cell — e.g. for slabs or systems where only some directions should relax.

Same parameters as `isotropicmtknpt`/`mtknpt`, plus:

| Parameter | Meaning |
|---|---|
| `mask` | 3-tuple, cast with `bool(m)`; `True` allows that axis to fluctuate, `False` fixes it |

Same references as `isotropicmtknpt`, plus (for the masked-cell conserved
quantities specifically): *Conserved quantities and ensemble measure for
Martyna-Tobias-Klein barostats with restricted cell degrees of freedom*,
https://arxiv.org/abs/2603.24061.

## `runMD()` logging

Independent of the ensemble, `runMD(init_conf, dyn, steps, interval,
md_output)` attaches a logger every `interval` steps that appends the current
frame to `md_output` and writes a line to `<md_output>.log` with:

```
Time (fs)   epot (eV)   k (eV)   v (ang^3)   T (K)
```
