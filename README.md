# mmacetools
- aimsout2xyz: extract positions/forces/stress_tensor from FHI-AIMS to xyz
    - For relaxation calculations extracts all steps (better sampling of the energy surface)
    - Add single atoms in train set (or in config.yaml)

- randomsample: select structures from ase md simulation (or any) and write the sampled geometries — see xyzgen/README.md for details
    - `--method random` (default): uniform random selection
    - `--method fps`: diversity-based farthest point sampling over SOAP descriptors, for a non-redundant subset

- runmd: run MD with a MACE calculator via ASE integrators — see runmd/README.md for the supported thermostats/barostats and their parameters


