# randomsample

Sample geometries from an ASE-readable MD trajectory (or any ASE-readable
multi-frame file) and write each selected frame to its own directory,
in a chosen ASE output format. Supports plain uniform random sampling and
diversity-based farthest point sampling (FPS) over SOAP descriptors.

## Arguments

| Argument | Default | Meaning |
|---|---|---|
| `--samples`, `-n` | required | Number of structures to extract |
| `--path`, `-i` | required | One or more ASE-readable files containing the MD simulation (all frames from all files are pooled together) |
| `--outfolder`, `-o` | `sample` | Prefix for the output directories (`<outfolder>-NNN/`) |
| `--format`, `-f` | `aims` | ASE output format for the written geometries |
| `--amplitude` | `None` | Std. dev. (Å) of Gaussian noise added to positions of the selected frames ("rattle"); disabled when unset |
| `--seed` | `None` | RNG seed; a time-based seed is used and printed when left unset |
| `--method` | `random` | Selection method: `random` or `fps` (see below) |
| `--no-normalize` | off (i.e. normalize) | Disable L2-normalization of the frame descriptors before FPS; only used with `--method fps` |
| `--soap-rcut` | `5.0` | SOAP cutoff radius (Å), only used with `--method fps` |
| `--soap-nmax` | `8` | SOAP `n_max`, only used with `--method fps` |
| `--soap-lmax` | `6` | SOAP `l_max`, only used with `--method fps` |
| `--soap-sigma` | `0.5` | SOAP Gaussian width `sigma`, only used with `--method fps` |

## Selection methods

### `random` — uniform random sampling

`random_sample()` shuffles the pooled frames and draws `n` indices without
replacement via `numpy.random.RandomState(seed)`. Simple and cheap, but the
result can be redundant (e.g. oversample a region the trajectory lingers
in) or miss rare/extreme configurations.

### `fps` — farthest point sampling over SOAP descriptors

`fps_sample()` computes a per-frame SOAP descriptor for every pooled frame
(via `dscribe.descriptors.SOAP`, `average="inner"`) and then greedily grows
the selection: starting from one random frame, at each step it adds the
frame whose distance to its nearest already-selected neighbor is largest.
This favors a diverse subset that covers configuration space rather than
just following trajectory order. Raises `ValueError` if `--samples` exceeds
the number of available frames.

Note: unlike `random`, which returns indices into the internally shuffled
list, `fps_sample()` returns indices in *selection order*, referring to the
original (unshuffled) frame ordering.

### Shared internals

The selection itself lives in `_greedy_fps()`, which works on any
`(n_frames, n_features)` descriptor array. Descriptor construction is
exposed separately as `soap_descriptors()` if you want the vectors
themselves.

### Descriptor normalization

`fps` L2-normalizes the per-frame descriptor to a unit vector before
selecting (`_normalize()`), unless `--no-normalize` is given.

Raw SOAP vectors have a magnitude that tracks the overall atomic
density, so unnormalized euclidean distances between frames are driven by
density and composition rather than by structural difference. For a single
fixed-composition, fixed-cell MD run that magnitude is nearly constant and
normalization changes the selection only slightly; but `--path` pools all
frames from all the files given, so as soon as trajectories with differing
composition or cell volume are mixed, normalization is what keeps FPS
selecting on structure instead of on descriptor size. On unit vectors the
distance used is the usual chordal distance of the normalized SOAP
kernel, `sqrt(2 - 2*cos)`, bounded in `[0, 2]`.

Note that this changes which frames are chosen relative to earlier
(unnormalized) runs at the same seed; pass `--no-normalize` to reproduce
the old behaviour.

## Amplitude / rattle

When `--amplitude` is set (for either method), each selected frame gets
independent Gaussian noise added to its atomic positions
(`numpy.random.RandomState(seed).randn(...) * amplitude`, in Å), applied
after the frame has been chosen.

## Reproducibility

If `--seed` is left unset, a time-based seed is generated, and the
resolved seed (whether given or generated) is always printed to stdout.
It is also stamped into a comment on line 3 of each written geometry file,
along with the frame's snapshot index and the amplitude used:

```
# Seed: <seed> Snapshot: <index> Amplitude: <amplitude> (ang)
```

so a run can be reproduced exactly by re-supplying the printed seed.

## Output layout

Any existing `<outfolder>*/` directories are removed first. Then, for each
selected frame (in the order returned by the selection method), a new
`<outfolder>-NNN/` directory is created (zero-padded, starting at `000`)
containing `<outfolder>-NNN.in`, written with ASE in `--format`, with the
seed/snapshot/amplitude comment stamped on line 3.

## Examples

```sh
# Uniform random: pick 10 frames from an MD run, write FHI-AIMS geometries
xyzgen/randomsample -i md.xyz -n 10 -o sample -f aims

# FPS: pick a diverse subset of 20 frames, with custom SOAP parameters,
# plus 0.01 Å rattle noise for extra diversity
xyzgen/randomsample -i md.xyz -n 20 --method fps \
    --soap-rcut 6.0 --soap-nmax 6 --soap-lmax 4 --amplitude 0.01
```

## Benchmarking `random` vs `fps`

`benchmark/sampling/test.py` calls `random_sample()` and `fps_sample()`
directly (the same functions this script uses) on an Al2O3 MD trajectory,
across a few sample sizes and seeds, and reports selection wall-clock time
alongside two quality metrics: mean/min pairwise SOAP-descriptor distance
among the selected frames (higher = more diverse) and the energy range
(eV/atom) spanned by the subset.
