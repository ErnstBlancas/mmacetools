This directory holds the two tools that build MACE training sets:
[`randomsample`](#randomsample) picks frames out of an MD trajectory, and
[`aimsout2xyz`](#aimsout2xyz) turns the resulting FHI-AIMS outputs into
extended XYZ.

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
| `--soap-rcut` | `5.0` | SOAP cutoff radius (Å); one value, or two to combine two descriptors. Only used with `--method fps` |
| `--soap-nmax` | `8` | SOAP `n_max`; one value, or one per descriptor. Only used with `--method fps` |
| `--soap-lmax` | `6` | SOAP `l_max`; one value, or one per descriptor. Only used with `--method fps` |
| `--soap-sigma` | `0.5` | SOAP Gaussian width `sigma`; one value, or one per descriptor. Only used with `--method fps` |
| `--soap-weight` | `0.5` | Share of the squared FPS distance carried by the first descriptor when two are given. Ignored with a single descriptor |

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

### Combining two SOAP descriptors

A single cutoff has to serve two jobs at once: `r_cut` small enough to
resolve conformation inside a molecule, and large enough to see how
molecules pack around each other. Giving two values to any of the four
`--soap-*` options builds two SOAP descriptors and concatenates them, so
FPS sees both length scales:

```sh
xyzgen/randomsample -i md.xyz -n 5 --method fps \
    --soap-rcut 3.0 7.0 --soap-nmax 6 4 --soap-lmax 4 3 --soap-sigma 0.3 0.6
```

Options given once are shared by both descriptors, so
`--soap-rcut 3.0 7.0 --soap-nmax 6` is a legal shorthand for two
descriptors that differ only in cutoff. More than two is rejected.

The two blocks are put on a common footing before being joined: each is
L2-normalized per frame, then divided by its own RMS pairwise distance
(`_block_scale()`), then scaled by `sqrt(w)` and `sqrt(1-w)`. **That
rescaling is not cosmetic.** How far a SOAP descriptor moves over a
trajectory depends strongly on its cutoff — across a short/long pair the
typical distance can differ by 4x, i.e. 15x in squared distance — so a
plain concatenation is silently dominated by one block and reproduces its
selection exactly, making the second descriptor a pure waste of CPU.
After the rescaling the squared distance is

```
d^2 = w * d1^2 + (1 - w) * d2^2
```

on the standardized per-block distances, so `--soap-weight` reads directly
as the fraction of the FPS distance contributed by the first descriptor,
and the default `0.5` really is an even split.

With a single descriptor none of this happens and the result is bit-identical
to earlier versions.

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

# Two-scale FPS: one descriptor for intramolecular geometry, one for
# packing, contributing equally to the FPS distance
xyzgen/randomsample -i md.xyz -n 5 --method fps \
    --soap-rcut 3.0 7.0 --soap-nmax 6 4 --soap-lmax 4 3 \
    --soap-sigma 0.3 0.6 --soap-weight 0.5
```

## Benchmarking `random` vs `fps`

`benchmark/sampling/test.py` calls `random_sample()` and `fps_sample()`
directly (the same functions this script uses) on an Al2O3 MD trajectory,
across a few sample sizes and seeds, and reports selection wall-clock time
alongside two quality metrics: mean/min pairwise SOAP-descriptor distance
among the selected frames (higher = more diverse) and the energy range
(eV/atom) spanned by the subset.

# aimsout2xyz

Turn FHI-AIMS outputs into an extended XYZ dataset for MACE. Reads energy,
positions, forces and (when present) the analytical stress tensor, and writes
them with the keys `energy_dft` / `forces_dft` / `stress` that
`multitrain/multimace.py` expects.

## Arguments

| Argument | Default | Meaning |
|---|---|---|
| `--files`, `-i` | required | One or more FHI-AIMS output files |
| `--fout`, `-o` | `aims_output` | Output name prefix (`<fout>.xyz`, or `<fout>.train.xyz` / `<fout>.test.xyz` with `--fraction`) |
| `--fraction`, `-f` | `None` | Split into a train and a test set; the value is the **test** share, given either as a fraction (`0.2`) or a percentage (`20`). Without it everything goes into one file |
| `--isolated`, `-iso` | `None` | FHI-AIMS outputs of isolated atoms; written as `config_type=IsolatedAtom` frames (the E0s references) at the top of the training set |
| `--seed` | `None` | RNG seed for the train-test shuffle; a time-based seed is used and printed when left unset. Only used with `--fraction` |
| `--no-shuffle` | off (i.e. shuffle) | Split in the order the files were given instead of shuffling. Only used with `--fraction` |

## What gets extracted

One frame per converged SCF cycle, so a single-point run gives one frame and a
relaxation gives one per relaxation step, each with the geometry that produced
that energy and those forces.

`Final atomic structure:` is deliberately ignored: AIMS repeats there the
geometry of the last relaxation step, which has already been collected from
`Updated atomic structure:` (or from the input geometry, when the relaxation
converged straight away). Counting it would give one geometry more than there
are energies.

Stress is written only when the run computed it. The sign convention is the one
AIMS prints, which is also ASE's (`stress` is σ, so the pressure is `-tr(σ)/3`).

## Train-test split

The file list is shuffled before being cut, so the two sets are a random draw
rather than the head and tail of the input order. Pass `--seed` to make the
draw reproducible, or `--no-shuffle` to keep the given order — with sorted
input that turns the test set into a whole-polymorph holdout, which measures
something quite different from a random split.

The split is over *files*, not frames: every structure from one relaxation stays
on the same side, which is what you want, since consecutive relaxation steps are
nearly identical and would otherwise leak between the sets.

## Incomplete runs

An output that never finished (no energy, no forces, truncated mid-file) is
reported on stderr and left out, and the remaining files are still written, so
one dead job does not cost you the whole dataset. The exit status is non-zero
whenever anything was skipped, so a pipeline notices. The counts of geometries,
force blocks and energies are compared for every file, and any disagreement is
reported rather than silently paired up.

Output files are truncated on open, so re-running a command overwrites its
dataset instead of appending a second copy to it.

## Examples

```sh
# Single dataset from a directory of single-point runs
xyzgen/aimsout2xyz -i benchmark/extract/*/*/*.out -o formII

# 20% test split, reproducible, with the isolated-atom references
xyzgen/aimsout2xyz -i dft/*/*.out -iso iso/{H,C,N}.out -o formII -f 20 --seed 42

# Keep the input order (e.g. a deliberate holdout of the last polymorphs)
xyzgen/aimsout2xyz -i $(ls -d dft/*/*.out) -o formII -f 20 --no-shuffle
```
