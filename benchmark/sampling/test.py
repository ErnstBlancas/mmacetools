"""Benchmark xyzgen/randomsample's selection methods (random vs FPS).

Loads benchmark/sampling/md.xyz (an Al2O3 MD trajectory: 201 frames x 240
atoms), then for a few subset sizes n calls randomsample's random_sample()
and fps_sample() directly (the same functions the `randomsample` CLI uses)
and reports, averaged over several seeds:
  - selection wall-clock time, as actually incurred end-to-end (for fps
    this includes recomputing the SOAP descriptors, since that's the real
    cost of one `randomsample --method fps` invocation)
  - selection quality: mean/min pairwise SOAP-descriptor distance among
    the n selected frames (higher = more diverse/less redundant subset,
    i.e. better coverage of configuration space) and the energy range
    (eV/atom) spanned by the subset

Quality metrics reuse one descriptor set precomputed once up front via
randomsample.soap_descriptors() -- the same helper (and hence the same
L2-normalized descriptor space) the selection itself uses, so the metric
measures what FPS actually optimizes. It is kept separate from the timings
above so descriptor cost isn't double-counted there.
"""
import importlib.util
import sys
import time
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np
from ase.io import read, write

SAMPLING_DIR = Path(__file__).resolve().parent
XYZGEN_DIR = SAMPLING_DIR.parent.parent / 'xyzgen'

# randomsample has no file extension at all (unlike e.g. run-md.py, which
# thermostats/test.py loads the same way), so importlib can't infer a
# loader from the path and spec_from_file_location() alone returns None;
# build the SourceFileLoader explicitly instead.
loader = SourceFileLoader("randomsample", str(XYZGEN_DIR / "randomsample"))
spec = importlib.util.spec_from_loader("randomsample", loader)
randomsample = importlib.util.module_from_spec(spec)
sys.modules["randomsample"] = randomsample
loader.exec_module(randomsample)

#################### Input ####################
traj_path = str(SAMPLING_DIR / 'md.xyz')
sample_sizes = [5, 20, 50]
seeds = [0, 1, 2]
methods = ['random', 'fps']
soap_kwargs = dict(rcut=5.0, nmax=8, lmax=6, sigma=0.5)
################################################


def load_trajectory():
    frames = read(traj_path, ":")
    # Tag each frame with its original position so we can map selected
    # frames back to `desc` (and to `frames` for energies) even after
    # random_sample shuffles its own copy of the list.
    for i, atoms in enumerate(frames):
        atoms.info['orig_idx'] = i
    return frames


def write_selection(frames, orig_idx, method, n, seed):
    """Write the selected frames (in selection order) to an xyz file here,
    so the actual generated subsets can be inspected/reused, not just the
    aggregate metrics below."""
    path = SAMPLING_DIR / f'sample-{method}-n{n}-seed{seed}.xyz'
    write(path, [frames[i] for i in orig_idx])
    return path


def select(method, frames, n, seed):
    samples = list(frames)  # shallow copy: methods may reorder the list itself
    t0 = time.perf_counter()
    if method == 'random':
        sels, _ = randomsample.random_sample(samples, n, seed=seed)
    else:
        sels, _ = randomsample.fps_sample(samples, n, seed=seed, **soap_kwargs)
    elapsed = time.perf_counter() - t0
    orig_idx = np.array([samples[s].info['orig_idx'] for s in sels])
    write_selection(frames, orig_idx, method, n, seed)
    return orig_idx, elapsed


def diversity(orig_idx, desc):
    sub = desc[orig_idx]
    if len(sub) < 2:
        return float('nan'), float('nan')
    d = np.linalg.norm(sub[:, None, :] - sub[None, :, :], axis=-1)
    pair = d[np.triu_indices(len(sub), k=1)]
    return pair.mean(), pair.min()


def energy_range(frames, orig_idx):
    e = np.array([frames[i].get_potential_energy() / len(frames[i]) for i in orig_idx])
    return e.max() - e.min()


def benchmark():
    frames = load_trajectory()
    print(f"Loaded {len(frames)} frames x {len(frames[0])} atoms from {traj_path}")

    print("Precomputing SOAP descriptors for the diversity/energy-range "
          "metrics (kept separate from the timings below)...")
    t0 = time.perf_counter()
    desc = randomsample.soap_descriptors(frames, **soap_kwargs)
    print(f"  done in {time.perf_counter() - t0:.2f}s\n")

    results = []
    for method in methods:
        for n in sample_sizes:
            times, means, mins_, eranges = [], [], [], []
            for seed in seeds:
                orig_idx, elapsed = select(method, frames, n, seed)
                mean_d, min_d = diversity(orig_idx, desc)
                times.append(elapsed)
                means.append(mean_d)
                mins_.append(min_d)
                eranges.append(energy_range(frames, orig_idx))
            results.append({
                'method': method,
                'n': n,
                'time_s': np.mean(times),
                'mean_pair_dist': np.mean(means),
                'min_pair_dist': np.mean(mins_),
                'dE_eV_per_atom': np.mean(eranges),
            })
    return results


if __name__ == '__main__':
    results = benchmark()

    print(f"{'method':<8} {'n':>4} {'time(s)':>10} {'mean_dist':>11} "
          f"{'min_dist':>10} {'dE(eV/at)':>11}   (averaged over seeds {seeds})")
    for r in results:
        print(f"{r['method']:<8} {r['n']:>4} {r['time_s']:>10.3f} "
              f"{r['mean_pair_dist']:>11.4f} {r['min_pair_dist']:>10.4f} "
              f"{r['dE_eV_per_atom']:>11.6f}")
