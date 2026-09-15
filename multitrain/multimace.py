"""two stage mace finetuning on slurm: train many seeds, rank them, extend the
best ones.

the same file plays both roles:

  multimace.py --watcher            prepares seed_*/ with their train/test
                                    splits and configs, submits the slurm jobs,
                                    watches them, ranks the results and submits
                                    stage two for the best ones

  multimace.py --finetune [CONFIG]  runs one mace finetuning in the current
                                    directory from the xyz sets and yaml
                                    already there (default config.yaml); this
                                    is what your slurm files call

the slurm scripts themselves are yours (slurmfiles), so anything the queue
needs (partition, walltime, modules, gpus) lives there and not here.  one is
copied into every seed directory and submitted from it, so it runs with the
seed as its working directory:

  #!/bin/bash
  #SBATCH --time=01:00:00
  #SBATCH --cpus-per-task=8
  module load mace
  srun python ../multimace.py --finetune            ## stage one
  srun python ../multimace.py --finetune config-ext.yaml   ## stage two
"""
import argparse, glob, logging, os, random, shutil, subprocess, sys, time, warnings
from ase.io import read, write
import numpy as np
warnings.filterwarnings("ignore")

#################### input ####################
dftsources = '*.xyz'    ## dftxyz files to shuffle again
nseeds = 3            ## number of seeds trained in stage one
nlast = 2               ## how many of them are extended in stage two
max_epochs = [1, 2]    ## max_epochs for [stage one, stage two]
device = ['cpu', 'gpu'] ## device(s) for [stage one, stage two]; give one value
                        ## to use it for both steps
use = 'force'           ## best model: force, energy, loss, stress (see metrics)
worst = False            ## if true train nlast-1 models and the worst performing one
config = {              ## or read corresponding config.yaml
          'model_dir': 'models',
          'log_dir': 'logs-res',
          'results': 'results-res',
          'checkpoints_dir': 'checkpoints',
          'name': 'finetune.model',
          'foundation_model': '../finetune.model.model',
          'multiheads_finetuning': False,
          'train_file': 'train.xyz', 
          'test_file': 'test.xyz', 
          'valid_fraction': 0.20,
          'energy_weight': 1.0,
          'forces_weight': 100.0,
          'lr': 0.01,
          'scaling': 'rms_forces_scaling', 
          'batch_size': 4,
          'max_num_epochs': 2,
          'ema': True,
          'ema_decay': 0.99,
          'default_dtype': 'float64',
          'r_max': 6.0,
          'energy_key': 'energy_dft',
          'forces_key': 'forces_dft',
          'E0s': "{8:-0.204170677752367E+04, 13:-0.661017373326158E+04}",
          'seed': 101,
          'restart_latest': False,
          }
## config written by this script and read by mace, one per stage.  the stage
## two slurm file has to pass the second one, it is the restart
configfiles = ['config.yaml', 'config-ext.yaml']
## your slurm scripts, [stage one, stage two]: copied into every directory of
## that stage and submitted there, so they see the config/train/test files next
## to them and mace writes its output in place.  each one is expected to run
## 'python ../multimace.py --finetune [config]' with the config of its stage
slurmfiles = ['stage-one.slurm', 'stage-two.slurm']
check = 10              ## seconds between squeue checks
logfile = 'multimace.log'   ## progress log of the whole run
rankfile = 'ranking.log'    ## ranking of the seeds, emptied at every launch
fraction = 0.8          ## dftsources use for training
clean = True            ## remove not valid seeds files
############################################### 

## selection criterion -> metric key logged in results-*.txt.  'stress' is only
## reported when mace computes it, i.e. with loss stress/huber/universal
metrics = {'force': 'rmse_f',
           'energy': 'rmse_e_per_atom',
           'loss': 'loss',
           'stress': 'rmse_stress'}

logpath = os.path.abspath(logfile)
rankpath = os.path.abspath(rankfile)

def log(msg):
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    fout = open(logpath, 'a')
    print(line, file=fout, flush=True)
    fout.close()
    print(line, flush=True)

def write_yaml(path, configfile, config, seed):
    fout = open(f'{path}/{configfile}', 'w')
    for i in config:
        if i == "E0s":
            print(f'{i}: "{config[i]}"', file=fout)
        elif i == "seed":
            print(f'{i}: {seed}', file=fout)
        else:
            print(f'{i}: {config[i]}', file=fout)
    fout.close()
    log(f'{path}: wrote {path}/{configfile} (seed {seed}, max_num_epochs '
        f'{config["max_num_epochs"]})')

def shuffle(path, dftsources, fraction, seed, config):
    ## the IsolatedAtom frames are the ones mace reads E0s from, and only
    ## in the training file: a split that leaves one out dies with
    ## 'Atomic number ... not found in atomic_energies_dict', so they are
    ## kept out of the shuffle and pinned to every training set
    iso = [i for i, s in enumerate(dftsources)
           if len(s) == 1 and s.info.get('config_type') == 'IsolatedAtom']
    rest = [i for i in range(len(dftsources)) if i not in set(iso)]
    ntrain = int(len(rest)*fraction)
    rs = np.random.RandomState(seed)
    idx = list(rs.permutation(rest))
    train, test = iso + idx[:ntrain], idx[ntrain:]
    train_file = config['train_file']
    test_file = config['test_file']
    write(f'{path}/{train_file}', [dftsources[i] for i in train])
    write(f'{path}/{test_file}', [dftsources[i] for i in test])
    log(f'{path}: wrote {path}/{train_file} ({len(train)} samples, {len(iso)} '
        f'isolated atoms) and {path}/{test_file} ({len(test)} samples)')

def prepare_seed(seed, config, dftsources, fraction, epochs, device):
    """fresh seed_*/ with its own train/test split, config and slurm script."""
    path = f'seed_{seed}'
    shutil.rmtree(path, ignore_errors=True)
    os.mkdir(path)
    ## own copy: the module level dict is shared by every seed
    cfg = dict(config)
    cfg['max_num_epochs'] = epochs
    cfg['device'] = device
    shuffle(path, dftsources, fraction, seed, cfg)
    write_yaml(path, configfiles[0], cfg, seed)
    shutil.copy(slurmfiles[0], path)
    return path

def prepare_extend(path, config, epochs, device):
    """config-ext.yaml restarting the seed from its latest checkpoint."""
    fin = open(f'{path}/{configfiles[0]}', 'r')
    old = {}
    for line in fin:
        line = line.strip()
        key = line.split(':')[0]
        old[key] = line[len(key)+1:]
    fin.close()
    ## the seed names the checkpoint that restart_latest has to pick up
    seed = int(old['seed'])
    cfg = dict(config)
    cfg['max_num_epochs'] = epochs
    cfg['device'] = device
    cfg['restart_latest'] = True
    cfg['log_dir'] = cfg['log_dir']+'-ext'
    cfg['results'] = cfg['results']+'-ext'
    write_yaml(path, configfiles[1], cfg, seed)
    shutil.copy(slurmfiles[1], path)
    log(f'{path}: restarting from the latest checkpoint, results in '
        f'{path}/{cfg["results"]}')
    return path

def finetune(configfile):
    """one mace run in the current directory: --finetune, the worker half.

    nothing is prepared or shuffled here, the xyz sets and the yaml are taken
    as they are, so a job can be resubmitted as often as needed without the
    split moving under it
    """
    if not os.path.isfile(configfile):
        raise FileNotFoundError(f'{configfile} not found in {os.getcwd()}')
    from mace.cli.run_train import main as mace_run_train_main
    import torch
    ## slurm gives the allocation, torch otherwise grabs every core on the node
    ncpu = os.environ.get('SLURM_CPUS_PER_TASK')
    if ncpu:
        torch.set_num_threads(int(ncpu))
    print(f'{os.getcwd()}: mace --config {configfile}'
          + (f', {ncpu} threads' if ncpu else ''), flush=True)
    ## mace takes over the root logger and writes its own log next to the config
    logging.getLogger().handlers.clear()
    sys.argv = ["program", "--config", configfile]
    mace_run_train_main()

def submit(path, script):
    ## --parsable prints the bare job id instead of 'Submitted batch job ...'
    p = subprocess.run(['sbatch', '--parsable', os.path.basename(script)],
                       capture_output=True, text=True, cwd=path)
    if p.returncode != 0:
        raise RuntimeError(f'sbatch failed in {path}: {p.stderr.strip()}')
    jobid = p.stdout.strip().split(';')[0]
    log(f'{path}: submitted {os.path.basename(script)} as job {jobid}')
    return jobid

def queued(ids):
    """the ids squeue still knows about; one that left the queue is done."""
    p = subprocess.run(['squeue', '-h', '-o', '%i', '-j', ','.join(ids)],
                       capture_output=True, text=True)
    return {i for i in p.stdout.split() if i}

def states(ids):
    """final state per job from the accounting database, {} if there is none."""
    p = subprocess.run(['sacct', '-n', '-X', '-P', '-o', 'JobID,State',
                        '-j', ','.join(ids)], capture_output=True, text=True)
    out = {}
    for line in p.stdout.splitlines():
        f = line.split('|')
        if len(f) >= 2:
            out[f[0]] = f[1].split()[0]   ## 'CANCELLED by 1000' -> 'CANCELLED'
    return out

def wait(jobs, stage):
    """block until every job of {jobid: path} has left the queue."""
    left = set(jobs)
    while left:
        alive = queued(sorted(left))
        for i in sorted(left - alive):
            log(f'{stage}: {jobs[i]} (job {i}) finished')
        left &= alive
        if left:
            time.sleep(check)
    st = states(sorted(jobs))
    bad = [jobs[i] for i in jobs if st.get(i, 'COMPLETED') != 'COMPLETED']
    log(f'{stage}: all {len(jobs)} job(s) finished'
        + (f', {len(bad)} not COMPLETED: ' + ', '.join(
            f'{jobs[i]} ({st[i]})' for i in sorted(jobs)
            if st.get(i, 'COMPLETED') != 'COMPLETED') if bad else ''))
    return bad

def process_results(path, nlast, use, worst=None):
    import json
    if use not in metrics:
        raise ValueError(f"unknown criterion '{use}', use one of {sorted(metrics)}")
    key = metrics[use]
    files, values, skipped = [], [], []
    for f in sorted(glob.glob(path)):
        ## results-*.txt interleaves mode="opt" records, which carry only the
        ## training loss, with the mode="eval" ones holding the rmse keys
        fin = open(f, 'r')
        evals = [r for r in (json.loads(l) for l in fin)
                 if r.get('mode') == 'eval' and r.get(key) is not None]
        fin.close()
        if not evals:
            skipped.append(f)
            continue
        files.append(f)
        values.append(evals[-1][key])
    if skipped:
        log(f'no {key} logged by {len(skipped)} results file(s): '
            + ', '.join(i.split(os.sep)[0] for i in skipped))
    if len(files) < nlast:
        raise RuntimeError(f'only {len(files)} of the seeds produced usable '
                           f'results, {nlast} needed')
    values = np.array(values)
    ## argsort, not argpartition: the latter only places element nlast, so its
    ## last index is not the worst model
    order = np.argsort(values)
    if worst and nlast < 2:
        log(f'worst=True keeps nlast-1 best models, so nlast={nlast} would pick '
            'the worst one and no best one: taking the best one instead')
        worst = False
    if worst:
        best = list(order[:nlast-1]) + [order[-1]]
    else:
        best = list(order[:nlast])
    paths = [files[i].split(os.sep)[0] for i in best]
    log(f'ranking of the {len(files)} seeds with usable results '
        f'by {key} ({use}):')
    ## same table in its own file, without the timestamps of the progress log
    ## and column aligned, so it can be read or plotted directly
    fout = open(rankpath, 'a')
    print(f'# ranking by {key} ({use}), {len(files)} seed(s) with usable '
          f'results, {time.strftime("%Y-%m-%d %H:%M:%S")}', file=fout)
    print(f'# rank  {"seed":24s} {key:>14s}  selected', file=fout)
    for rank, i in enumerate(order):
        name = files[i].split(os.sep)[0]
        mark = '*' if i in best else ' '
        log(f'  {mark} {rank+1:3d}. {name:24s} {key} = {values[i]:.6f}')
        print(f'{rank+1:6d}  {name:24s} {values[i]:14.6f}  {mark}', file=fout)
    fout.close()
    log(f'ranking written to {rankpath}')
    if worst:
        log(f'selection: {nlast-1} best + the worst one (marked with *)')
    else:
        log(f'selection: the {nlast} best ones (marked with *)')
    return paths, values

def clean_init(all_path, best_path):
    keep = {os.path.normpath(p) for p in best_path}
    removed = []
    for i in glob.glob(all_path):
        if os.path.normpath(i) in keep:
            continue
        else:
            shutil.rmtree(i, ignore_errors=True)
            removed.append(os.path.normpath(i))
    log(f'cleanup: removed {len(removed)} discarded seed director(y/ies)'
        + (': ' + ', '.join(sorted(removed)) if removed else ''))

def parse_args():
    p = argparse.ArgumentParser(
        description='two stage mace finetuning on slurm',
        epilog='--watcher drives the run from the login node, --finetune is '
               'the single training the slurm files call inside each seed')
    ## one of the two is always required: running the file bare used to launch
    ## the whole queue, which is not something to do by accident
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--watcher', action='store_true',
                   help='prepare the seeds, submit, watch, rank and extend')
    g.add_argument('--finetune', nargs='?', const=configfiles[0],
                   metavar='CONFIG',
                   help=f'finetune in the current directory from CONFIG '
                        f'(default {configfiles[0]}) and the xyz sets next to it')
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.finetune:
        ## worker: inside a slurm job, in a seed directory
        finetune(args.finetune)
        sys.exit(0)
    ## the progress log is appended to across runs, the ranking is not: it
    ## describes this run's seeds only, so a stale one would be misleading
    if os.path.isfile(rankpath):
        os.remove(rankpath)
    log(f'=== multimace start, log in {logpath}, ranking in {rankpath} ===')
    for i in slurmfiles:
        if not os.path.isfile(i):
            raise FileNotFoundError(f'slurm script {i} not found')
    if type(dftsources) == str:
        files = glob.glob(dftsources)
    else:
        files = dftsources
    dftsamples = []
    for f in files:
        for s in read(f, ':'):
            dftsamples.append(s)
    log(f'read {len(dftsamples)} samples from {len(files)} file(s): '
        + ', '.join(str(f) for f in files))
    ## device(s): one value covers both steps, two values assign one per step
    device1 = device[0]
    device2 = device[0] if len(device) == 1 else device[1]
    ## stage one: one directory and one job per seed
    ## drawn without replacement: a repeated seed means a shared seed_* directory
    seeds = random.sample(range(2**31-1), nseeds)
    log(f'stage one: {nseeds} seeds, {max_epochs[0]} epochs, device {device1}, '
        f'{slurmfiles[0]}')
    jobs = {}
    for s in seeds:
        path = prepare_seed(s, config, dftsamples, fraction, max_epochs[0],
                            device1)
        jobs[submit(path, slurmfiles[0])] = path
    bad = wait(jobs, 'stage one')
    ## stage two: rank what stage one produced and extend the selection
    paths, stats = process_results('seed*/results-res/*txt', nlast, use, worst)
    ## a run that died mid training still leaves the eval records of the epochs
    ## it did get through, so it can be ranked and picked like a complete one
    crashed = [i for i in paths if i in set(bad)]
    if crashed:
        log('warning: picked for stage two although stage one did not finish: '
            + ', '.join(crashed))
    log(f'stage two: extending {len(paths)} model(s) to {max_epochs[1]} epochs, '
        f'device {device2}, {slurmfiles[1]}: ' + ', '.join(paths))
    jobs = {}
    for p in paths:
        prepare_extend(p, config, max_epochs[1], device2)
        jobs[submit(p, slurmfiles[1])] = p
    wait(jobs, 'stage two')
    # remove all the useless stuff
    if clean:
        clean_init('seed*/', paths)
    log('=== multimace done ===')
