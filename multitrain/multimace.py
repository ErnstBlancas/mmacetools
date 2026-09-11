import os, glob, logging, random, shutil, sys, time, warnings
from multiprocessing import Process
from ase.io import read, write
import numpy as np
warnings.filterwarnings("ignore")

#################### input ####################
dftsources = '*.xyz'    ## dftxyz files to shuffle again
nmax = 256               ## max number of cores
ncore_per_task0 = 2     ## to run nmax//ncore_per_task0 initial seeds
nlast = 8               ## run nmax//nlast best models with more epochs
max_epochs = [2, 20]   ## max_epochs for the best seeds
device = ['cpu', 'cpu'] ## device(s) for [stage one, stage two]; give one value
                        ## to use it for both steps
use = 'force'           ## best model: force, energy, loss, stress (see metrics)
worst = True            ## if true train nlast-1 models and the worst performing one
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
configfile0 = 'config.yaml' ## config filename
logfile = 'multimace.log'   ## progress log of the whole run
fraction = 0.8          ## dftsources use for training
clean = True            ## remove not valid seeds files
############################################### 

## selection criterion -> metric key logged in results-*.txt.  'stress' is only
## reported when mace computes it, i.e. with loss stress/huber/universal
metrics = {'force': 'rmse_f',
           'energy': 'rmse_e_per_atom',
           'loss': 'loss',
           'stress': 'rmse_stress'}

## the workers chdir into their own seed_*/ and mace takes over the root logger,
## so the progress log is a plain append-and-flush file on an absolute path
logpath = os.path.abspath(logfile)

def log(msg):
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    fout = open(logpath, 'a')
    print(line, file=fout, flush=True)
    fout.close()
    print(line, flush=True)

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
    for rank, i in enumerate(order):
        name = files[i].split(os.sep)[0]
        mark = '*' if i in best else ' '
        log(f'  {mark} {rank+1:3d}. {name:24s} {key} = {values[i]:.6f}')
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

class mace_process(Process):
    def __init__(self, ncpu, config, configfile, dft, trainfraction, epochs,
                 dir=None, seed=None):
        ## init tye process
        super().__init__()
        self.ncpu = ncpu
        self.seed = seed if seed is not None else random.randrange(2**31-1)
        self.is_extend = False
        ## own copy: the module level dict is shared by every process
        self.config = dict(config)
        self.configfile = configfile
        self.dftsources = dft
        self.fraction = trainfraction
        if dir:
            self.is_extend = True
            self.path = dir
        self.config['max_num_epochs'] = epochs

    @property
    def label(self):
        ## the parent never runs gen_path (the fork does), so the run is named
        ## after the seed/directory already known at construction time
        return self.path if self.is_extend else f'seed_{self.seed}'

    def _write_yaml(self):
        fout = open(f'{self.path}/{self.configfile}', 'w')
        for i in self.config:
            if i == "E0s":
                print(f'{i}: "{self.config[i]}"', file=fout)
            elif i == "seed":
                print(f'{i}: {self.seed}', file=fout)
            else:
                print(f'{i}: {self.config[i]}', file=fout)
        fout.close()
        log(f'{self.label}: wrote {self.path}/{self.configfile} '
            f'(seed {self.seed}, max_num_epochs {self.config["max_num_epochs"]})')

    def _shuffle(self):
        ## the IsolatedAtom frames are the ones mace reads E0s from, and only
        ## in the training file: a split that leaves one out dies with
        ## 'Atomic number ... not found in atomic_energies_dict', so they are
        ## kept out of the shuffle and pinned to every training set
        iso = [i for i, s in enumerate(self.dftsources)
               if len(s) == 1 and s.info.get('config_type') == 'IsolatedAtom']
        rest = [i for i in range(len(self.dftsources)) if i not in set(iso)]
        ntrain = int(len(rest)*self.fraction)
        rs = np.random.RandomState(self.seed)
        idx = list(rs.permutation(rest))
        train, test = iso + idx[:ntrain], idx[ntrain:]
        train_file = self.config['train_file']
        test_file = self.config['test_file']
        write(f'{self.path}/{train_file}',
              [self.dftsources[i] for i in train])
        write(f'{self.path}/{test_file}',
              [self.dftsources[i] for i in test])
        log(f'{self.label}: wrote {self.path}/{train_file} ({len(train)} samples'
            f', {len(iso)} isolated atoms) and {self.path}/{test_file} '
            f'({len(test)} samples)')

    def gen_path(self):
        self.path = f'seed_{self.seed}'
        shutil.rmtree(self.path, ignore_errors=True)
        os.mkdir(self.path)
        log(f'{self.label}: created directory {self.path}/')
        self._shuffle()
        self._write_yaml()

    def load_config(self):
        fin = open(f'{self.path}/{self.configfile}', 'r')
        config = {}
        for line in fin:
            line = line.strip()
            key = line.split(':')[0]
            config[key] = line[len(key)+1:]
        fin.close()
        ## the seed names the checkpoint that restart_latest has to pick up
        self.seed = int(config['seed'])
        self.config['restart_latest'] = True
        self.config['log_dir'] = self.config['log_dir']+'-ext'
        self.config['results'] = self.config['results']+'-ext'
        self.configfile = "config-ext.yaml"
        self._write_yaml()
        log(f'{self.label}: restarting from the latest checkpoint, results in '
            f'{self.path}/{self.config["results"]}')

    def finetune(self):
        from mace.cli.run_train import main as mace_run_train_main
        import torch
        torch.set_num_threads(self.ncpu)
        logging.getLogger().handlers.clear()
        sys.argv = ["program", "--config", self.configfile]
        mace_run_train_main()

    def run(self):
        stage = 'extend' if self.is_extend else 'init'
        log(f'{self.label}: start ({stage}, {self.ncpu} threads, '
            f'{self.config["max_num_epochs"]} epochs)')
        if self.is_extend:
            self.load_config()
        if not self.is_extend:
            self.gen_path()
        os.chdir(self.path)
        try:
            self.finetune()
        except Exception as e:
            log(f'{self.label}: FAILED ({stage}): {type(e).__name__}: {e}')
            raise
        log(f'{self.label}: done ({stage})')

def wait(pps, stage):
    ## report every worker as it is joined, so the log tells finished from dead
    for i in pps:
        i.join()
    ok = [i.label for i in pps if i.exitcode == 0]
    bad = [i.label for i in pps if i.exitcode != 0]
    log(f'{stage}: {len(ok)}/{len(pps)} finished'
        + (': ' + ', '.join(ok) if ok else ''))
    if bad:
        log(f'{stage}: {len(bad)} did not finish: ' + ', '.join(
            f'{i.label} (exit {i.exitcode})' for i in pps if i.exitcode != 0))
    return ok, bad

if __name__ == "__main__":
    log(f'=== multimace start, log in {logpath} ===')
    # no parallel
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
    ## init
    nseeds = nmax//ncore_per_task0
    ## drawn without replacement: a repeated seed means a shared seed_* directory
    seeds = random.sample(range(2**31-1), nseeds)
    config['device'] = device1
    pps = []
    for i in range(nseeds):
        pps.append(mace_process(ncore_per_task0, config,configfile0,
                                dftsamples, fraction,max_epochs[0],
                                seed=seeds[i])
                   )
    log(f'stage one: {nseeds} seeds, {ncore_per_task0} threads each, '
        f'{max_epochs[0]} epochs, device {device1}')
    for i in pps:
        i.start()
    ok, bad = wait(pps, 'stage one')
    paths, stats = process_results('seed*/results-res/*txt', nlast, use, worst)
    ## a run that died mid training still leaves the eval records of the epochs
    ## it did get through, so it can be ranked and picked like a complete one
    crashed = [i for i in paths if i in set(bad)]
    if crashed:
        log('warning: picked for stage two although stage one did not finish: '
            + ', '.join(crashed))
    ## best
    config['device'] = device2
    pps = []
    for path in paths:
        pps.append(mace_process(nmax//nlast, config, configfile0,
                                dftsamples, fraction,max_epochs[1], path)
                   )
    log(f'stage two: extending {len(paths)} model(s) to {max_epochs[1]} epochs '
        f'with {nmax//nlast} threads each, device {device2}: ' + ', '.join(paths))
    for i in pps:
        i.start()
    wait(pps, 'stage two')
    # remove all the useless stuff
    if clean:
        clean_init('seed*/', paths)
    log('=== multimace done ===')
