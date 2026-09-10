import os, glob, logging, random, shutil, sys, warnings
from multiprocessing import Process
from ase.io import read, write
import numpy as np
warnings.filterwarnings("ignore")

#################### input ####################
dftsources = '*.xyz'    ## dftxyz files to shuffle again
nmax = 256               ## max number of core
ncore_per_task0 = 2     ## to run nmax//ncore_per_task0 initial seeds
nlast = 8               ## run nmax//nlast best models with more epochs
max_epochs = [2, 20]   ## max_epochs for the best seeds
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
          'device': 'cpu',
          'r_max': 6.0, 
          'energy_key': 'energy_dft',
          'forces_key': 'forces_dft',
          'E0s': "{8:-0.204170677752367E+04, 13:-0.661017373326158E+04}",
          'seed': 101,
          'restart_latest': False,
          }
configfile0 = 'config.yaml' ## config filename
fraction = 0.8          ## dftsources use for training
clean = True            ## remove not valid seeds files
############################################### 

## selection criterion -> metric key logged in results-*.txt.  'stress' is only
## reported when mace computes it, i.e. with loss stress/huber/universal
metrics = {'force': 'rmse_f',
           'energy': 'rmse_e_per_atom',
           'loss': 'loss',
           'stress': 'rmse_stress'}

def process_results(path, nlast, use, worst=None):
    import json
    if use not in metrics:
        raise ValueError(f"unknown criterion '{use}', use one of {sorted(metrics)}")
    key = metrics[use]
    files, values = [], []
    for f in sorted(glob.glob(path)):
        ## results-*.txt interleaves mode="opt" records, which carry only the
        ## training loss, with the mode="eval" ones holding the rmse keys
        fin = open(f, 'r')
        evals = [r for r in (json.loads(l) for l in fin)
                 if r.get('mode') == 'eval' and r.get(key) is not None]
        fin.close()
        if not evals:
            continue
        files.append(f)
        values.append(evals[-1][key])
    if len(files) < nlast:
        raise RuntimeError(f'only {len(files)} of the seeds produced usable '
                           f'results, {nlast} needed')
    values = np.array(values)
    ## argsort, not argpartition: the latter only places element nlast, so its
    ## last index is not the worst model
    order = np.argsort(values)
    if worst:
        best = list(order[:nlast-1]) + [order[-1]]
    else:
        best = list(order[:nlast])
    paths = [files[i].split(os.sep)[0] for i in best]
    return paths, values

def clean_init(all_path, best_path):
    keep = {os.path.normpath(p) for p in best_path}
    for i in glob.glob(all_path):
        if os.path.normpath(i) in keep:
            continue
        else:
            shutil.rmtree(i, ignore_errors=True)

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

    def _shuffle(self):
        n = len(self.dftsources)
        ntrain = int(n*self.fraction)
        rs = np.random.RandomState(self.seed)
        idx = rs.permutation(n)
        train_file = self.config['train_file']
        test_file = self.config['test_file']
        write(f'{self.path}/{train_file}',
              [self.dftsources[i] for i in idx[:ntrain]])
        write(f'{self.path}/{test_file}',
              [self.dftsources[i] for i in idx[ntrain:]])

    def gen_path(self):
        self.path = f'seed_{self.seed}'
        shutil.rmtree(self.path, ignore_errors=True)
        os.mkdir(self.path)
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

    def finetune(self):
        from mace.cli.run_train import main as mace_run_train_main
        import torch
        torch.set_num_threads(self.ncpu)
        logging.getLogger().handlers.clear()
        sys.argv = ["program", "--config", self.configfile]
        mace_run_train_main()

    def run(self):
        if self.is_extend:
            self.load_config()
        if not self.is_extend:
            self.gen_path()
        os.chdir(self.path)
        self.finetune()

if __name__ == "__main__":
    # no parallel
    if type(dftsources) == str:
        files = glob.glob(dftsources)
    else:
        files = dftsources
    dftsamples = []
    for f in files:
        for s in read(f, ':'):
            dftsamples.append(s)
    ## init
    nseeds = nmax//ncore_per_task0
    ## drawn without replacement: a repeated seed means a shared seed_* directory
    seeds = random.sample(range(2**31-1), nseeds)
    pps = []
    for i in range(nseeds):
        pps.append(mace_process(ncore_per_task0, config,configfile0,
                                dftsamples, fraction,max_epochs[0],
                                seed=seeds[i])
                   ) 
    for i in pps:
        i.start()
    for i in pps:
        i.join()
    paths, stats = process_results('seed*/results-res/*txt', nlast, use, worst)
    ## best
    pps = []
    for path in paths:
        pps.append(mace_process(nmax//nlast, config, configfile0,
                                dftsamples, fraction,max_epochs[1], path)
                   )
    for i in pps:
        i.start()
    for i in pps:
        i.join()
    # remove all the useless stuff
    if clean:
        clean_init('seed*/', paths)
