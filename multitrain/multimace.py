import os, glob, logging, sys, warnings, time
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
use = 'force'           ## best model: force='rmse_f', energy='rmse_e_per_atom', loss='loss'
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
          'forces_key': 'forces',
          'E0s': "{8:-0.204170677752367E+04, 13:-0.661017373326158E+04}",
          'seed': 101,
          'restart_latest': False,
          }
configfile0 = 'config.yaml' ## config filename
fraction = 0.8          ## dftsources use for training
clean = True            ## remove not valid seeds files
############################################### 

def process_results(path, nlast, use, worst=None):
    import json
    files = glob.glob(path)
    stats = np.zeros((len(files), 3))
    cc = 0
    for f in files:
        fin = open(f, 'r').readlines()
        res = json.loads(fin[-1])
        stats[cc] = np.array([res['loss'],res['rmse_e_per_atom'],res['rmse_f']])
        cc += 1
    if use == 'force':
        best = np.argpartition(stats[:, 2],nlast)
    elif use == 'energy':
        best = np.argpartition(stats[:, 1],nlast)
    elif use =='loss':
        best = np.argpartition(stats[:, 0],nlast)

    if worst:
        paths = [files[best[i]].split('/')[0] for i in range(nlast-1)]
        paths.append(files[best[-1]].split('/')[0])
    else:
        paths = [files[best[i]].split('/')[0] for i in range(nlast)]
    return paths, stats

def clean_init(all_path, best_path):
    for i in glob.glob(all_files):
        if i in path_best:
            continue
        else:
            os.system(f'rm -rf {i}')

class mace_process(Process):
    def __init__(self, ncpu, config, configfile, dft, trainfraction, epochs, dir=None):
        ## init tye process
        super().__init__()
        self.ncpu = ncpu
        self.seed = np.random.randint(0, 10000)
        self.is_extend = False
        self.config = config
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

    def _write_yaml_ext(self):
        fout = open(f'{self.path}/{self.configfile}', 'w')
        for i in self.config:
            if i == "E0s":
                print(f'{i}: "{self.config[i].strip()}"', file=fout)
            else:
                print(f'{i}: {self.config[i].strip()}', file=fout)
        fout.close()

    def _shuffle(self):
        os.system(f' rm {self.path}/*xyz 2>/dev/null')
        n = len(self.dftsources)
        ntrain = int(n*self.fraction)
        rs = np.random.RandomState(self.seed)
        rs.shuffle(self.dftsources)
        for i in range(ntrain):
            write(f'tmp-{self.seed}.xyz', self.dftsources[i])
            os.system(f'cat tmp-{self.seed}.xyz >> {self.path}/{self.config['train_file']}')
        for i in range(ntrain,n):
            write(f'tmp-{self.seed}.xyz', self.dftsources[i])
            os.system(f'cat tmp-{self.seed}.xyz >> {self.path}/{self.config['test_file']}')
        os.remove(f'tmp-{self.seed}.xyz')

    def gen_path(self):
        self.path = f'seed_{self.seed}'
        if os.path.isdir(self.path):
            os.system(f'rm -f {path}')
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
        self.seed = int(config['seed'])
        self.config['restart_latest'] = True
        self.config['log_dir'] = self.config['log_dir']+'ext'
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
    dftsamples = []
    for f in files:
        for s in read(f, ':'):
            dftsamples.append(s)
    ## init
    pps = []
    for i in range(nmax//ncore_per_task0):
        pps.append(mace_process(ncore_per_task0, config,configfile0,
                                dftsamples, fraction,max_epochs[0])
                   ) 
    for i in pps:
        i.start()
    for i in pps:
        i.join()
    paths, stats = process_results('seed*/results-res/*txt', nlast, use, True)
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
        clean_init('seed*/', paths):
