#!/home/ernesto/venvs/base/bin/python3.11
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.lines import Line2D # leyenda custom
import glob, json

plt.style.use('/home/ernesto/.template/plot.config')

def cut_outliers(data, cut):
    d = np.abs(data -np.median(data))
    mdev = np.median(d)
    s= d/mdev
    data[np.where(s>cut)] = np.nan
    return data


def get_seed(words):
    for zz in words:
        if 'seed' in zz:
            return int(zz[5:])
        elif 'best' in zz:
            return int(zz.split('_')[-1])

files = '*/results-res/*txt'
#print(glob.glob(files))


### files = 'seed*/results-res/*txt'
stats, seed = [], []
for f in glob.glob(files):
    fin = open(f, 'r').readlines()
    seed.append(get_seed(f.split('/')))
    res = json.loads(fin[-1])
    stats.append(np.array([res['loss'],res['rmse_e_per_atom']*1000,res['mae_e_per_atom']*1000,
                           res['rmse_f'], res['rel_rmse_f'],
                           res['mae_f'], res['rel_mae_f']]))
stats = np.array(stats)

files_ext = '*/results-res-ext/*txt'
stats_ext, seed_ext = [], []
for f in glob.glob(files_ext):
    fin = open(f, 'r').readlines()
    seed_ext.append(get_seed(f.split('/')))
    res = json.loads(fin[-1])
    stats_ext.append(np.array([res['loss'],res['rmse_e_per_atom']*1000,res['mae_e_per_atom']*1000,
                            res['rmse_f'], res['rel_rmse_f'],
                            res['mae_f'], res['rel_mae_f']]))
stats_ext = np.array(stats_ext)

idx_best = []
for i in seed_ext:
    cc = 0
    for j in seed:
        if i == j:
            idx_best.append(cc)
        cc+=1

## LOSS PLOT
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(6.4*2, 4.8*1), sharex=False,
                       sharey=False, gridspec_kw={})
ax
xx = np.arange(0, stats.shape[0])
ax[0].plot(xx, stats[:, 0], color='blue', marker='x', linestyle='none')
cc=0
for i in idx_best:
    ax[0].scatter(xx[i], stats[i][0], color='red', marker='o', facecolor='none', s=200)
    ax[1].plot(xx[i], stats_ext[cc][0], color='red', marker='o')
    ax[1].text(xx[i]+stats_ext[cc][0]*1.5,stats_ext[cc][0], seed_ext[cc],fontsize=16)
    cc+=1
ax[0].scatter(xx[i], stats[i][0], color='red', marker='o', facecolor='none', s=200, label='Best')
ax[0].legend()
for i in range(2):
    ax[i].set_xticklabels([])
    ax[i].set_xlabel('Random seed')
    ax[i].set_ylabel('loss')
ax[0].set_title('First 10 epoch',fontsize=18)
ax[1].set_title('Best after 200 epoch',fontsize=18)
fig.savefig('loss.png')


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(6.4*2, 4.8*1), sharex=False,
                       sharey=False, gridspec_kw={})
xx = np.arange(0, stats.shape[0])
ax[0].plot(xx, stats[:, 3]*1e3, color='blue', marker='x', linestyle='none',
           label='RMSE F')
ax[0].plot(xx, stats[:, 5]*1e3, color='green', marker='s', linestyle='none',
           label='MAE F')
cc=0
for i in idx_best:
    ax[0].scatter(xx[i], stats[i][3]*1e3, color='red', marker='o',
                  facecolor='none', s=200)
    ax[0].scatter(xx[i], stats[i][5]*1e3, color='red', marker='o',
                  facecolor='none', s=200)
    ax[1].plot(xx[i], stats_ext[cc][3]*1e3, color='blue', marker='x')
    ax[1].plot(xx[i], stats_ext[cc][5]*1e3, color='green', marker='s')
    ax[1].text(xx[i]+stats_ext[cc][5]*0.1,stats_ext[cc][5]*1e3,
               seed_ext[cc],fontsize=16)
    cc+=1

ax[0].scatter(xx[i], stats[i][5]*1e3, color='red', marker='o',
              facecolor='none', s=200, label='Best')
ax[1].plot(xx[i], stats_ext[cc-1][3]*1e3, color='blue', marker='x', label='RMSE')
ax[1].plot(xx[i], stats_ext[cc-1][5]*1e3, color='green', marker='s', label='MAE')

ax[0].legend(frameon=True)
ax[1].legend(frameon=True)
for i in range(2):
    ax[i].set_xticklabels([])
    ax[i].set_ylabel('MAE/RMSE (meV/ang)')
    ax[i].set_xlabel('seed')
ax[0].set_title('First 10 epoch',fontsize=18)
ax[1].set_title('Best after 200 epoch',fontsize=18)
fig.savefig('forces.png')

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(6.4*2, 4.8*1), sharex=False,
                       sharey=False, gridspec_kw={})
xx = np.arange(0, stats.shape[0])
ax[0].plot(xx, stats[:, 1], color='blue', marker='x', linestyle='none',
           label='RMSE F')
ax[0].plot(xx, stats[:, 2], color='green', marker='s', linestyle='none',
           label='MAE F')
cc=0
for i in idx_best:
    ax[0].scatter(xx[i], stats[i][1], color='red', marker='o',
                  facecolor='none', s=200)
    ax[0].scatter(xx[i], stats[i][2], color='red', marker='o',
                  facecolor='none', s=200)
    ax[1].plot(xx[i], stats_ext[cc][1], color='blue', marker='x')
    ax[1].plot(xx[i], stats_ext[cc][2], color='green', marker='s')
    ax[1].text(xx[i]+stats_ext[cc][2]*0.1,stats_ext[cc][2],
               seed_ext[cc],fontsize=16)
    cc+=1

ax[0].scatter(xx[i], stats[i][2], color='red', marker='o',
              facecolor='none', s=200, label='Best')
ax[1].plot(xx[i], stats_ext[cc-1][1], color='blue', marker='x', label='RMSE')
ax[1].plot(xx[i], stats_ext[cc-1][2], color='green', marker='s', label='MAE')

ax[0].legend(frameon=True)
ax[1].legend(frameon=True)
for i in range(2):
    ax[i].set_xticklabels([])
    ax[i].set_ylabel('E MAE/RMSE per atom (meV)')
    ax[i].set_xlabel('seed')
ax[0].set_title('First 10 epoch',fontsize=18)
ax[1].set_title('Best after 200 epoch',fontsize=18)
fig.savefig('energy.png')




### print(f"Best lost: {seed_ext[np.argmin(stats_ext[:, 0])]}")
### print(f"Best rmse_per_atom: {seed_ext[np.argmin(stats_ext[:, 1])]}")
### print(f"Best mae_per_atom: {seed_ext[np.argmin(stats_ext[:, 2])]}")
### print(f"Best rmse_f: {seed_ext[np.argmin(stats_ext[:, 3])]}")
### print(f"Best rel_rmse_f: {seed_ext[np.argmin(stats_ext[:, 4])]}")
### print(f"Best mae_f: {seed_ext[np.argmin(stats_ext[:, 5])]}")
### print(f"Best rel_mae_f: {seed_ext[np.argmin(stats_ext[:, 6])]}")
### 
###     #stats.append(np.array([res['loss'],res['rmse_e_per_atom']*1000,res['mae_e_per_atom']*1000,
###     #                        res['rmse_f'], res['rel_rmse_f'],
###     #                        res['mae_f'], res['rel_mae_f']]))
### 
