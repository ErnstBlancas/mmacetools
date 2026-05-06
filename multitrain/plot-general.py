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
    print(cut)
    data[np.where(s>cut)] = np.nan
    return data

files = '*/results-res/*txt'
stats, seed = [], []
for f in glob.glob(files):
    fin = open(f, 'r').readlines()
    seed.append(int(f.split('/')[0][5:]))
    res = json.loads(fin[-1])
    stats.append(np.array([res['loss'],res['rmse_e_per_atom']*1000,res['mae_e_per_atom']*1000,
                            res['rmse_f'], res['rel_rmse_f'],
                            res['mae_f'], res['rel_mae_f']]))
stats = np.array(stats)
files_ext = '*/results-res-ext/*txt'
stats_ext, seed_ext = [], []
for f in glob.glob(files_ext):
    fin = open(f, 'r').readlines()
    seed_ext.append(int(f.split('/')[0][5:]))
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

xx = np.arange(0, stats.shape[0])

## loss plot
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(6.4*2, 4.8*1), sharex=False,
                       sharey=False, gridspec_kw={})


ax[0].plot(xx, cut_outliers(stats[:, 0], 20), color='blue', marker='.', linestyle='dashed')
cc=0
for i in idx_best:
    ax[0].plot(xx[i], stats[i][0], color='red', marker='o')
    ax[1].plot(xx[i], stats_ext[cc][0], color='red', marker='o')
    ax[1].text(xx[i]+2,stats_ext[cc][0], seed_ext[cc])
    cc+=1

for i in range(2):
    ax[i].set_xticklabels([])
    ax[i].set_xlabel('Random seed')
    ax[i].set_ylabel('loss')
    ax[i].set_xlim([0, len(xx)])
ll = []
ll.append(Line2D([0], [0], color='blue', label='loss', linestyle='dashed', marker='.'))
ll.append(Line2D([0], [0], color='red', label='loss best', linestyle='none', marker='o'))
ax[0].legend(handles=ll)
ll = []
ll.append(Line2D([0], [0], color='red', label='loss ext', linestyle='none', marker='o'))
ax[1].legend(handles=ll)
fig.savefig('loss.png')


# energy plot
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(6.4*2, 4.8*1), sharex=False,
                       sharey=False, gridspec_kw={})

ax[0].plot(xx, cut_outliers(stats[:, 1], 200), color='blue', marker='.', linestyle='none')
ax[0].plot(xx, cut_outliers(stats[:, 2], 200), color='red', marker='x', linestyle='none')
xlim = ax[0].get_xlim()
cc=0
for i in idx_best:
    ax[0].plot(xx[i], stats[i][1], color='green', marker='o')
    ax[0].plot(xx[i], stats[i][2], color='green', marker='d')
    ax[1].plot(xx[i], stats_ext[cc][1], color='blue', marker='.')
    ax[1].text(xx[i]+2,stats_ext[cc][1], seed_ext[cc])
    ax[1].plot(xx[i], stats_ext[cc][2], color='red', marker='x')
    cc+=1
for i in range(2):
    ax[i].set_xticklabels([])
    ax[i].set_xlabel('Random seed')
    ax[i].set_ylabel('RMSE/MAE E per atom (meV)')
    ax[i].set_xlim(xlim)

ll1 = []
ll1.append(Line2D([0], [0], color='blue', label='RMSE', linestyle='none', marker='.'))
ll1.append(Line2D([0], [0], color='blue', label='MAE', linestyle='none', marker='x'))
ll1.append(Line2D([0], [0], color='green', label='RMSE best', linestyle='none', marker='o'))
ll1.append(Line2D([0], [0], color='green', label='MAE best', linestyle='none', marker='d'))
ax[0].legend(handles=ll1)
ll2 = []
ll2.append(Line2D([0], [0], color='blue', label='RMSE ext', linestyle='none', marker='.'))
ll2.append(Line2D([0], [0], color='red', label='MAE ext', linestyle='none', marker='x'))
ax[1].legend(handles=ll2)
fig.savefig('energy.png')



##forces
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(6.4*2, 4.8*1), sharex=False,
                       sharey=False, gridspec_kw={})

ax[0].plot(xx, cut_outliers(stats[:, 3], 20), color='blue', marker='.', linestyle='none')
ax[0].plot(xx, cut_outliers(stats[:, 5], 20), color='red', marker='x', linestyle='none')
xlim = ax[0].get_xlim()
cc=0
for i in idx_best:
    ax[0].plot(xx[i], stats[i][3], color='green', marker='o')
    ax[0].plot(xx[i], stats[i][5], color='green', marker='d')
    ax[1].plot(xx[i], stats_ext[cc][3]*1000, color='blue', marker='.')
    ax[1].text(xx[i]+2,stats_ext[cc][3]*1000, seed_ext[cc])
    ax[1].plot(xx[i], stats_ext[cc][5]*1000, color='red', marker='x')
    cc+=1
for i in range(2):
    ax[i].set_xticklabels([])
    ax[i].set_xlabel('Random seed')
    ax[i].set_ylabel('RMSE/MAE F (eV/ang)')
    ax[i].set_xlim(xlim)
#
ax[i].set_ylabel('RMSE/MAE F (meV/ang)')
ax[0].legend(handles=ll1)
ax[1].legend(handles=ll2)
fig.savefig('forces.png')

# forces rel
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(6.4*2, 4.8*1), sharex=False,
                       sharey=False, gridspec_kw={})

ax[0].plot(xx, cut_outliers(stats[:, 4], 20), color='blue', marker='.', linestyle='none')
ax[0].plot(xx, cut_outliers(stats[:, 6], 20), color='red', marker='x', linestyle='none')
xlim = ax[0].get_xlim()
cc=0
for i in idx_best:
    ax[0].plot(xx[i], stats[i][4], color='green', marker='o')
    ax[0].plot(xx[i], stats[i][6], color='green', marker='d')
    ax[1].plot(xx[i], stats_ext[cc][4], color='blue', marker='.')
    ax[1].text(xx[i]+2,stats_ext[cc][4], seed_ext[cc])
    ax[1].plot(xx[i], stats_ext[cc][6], color='red', marker='x')
    cc+=1
for i in range(2):
    ax[i].set_xticklabels([])
    ax[i].set_xlabel('Random seed')
    ax[i].set_ylabel('Rel error F (%)')
    ax[i].set_xlim(xlim)
#
ax[i].set_ylabel('Rel error F (%)')
ax[0].legend(handles=ll1)
ax[1].legend(handles=ll2)
fig.savefig('rel-forces.png')
