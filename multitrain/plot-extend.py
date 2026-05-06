#!/home/ernesto/venvs/base/bin/python3.11
from matplotlib.lines import Line2D # leyenda custom
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import glob, json

plt.style.use('/home/ernesto/.template/plot.config')

def cut_outliers(data, cut):
    d = np.abs(data -np.median(data))
    mdev = np.median(d)
    s= d/mdev
    print(cut)
    data[np.where(s>cut)] = np.nan
    return data

def read_full(fin):
    data = open(fin ,'r')
    nice = []
    for line in data:
        if len(line) >= 100:
            l = json.loads(line)
            nice.append([
                l['epoch'],l['loss'],l['mae_e_per_atom']*1000,l['rmse_e_per_atom']*1000,
                l['mae_f']*1000,l['rmse_f']*1000,l['rel_mae_f'],l['rel_rmse_f'],l['time']
                ])
    data.close()
    nice[0][0] = 0
    return np.array(nice)


def plot_cmap(data, fout, ylabel, norm):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6.4*1, 4.8*1),
                           sharex=False, sharey=False, gridspec_kw={})
    x = np.arange(0, data.shape[1])
    y = np.arange(0, data.shape[0])
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6.4*1*2, 4.8*1*2),
                           sharex=False, sharey=False, gridspec_kw={})
    img = ax.pcolormesh(x, y, data, norm=norm)
    fig.colorbar(img, ax=ax)
    ax.set_yticks(y, seed)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    ax.set_xticks(x, x)
    fig.savefig(fout)

seed = []
megadata = []
loss = []
for f in glob.glob('*/results-res-ext/*.txt'):
    data = read_full(f)
    seed.append(f.split('/')[0][5:])
    megadata.append(data)
    loss.append(data[:, 1])

megadata = np.array(megadata)
loss = np.array(loss)
#megadata[:,:, 1]

labels = ['loss','mae_e_per_atom','rmse_e_per_atom',
          'mae_f','rmse_f','rel_mae_f','rel_rmse_f','time']

norms = [colors.LogNorm(vmin=loss.min(), vmax=loss.max()*0.001), 
         colors.LogNorm(vmin=loss.min(), vmax=loss.max()*0.1),
         colors.LogNorm(vmin=loss.min(), vmax=loss.max()*0.1),
         'log', 'log', 'log', 'log', None]

ylabels = ['loss','mae_e (meV/atom)','rmse_e (mev/atom)',
          'mae_f (meV/ang)','rmse_f (meV/ang)','rel_mae_f (%)','rel_rmse_f (%)','time (s)']

for i in range(8):
    print(labels[i])
    plot_cmap(megadata[:,:, i+1], f'{labels[i]}-extend.png', ylabels[i], norms[i])
    print(megadata[:, :, i+1])

    
