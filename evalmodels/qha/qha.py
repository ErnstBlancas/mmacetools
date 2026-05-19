#!/home/ernesto/install/p314/bin/python
from subprocess import check_output, run
import numpy as np
import glob, os

################### Input ###################
remove = True
plot = True
dft_energies = '*/0-relax/aims.out'
qha_dft = '9-eval/qha-2x-prim/full-dft.dos'
qha_line = """SET IGNORE_NEG_CUTOFF 999 \nroot {1} \nmm 203.923\nvfree 10\npressure 0\ntemperature 0 50 2300\nphase qha z 1 file {0} units volume ang energy ev freq cm-1 tmodel qha"""
qha_mls = ['9-eval/qha-2x-prim/iter1.dos',
           '9-eval/qha-2x-stand/iter1.dos',
           '9-eval/qha-2x-prim/iter2.dos',
           '9-eval/qha-2x-stand/iter2.dos'
           ]

qha_labels = ['iter1-2x-prim', 'iter1-2x-stand',
              'iter2-2x-prim', 'iter2-2x-stand' ]

ml_energies = ['iter1-force.out', 'iter1-force-std.out', 'iter2-forces.out',
               'iter2-forces-stand.out']
#############################################
fout = open('evdft.tmp', 'w')
for f in glob.glob(dft_energies):
    line = f"grep -ri 'uncorrected' {f} " + "| tail -1  | awk '{printf \"%.10f\", $6}'"
    e = np.float64(check_output(line, shell=True).decode('utf8'))
    line = f"grep -ri 'Unit cell volume' {f} " + "| tail -1  | awk '{printf \"%.10f\", $6}'"
    v = np.float64(check_output(line, shell=True).decode('utf8'))
    path = f.split('/')[0]
    print(v, e, path+'/'+qha_dft, file=fout)
fout.close()

os.system(f"printf \"{qha_line.format('evdft.tmp', 'qha-dft')}\" | gibbs2 > kk")

for i in range(len(qha_mls)):
    print(qha_mls[i])
    fout = open(qha_labels[i]+'.dat', 'w')
    for f in glob.glob(dft_energies):
        ff = "/".join(f.split('/')[0:2]) +"/"+ml_energies[i]
        e = np.genfromtxt(ff)
        path = f.split('/')[0]+"/"+qha_mls[i]
        print(f'{e[0]} {e[1]} {path}', file=fout)
    fout.close()
    print(qha_line.format(f'{qha_labels[i]}.dat', qha_labels[i]))
    os.system(f"printf \"{qha_line.format(f'{qha_labels[i]}.dat', qha_labels[i])}\" | gibbs2 > kk")
#
if plot:
    import matplotlib.pyplot as plt
    plt.style.use('/home/ernesto/.template/plot.config')
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(6.4*2, 4.8*1),
                           sharex=False, sharey=False, gridspec_kw={})
    qha_dft = np.genfromtxt('qha-dft.eos')
    ax[0].plot(qha_dft[:, 1], qha_dft[:, 2], color='black', label='FULL DFT',linestyle='dashed')
    ax[0].set_xlabel('T (K)')
    ax[0].set_ylabel('V (bohr^3)')
    ax[1].plot(qha_dft[:, 1], qha_dft[:, 15], color='black', label='FULL DFT',linestyle='dashed')
    ax[1].set_xlabel('T (K)')
    ax[1].set_ylabel('alpha (10^-5/K)')
    for iter in qha_labels:
        qha_ml= np.genfromtxt(f'{iter}.eos')
        ax[0].plot(qha_ml[:, 1], qha_ml[:, 2], label={iter})
        ax[1].plot(qha_ml[:, 1], qha_ml[:, 15], label={iter})


    ax[0].legend()
    ax[1].legend()
    fig.savefig('plotqha.png')
    fig.savefig('betterqha.pdf')
    os.system('kitten icat plotqha.png')

if remove:
    os.remove('evdft.tmp')
    os.system('rm qha-* kk')
    for qha_label in qha_labels:
        os.system(f'rm {qha_label}*')
