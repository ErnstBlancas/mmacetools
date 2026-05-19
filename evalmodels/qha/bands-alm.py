#!/home/ernesto/install/p314/bin/python
import glob
import numpy as np
dfsets = '*/*/*/DFSET*'
kds = ['Al', 'O'] ## correct order
kpoints=f"""1
G 0.0 0.0 0.0 M 0.5 0.0 0.0 101
M 0.5 0.0 0.0 K {1/3} {1/3} 0.0 101 
K {1/3} {1/3} 0.0 G 0 0 0 101
G 0 0 0 A 0 0 0.5 101
A 0 0 0.5 L 0.5 0 0.5 101
L 0.5 0 0.5 H {1/3} {1/3} 0.5 101
H {1/3} {1/3} 0.5 A 0.5 0 0 101"""



opt_alm = """&general
  PREFIX = {0} 
  MODE = optimize
  NAT = 240
  NKD = 2
  KD = Al O
  PRIMCELL=AUTO
/
&optimize
  MAXITER=1000000
  DFSET = {1} 
  ICONST = 11
  CONV_TOL = 1e-10
  LMODEL = 1
/
&interaction
  NORDER = 1  # 1: harmonic, 2: cubic, ..
/
&cutoff
  *-* None
/"""
opt_anphon ="""&general
  PREFIX = {0}
  MODE = phonons
  KD = Al O
  TMIN=0;TMAX=2300;DT=10
  FCSFILE={0}.xml
/
&kpoint
 {1}
/
"""
run = """#!/bin/bash
#SBATCH -J anphon
#SBATCH -o anphon.sout
#SBATCH -e anphon.serr
#SBATCH -N 1
#SBATCH -c 4

export OMP_NUM_THREADS=4

ulimit -s  unlimited

## /home/ernesto/git/alamode/build/alm/alm {0}.alm.in > {0}.alm.out
/home/ernesto/git/alamode/build/anphon/anphon {0}.anphon.in > {0}.anphon.out
rm *dfc
rm *dymat
"""

def read_aims_in(fin):
    angs2born = 1.8897261
    fin = open(fin, 'r')
    cell = []
    pos = []
    symbols = []
    for line in fin:
        if "lattice_vector" in line:
            cell.append(np.array([np.float64(zz) for zz in line.split()[1:]]))
        elif "atom_frac" in line:
            pos.append(np.array([np.float64(zz) for zz in line.split()[1:4]]))
            symbols.append(line.split()[4])
    fin.close()
    cell = np.array(cell)
    pos = np.array(pos)
    return cell*angs2born, pos, symbols

def alm_syms(syms, kds):
    syms = np.array(syms)
    oks = syms.copy()
    cc = 1
    for kd in kds:
        oks[np.where(syms == kd)] = cc
        cc+=1
    return oks

def write_alm(prefix, dfset, path, cell, pos, sym, header):
    with open(f'{path}/{prefix}.alm.in', 'w') as fin:
        print(header.format(prefix, dfset), file=fin)
        print('&cell', file=fin)
        print('1.00', file=fin)
        for i in range(3):
            print("  "+"%8.18f %8.18f %8.18f" % (cell[i][0],  cell[i][1], cell[i][2],), file=fin)
        print('/', file=fin)
        print('&position', file=fin)
        for i in range(len(sym)):
            print("  "+ sym[i] + "  %8.18f %8.18f %8.18f" % (pos[i][0],  pos[i][1], pos[i][2],), file=fin)
        print('/', file=fin)

def write_anphon(prefix, path, cell, header):
    with open(f'{path}/{prefix}.anphon.in', 'w') as fin:
        print(header.format(prefix, dfset), file=fin)
        print('&cell', file=fin)
        print('1.00', file=fin)
        for i in range(3):
            print("  "+"%8.18f %8.18f %8.18f" % (cell[i][0],  cell[i][1], cell[i][2],), file=fin)
        print('/', file=fin)


for f in  glob.glob(dfsets):
    ff  = f.split('/')
    dfset = ff[-1]
    path = "/".join(ff[0:-1])
    fpath = "/".join(ff[0:-2])
    prefix = dfset.split('_')[-1].lower()
    cell, pos, syms = read_aims_in(fpath+'/supercell.in')
    print(path, prefix)
    syms = alm_syms(syms, kds)
    write_alm(prefix, dfset, path, cell, pos, syms, opt_alm)
    cell, pos, syms = read_aims_in(fpath+'/primitive.in')
    write_anphon(prefix, path, cell, opt_anphon.format(prefix, kpoints))
    with open(path+'/'+prefix+'.sub', 'w') as fin:
        print(run.format(prefix), file=fin)
    

    
