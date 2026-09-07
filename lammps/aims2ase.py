import numpy as np

def check_first(line):
    if '#' in line.split()[0]:
        return False
    else:
        return True

def read_aims_in(fin, flabel):
    fin = open(fin, 'r')
    cell = []
    pos = []
    atoms = []
    fidx = []
    cc = 0
    isatom = False
    for line in fin:
        if check_first(line):
            if "lattice_vector" in line:
                cell.append(np.array([np.float64(zz) for zz in line.split()[1:]]))
            elif "atom_frac" == line.split()[0]:
                pos.append(np.array([np.float64(zz) for zz in line.split()[1:4]]))
                atoms.append(line.split()[4])
                if flabel in line:
                    fidx.append(cc)
                cc += 1
            elif "atom" == line.split()[0]:
                isatom=True 
                pos.append(np.array([np.float64(zz) for zz in line.split()[1:4]]))
                atoms.append(line.split()[4])
                cc+=1
    fin.close()
    cell = np.array(cell)
    pos = np.array(pos)
    if isatom:
        pos = np.array([np.linalg.inv(cell.T)@zz for zz in pos])
    return cell, pos, atoms, fidx

cell, pos, atoms, _ = read_aims_in('super.in', 'shit')

#def parse_lammps_format(fout, cell, atoms):
#    fout = open(fout, 'w')
#    n = len(atoms)
#
#
#
#    fout.write("## Lammps data file XRA by Erni\n")
#    fout.write("## Lammps data file XRA by Erni\n")
