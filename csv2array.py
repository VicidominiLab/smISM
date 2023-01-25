import numpy as np
from numpy import genfromtxt

def csv2array(file, dlmt='\t'):
    data = genfromtxt(file, delimiter=dlmt)
    return data


def array2csv(data, fname='test.csv', dlmt='\t', dtype=float):
    data = np.asarray(data, dtype)
    if dtype == int:
        np.savetxt(fname, data, delimiter=dlmt, fmt='%i')
    else:
        np.savetxt(fname, data, delimiter=dlmt)