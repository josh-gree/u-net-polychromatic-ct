import numpy as np
import h5py

from glob import glob

def open_batch(fs,batch_size):
    
    inp_array = np.zeros((batch_size,256,256,1))
    lab_array = np.zeros((batch_size,256,256,1))

    for k in range(batch_size):
        
        fname = fs[k]
        f = h5py.File(fname, "r")
        inp,lab = f['lim'][:],f['full'][:]
        
        inp = inp[...,np.newaxis]
        lab = lab[...,np.newaxis]
        
        inp_array[k,...] = inp
        lab_array[k,...] = lab
    
    return inp_array,lab_array

def gen(path,batch_size=4):
    
    fnames = np.array(glob(path + '*'))
    N = len(fnames)
    
    if N % batch_size == 0:
        batch_num = N // batch_size
    else:
        batch_num = N // batch_size + 1
        
    while True:
        perm = np.random.permutation(N)
        fnames = fnames[perm]
        
        for k in range(batch_num):
            
            yield open_batch(fnames[k*batch_size:(k+1)*batch_size],batch_size)
