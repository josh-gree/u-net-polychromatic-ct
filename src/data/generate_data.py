import pickle
import h5py

from utils import make_make_data
import sys
import os


Np_lim,Nd_lim = int(sys.argv[1]),int(sys.argv[2])
make_data = make_make_data(Np_lim,Nd_lim)

train_path = 'train_{}_{}/'.format(Np_lim,Nd_lim)
val_path = 'val_{}_{}/'.format(Np_lim,Nd_lim)
test_path = 'test_{}_{}/'.format(Np_lim,Nd_lim)

os.mkdir(train_path)
os.mkdir(val_path)
os.mkdir(test_path)


# make train
for i in range(500):
    full,lim = make_data()
    f = h5py.File(train_path + "{}.hdf5".format(i), "w")
    f.create_dataset('lim', data=lim)
    f.create_dataset('full', data=full)
    f.close()

# make validation
for i in range(100):
    full,lim = make_data()
    f = h5py.File(val_path + "{}.hdf5".format(i), "w")
    f.create_dataset('lim', data=lim)
    f.create_dataset('full', data=full)
    f.close()

# make test
for i in range(500):
    full,lim = make_data()
    f = h5py.File(test_path + "{}.hdf5".format(i), "w")
    f.create_dataset('lim', data=lim)
    f.create_dataset('full', data=full)
    f.close()
