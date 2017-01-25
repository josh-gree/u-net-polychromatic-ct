import pickle
import h5py

from utils import make_data

train_path = 'train/'
val_path = 'val/'
test_path = 'test/'

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


