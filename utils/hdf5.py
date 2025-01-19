import h5py
import numpy as np
import os
import random

def copy_attributes(source, target):
    """Copy attributes from source to target"""
    for key, value in source.attrs.items():
        target.attrs[key] = value
def create_hdf5_filter_key(hdf5_path, demo_keys, key_name):
    """
    Creates a new hdf5 filter key in hdf5 file @hdf5_path with
    name @key_name that corresponds to the demonstrations
    @demo_keys. Filter keys are generally useful to create
    named subsets of the demonstrations in an hdf5, making it
    easy to train, test, or report statistics on a subset of
    the trajectories in a file.

    Returns the list of episode lengths that correspond to the filtering.

    Args:
        hdf5_path (str): path to hdf5 file
        demo_keys ([str]): list of demonstration keys which should
            correspond to this filter key. For example, ["demo_0",
            "demo_1"].
        key_name (str): name of filter key to create

    Returns:
        ep_lengths ([int]): list of episode lengths that corresponds to
            each demonstration in the new filter key
    """
    f = h5py.File(hdf5_path, "a")
    demos = sorted(list(f["data"].keys()))

    # collect episode lengths for the keys of interest
    ep_lengths = []
    for ep in demos:
        ep_data_grp = f["data/{}".format(ep)]
        if ep in demo_keys:
            ep_lengths.append(ep_data_grp.attrs["num_samples"])

    # store list of filtered keys under mask group
    k = "mask/{}".format(key_name)
    if k in f:
        del f[k]
    f[k] = np.array(demo_keys, dtype='S')

    f.close()
    return ep_lengths

def split_train_val_from_hdf5(hdf5_path, val_ratio=0.1, filter_key=None):
    """
    Splits data into training set and validation set from HDF5 file.

    Args:
        hdf5_path (str): path to the hdf5 file
            to load the transitions from

        val_ratio (float): ratio of validation demonstrations to all demonstrations

        filter_key (str): if provided, split the subset of demonstration keys stored
            under mask/@filter_key instead of the full set of demonstrations
    """

    # retrieve number of demos
    f = h5py.File(hdf5_path, "r")
    if filter_key is not None:
        print("using filter key: {}".format(filter_key))
        demos = sorted([elem.decode("utf-8") for elem in np.array(f["mask/{}".format(filter_key)])])
    else:
        demos = sorted(list(f["data"].keys()))
    num_demos = len(demos)
    f.close()

    # get random split
    num_demos = len(demos)
    num_val = int(val_ratio * num_demos)
    mask = np.zeros(num_demos)
    mask[:num_val] = 1.
    np.random.shuffle(mask)
    mask = mask.astype(int)
    train_inds = (1 - mask).nonzero()[0]
    valid_inds = mask.nonzero()[0]
    train_keys = [demos[i] for i in train_inds]
    valid_keys = [demos[i] for i in valid_inds]
    print("{} validation demonstrations out of {} total demonstrations.".format(num_val, num_demos))

    # pass mask to generate split
    name_1 = "train"
    name_2 = "valid"
    if filter_key is not None:
        name_1 = "{}_{}".format(filter_key, name_1)
        name_2 = "{}_{}".format(filter_key, name_2)

    train_lengths = create_hdf5_filter_key(hdf5_path=hdf5_path, demo_keys=train_keys, key_name=name_1)
    valid_lengths = create_hdf5_filter_key(hdf5_path=hdf5_path, demo_keys=valid_keys, key_name=name_2)

    print("Total number of train samples: {}".format(np.sum(train_lengths)))
    print("Average number of train samples {}".format(np.mean(train_lengths)))

    print("Total number of valid samples: {}".format(np.sum(valid_lengths)))
    print("Average number of valid samples {}".format(np.mean(valid_lengths)))