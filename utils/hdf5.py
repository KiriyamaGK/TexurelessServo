import h5py
import numpy as np
import os
import random
import json

def delete_hdf5_keys(new_f_out:h5py.File,key,demo_id):
    if key =="actions":
        del new_f_out['data/{}/actions'.format(demo_id)]
    else:
        assert key in new_f_out['data/{}/obs/'.format(demo_id)]
        del new_f_out['data/{}/obs/'.format(demo_id)+key]

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

def add_useless_things(new_f_out:h5py.File,epi_len:int,demo_ind):
    if isinstance(demo_ind,str):
        assert "demo_"in demo_ind
        demo_ind = int(demo_ind[5:])
    new_f_out.create_dataset('data/demo_{}/dones'.format(demo_ind), data=np.zeros((epi_len - 1)))
    new_f_out.create_dataset('data/demo_{}/interventions'.format(demo_ind), data=np.zeros((epi_len, 1)))
    new_f_out.create_dataset('data/demo_{}/policy_acting'.format(demo_ind), data=np.zeros((epi_len)))
    new_f_out.create_dataset('data/demo_{}/rewards'.format(demo_ind), data=np.zeros((epi_len - 1)))
    new_f_out.create_dataset('data/demo_{}/states'.format(demo_ind), data=np.zeros((0)))
    new_f_out.create_dataset('data/demo_{}/user_acting'.format(demo_ind), data=np.zeros((epi_len, 1)))

def delete_useless_things(f:h5py.File,ep):
    del f['data/{}/dones'.format(ep)]
    del f['data/{}/interventions'.format(ep)]
    del f['data/{}/policy_acting'.format(ep)]
    del f['data/{}/rewards'.format(ep)]
    del f['data/{}/states'.format(ep)]
    del f['data/{}/user_acting'.format(ep)]

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

def add_env_meta(new_f_out:h5py.File,additional_itms=None):
    env_meta = {
        "env_name": "Libero_Kitchen_Tabletop_Manipulation",
        "env_version": "1.4.1",
        "type": 1,
        "env_kwargs": {
            "robots": [
                "Panda"
            ],
            "controller_configs": {
                "type": "OSC_POSE",
                "input_max": 1,
                "input_min": -1,
                "output_max": [
                    0.05,
                    0.05,
                    0.05,
                    0.5,
                    0.5,
                    0.5
                ],
                "output_min": [
                    -0.05,
                    -0.05,
                    -0.05,
                    -0.5,
                    -0.5,
                    -0.5
                ],
                "kp": 150,
                "damping_ratio": 1,
                "impedance_mode": "fixed",
                "kp_limits": [
                    0,
                    300
                ],
                "damping_ratio_limits": [
                    0,
                    10
                ],
                "position_limits": None,
                "orientation_limits": None,
                "uncouple_pos_ori": True,
                "control_delta": True,
                "interpolation": None,
                "ramp_ratio": 0.2
            },
            "bddl_file_name": None,
            "reward_shaping": False,
            "camera_names": [
                "agentview",
                "robot0_eye_in_hand"
            ],
            "camera_heights": 84,
            "camera_widths": 84,
            "has_renderer": False,
            "has_offscreen_renderer": True,
            "ignore_done": True,
            "use_object_obs": True,
            "use_camera_obs": True,
            "camera_depths": False,
            "render_gpu_device_id": 0
        }
    }
    if additional_itms is not None:
        for k, v in additional_itms.items():
            env_meta[k] = v
    dat = new_f_out['data']
    dat.attrs['env_args'] = json.dumps(env_meta, indent=4)

def add_config(new_f_out:h5py.File,config:dict):
    dat = new_f_out['data']
    dat.attrs['env_args'] = json.dumps(config, indent=4)

def compute_num_samples(hdf5_path:str):
    total_samples = 0
    f = h5py.File(hdf5_path, "a")  # edit mode
    for ep in f["data"]:
        # add "num_samples" into per-episode metadata
        if "num_samples" in f["data/{}".format(ep)].attrs:
            del f["data/{}".format(ep)].attrs["num_samples"]
        n_sample = f["data/{}/actions".format(ep)].shape[0] - 1
        f["data/{}".format(ep)].attrs["num_samples"] = n_sample
        total_samples += n_sample

        # print("num_samples:",n_sample)
    # add total samples to global metadata
    if "total" in f["data"].attrs:
        del f["data"].attrs["total"]
    f["data"].attrs["total"] = total_samples
    f.close()