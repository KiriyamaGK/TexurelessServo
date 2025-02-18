import collections
import json
from typing import OrderedDict
from utils.input_process import input_dict_preprocess
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split
from tqdm import tqdm

class HDF5Dataset(Dataset):
    def __init__(self, hdf5_path, seq_length: int = 1, specific_obs_keys: list = None, bgr2rgb: bool = False, num_demos: int = None):
        """
        Dataset class for loading demonstrations from a hdf5 file.
        :param hdf5_path: path to the hdf5 file containing the demonstrations
        :param seq_length: number of observations to be used for each state
        :param specific_obs_keys: list of specific observation keys to use
        :param bgr2rgb: whether to convert BGR images to RGB
        :param num_demos: number of demonstrations to load
        """
        assert seq_length > 0, "seq_length must be greater than 0"

        self.seq_length = seq_length
        self.bgr2rgb = bgr2rgb
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hdf5_path = hdf5_path
        self.file=None

        with h5py.File(self.hdf5_path, "r") as f:
            episode_keys = list(f["data"].keys())
            self.num_demos = min(num_demos, len(episode_keys))

            mask = [int(episode_keys[i][5:]) for i in range(len(episode_keys))]
            sorted_pairs = sorted(zip(mask, episode_keys))
            self.episode_keys = [pair[1] for pair in sorted_pairs][0:self.num_demos] #assert sequence of episode_keys is right

            self.episode_lens = [len(f["data/{}/actions".format(ep)]) for ep in self.episode_keys]

        self._specific_obs_keys = specific_obs_keys if specific_obs_keys is not None else []
        self.specific_obs_keys = list(set(self._specific_obs_keys))

    def __len__(self):
        return sum(self.episode_lens)

    def __getitem__(self, idx):
        """
        Get the idx-th state in the dataset
        :param idx: index for the item
        :return: state, action
        """
        if self.file is None:
            self.file = h5py.File(self.hdf5_path, "r")

        # find the episode that idx belongs to
        ep_idx = 0
        while idx >= self.episode_lens[ep_idx]:
            idx -= self.episode_lens[ep_idx]
            ep_idx += 1

        #instead of using:with h5py.File... as f /f=h5py.File... to avoid frequently opening hdf_file
        ep_key = self.episode_keys[ep_idx]
        actions = self.file["data/{}/actions".format(ep_key)]
        observations = {key: self.file["data/{}/obs/{}".format(ep_key, key)] for key in self.specific_obs_keys}

        obs_dict = {}
        start_idx = max(0, idx - self.seq_length + 1)
        end_idx = idx + 1
        for k in self.specific_obs_keys:
            obs_dict[k] = observations[k][start_idx:end_idx]  # [len, h, w, c] or [len,]
            if obs_dict[k].shape[0] < self.seq_length:
                if np.isscalar(obs_dict[k][0]):
                    pad_obs = np.array([obs_dict[k][0]])
                else:
                    pad_obs = obs_dict[k][0].copy()[np.newaxis, :]
                for i in range(self.seq_length - obs_dict[k].shape[0]):
                    obs_dict[k] = np.concatenate((pad_obs, obs_dict[k]), axis=0)
            assert obs_dict[k].shape[0] == self.seq_length
        obs_dict = input_dict_preprocess(obs_dict, bgr2rgb=self.bgr2rgb)  # change format to tensor, rgb,.....

        # get the action_length actions after idx
        actions = actions[start_idx:end_idx]  # [len,]
        if actions.shape[0] < self.seq_length:
            pad_act = actions[0].copy()[np.newaxis, :]
            for j in range(self.seq_length - actions.shape[0]):
                actions = np.concatenate((pad_act, actions), axis=0)
        assert actions.shape[0] == self.seq_length
        actions = torch.from_numpy(actions).type(torch.float32).to(self.device)  # change format to tensor and to device

        return {
            "observations": obs_dict,
            "actions": actions
        }

    # get indexes for train and test rows
    def get_splits(self, split_ratio: float = 0.8):
        train_size = int(split_ratio * len(self))
        test_size = len(self) - train_size
        # calculate the split
        return random_split(self, [train_size, test_size])