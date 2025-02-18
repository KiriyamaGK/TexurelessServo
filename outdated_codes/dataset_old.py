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
    def __init__(self,
             hdf5_path,
             seq_length : int = 1,
             specific_obs_keys : list = None,
             bgr2rgb: bool = False,
             num_demos:int = None,
         ):
        """
        Dataset class for loading demonstrations from a hdf5 file.
        :param hdf5_path: path to the hdf5 file containing the demonstrations
        :param observation_length: number of observations to be used for each state
        :param action_length: number of actions to be used for each state
        :param specific_obs_keys: list of specific observation keys to use,
            this is not compatible with state or image observations being used
        :return:
        """
        assert seq_length > 0, "observation_length must be greater than 0"


        self.seq_length = seq_length
        self.bgr2rgb = bgr2rgb
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        with h5py.File(hdf5_path, "r") as f:

            # list of all demonstrations episodes
            episode_keys = list(f["data"].keys())
            self.num_demos = min(num_demos, len(episode_keys))

            mask=[int(episode_keys[i][5:]) for i in range(len(episode_keys))]
            sorted_pairs = sorted(zip(mask, episode_keys))
            self.episode_keys = [pair[1] for pair in sorted_pairs][0:self.num_demos]  #assert sequence of episode_keys is right

            self.episode_lens = [len(f["data/{}/actions".format(ep)]) for ep in self.episode_keys]
            # self.actions = [f["data/{}/actions".format(ep)][()] for ep in self.episode_keys][:]
            # self.observations = {ep: {key: f["data/{}/obs/{}".format(ep, key)][()] for key in f["data/{}/obs".format(ep)].keys()} for ep in self.episode_keys}
            self.actions = []
            self.observations = {}
            for ep in tqdm(self.episode_keys, desc="Loading episodes"):
                self.actions.append(f["data/{}/actions".format(ep)][()])
                self.observations[ep] = {key: f["data/{}/obs/{}".format(ep, key)][()] for key in
                                         f["data/{}/obs".format(ep)].keys()}
            f.close()

        self._specific_obs_keys = specific_obs_keys if specific_obs_keys is not None else []
        # make list of specific observation keys unique
        self.specific_obs_keys = list(set(self._specific_obs_keys))

    def __len__(self):
        return sum(self.episode_lens)

    def __getitem__(self, idx):
        """
        Get the idx-th state in the dataset
        :param idx: index for the item
        :return: state, action
        """
        # find the episode that idx belongs to
        ep_idx = 0
        while idx >= self.episode_lens[ep_idx]:
            idx -= self.episode_lens[ep_idx]
            ep_idx += 1

        obs_dict = {}
        for k in self.specific_obs_keys:
            str_idx='demo_'+str(ep_idx)
            assert isinstance(self.observations[str_idx][k], np.ndarray)
            # get the observation_length states before idx
            start_idx = max(0, idx - self.seq_length + 1)
            obs_dict[k] = self.observations[str_idx][k][start_idx:idx+1] #[len,h,w,c] or [len,]
            # pad with first state if necessary
            if obs_dict[k].shape[0] < self.seq_length:
                if np.isscalar(obs_dict[k][0]):
                    pad_obs = np.array([obs_dict[k][0]])
                else:
                    pad_obs = obs_dict[k][0].copy()[np.newaxis, :]
                for i in range(self.seq_length - obs_dict[k].shape[0]):
                    obs_dict[k] =  np.concatenate((pad_obs, obs_dict[k]), axis=0)
            assert obs_dict[k].shape[0] == self.seq_length
        obs_dict= input_dict_preprocess(obs_dict,bgr2rgb=self.bgr2rgb) #change format to tensor,rgb,.....

        # get the action_length actions after idx
        start_idx = max(0, idx - self.seq_length + 1)
        assert isinstance(self.actions[ep_idx], np.ndarray)
        actions = self.actions[ep_idx][start_idx:idx+1]  #[len,] 注：a[：，0:1】保留了最后一维的维度，但是a[:,0]没有
        if actions.shape[0] < self.seq_length:
            pad_act=actions[0].copy()[np.newaxis, :]
            for j in range(self.seq_length - actions.shape[0]):
                actions = np.concatenate((pad_act,actions), axis=0)
        assert actions.shape[0] == self.seq_length
        actions = torch.from_numpy(actions).type(torch.float32).to(self.device) #change format to tensor and to device

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