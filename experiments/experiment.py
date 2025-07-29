import shutil

from algo.bc import BehaviorCloning
from networks.helpers import get_loss_fn, get_optimizer_cls, get_network_cls
import os
import time
import logging
import yaml
import torch
import random
import numpy as np
import json
from torch.utils.data import DataLoader
from dataset.dataset import dataset_factory
from utils.paths import path_completion, TRAINED_MODELS_DIR, LOG_DIR, PROJECT_ROOT_DIR


class BehaviorCloningExperiment():
    """
    Experiment class for behavior cloning.
    """

    def __init__(self, config_path: str):
        """
        Base class for all experiments.

        Requires the following methods to be implemented:
            - _setup_algorithm

        :param config_path: path to the configs file
        """
        self._config, self.config_path = self._load_config(config_path)
        self._setup_seed()
        self._setup_device()
        self._setup_dataset()
        self._setup_paths()

        self.alg = self._setup_algorithm()

        self._setup_logging()

    def run(self):  
        """
        Run the experiment.
        """
        self.alg.train_loop(
            self._train_loader,
            self._eval_loader,
            self._model_out_dir,
            num_epochs=self._config["training"]["num_epochs"],
            num_epochs_logging=self._config["training"]["num_epochs_logging_and_valid"],
            num_epochs_save=self._config["training"]["num_epochs_save"],
            num_train_steps=self._config["training"]["num_train_steps_per_epoch"],
            num_val_steps=self._config["training"]["num_val_steps_per_epoch"],
            logger=self._logger,
        )

    @staticmethod
    def _load_config(config_path: str) -> tuple:
        """
        Load the configuration file.
        :param config_path: path to the configs file
        :return: configuration dictionary
        """
        config_path = path_completion(config_path, os.path.join(PROJECT_ROOT_DIR, "configs"))  # 根目录/il_path
        with open(config_path, "r") as f:
            config = json.load(f)
        return config, config_path

    def _setup_seed(self):
        """
        Set up the seed for reproducibility.
        """
        seed = self._config["seed"]
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def _setup_device(self):
        """
        Set up the device for training.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _setup_paths(self):
        """
        Set up the paths for saving models.
        """
        self._model_out_dir = path_completion(self._config["model_out_dir"], TRAINED_MODELS_DIR)  # 根目录/trained_models
        self._log_dir = path_completion(self._config["log_dir"], LOG_DIR)  # 根目录/log_dir

        # add timestamp in format YYYY-MM-DD_HH-MM-SS
        self._timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self._model_out_dir = os.path.join(self._model_out_dir, self._timestamp)
        self._log_dir = os.path.join(self._log_dir, self._timestamp)
        self._log_path = os.path.join(self._log_dir, "log.log")

        os.makedirs(self._model_out_dir, exist_ok=False)
        os.makedirs(self._log_dir, exist_ok=False)
        shutil.copy(self.config_path, os.path.join(self._model_out_dir, "config.json"))
        with open(os.path.join(self._model_out_dir, "config.json"), 'r+') as f:
            data = json.load(f) #json.load:参数是文件对象；json.loads:参数是json格式的字符串
            if self.additional_demo_info is not None and "init" in self.additional_demo_info["demo_collection"].keys():
                data['dataset']['additional_demo_info']={"pose_and_orientations":self.additional_demo_info["demo_collection"]["init"]["pose_and_orientations"]}
            data['dataset']['hdf5_img_size'] = self.hdf_img_size
            f.seek(0)    # 将文件指针移动到文件开头
            json.dump(data, f, indent=4)
            f.truncate()  #确保文件末尾没有多余的内容

    def _setup_dataset(self):
        """
        Set up the dataset.
        """
        print("start preparing dataset...")
        train_set  = dataset_factory(self._config["dataset"],  img_size=self._config["algorithm"]["policy"]["params"]["encoder"]["params"]["img_size"],filter_by_attribute='train') # TODO: remember to convert
        valid_set  = dataset_factory(self._config["dataset"],  img_size=self._config["algorithm"]["policy"]["params"]["encoder"]["params"]["img_size"],filter_by_attribute='valid')

        self._train_loader = DataLoader(  # shuffle=True：每个epoch开始时重新随机打乱并采样batch
            dataset=train_set,
            batch_size=self._config["training"]["batch_size"],
            shuffle=True,
            num_workers=self._config["training"]["num_data_workers"],
            drop_last=True
        )
        self._eval_loader = DataLoader(
            dataset=valid_set,
            batch_size=self._config["training"]["batch_size"],
            shuffle=True,
            num_workers=1,
            drop_last=True
        )
        self.hdf_img_size=train_set.hdf_img_size
        self.additional_demo_info=train_set.additional_demo_info
    def _setup_logging(self):
        """
        Set up the logging.
        """
        # Configure logging
        logging.basicConfig(
            filename=self._log_path,
            level=logging.INFO,  # Logging level
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

        self._logger = logging.getLogger()

        config_string = json.dumps(self._config, indent=4)
        self._logger.info(f"Config:\n{config_string}")

    def _setup_algorithm(self) -> BehaviorCloning:
        """
        Set up the algorithm.
        """
        assert self._config["algorithm"]["name"] == "BC", "Wrong algorithm name in configs file."

        # Set up the model
        model = self._setup_model(self._config)
        optimizer = self._setup_optimizer(self._config["algorithm"]["optimizer"], model)
        criterion = self._setup_criterion(self._config["algorithm"]["loss"],self._config["dataset"]["seq_length"],self._config["dataset"]["output_dim"])

        return BehaviorCloning(model, optimizer, criterion)

    def _setup_model(self, model_config: dict):
        """
        Set up the model.
        """
        model,need_init_params = get_network_cls(model_config["algorithm"]["policy"]["name"])
        if need_init_params:
            return model(
                input_low_dim=model_config["dataset"]["input_low_dim"],
                output_dim=model_config["dataset"]["output_dim"],
                obs_keys=model_config["dataset"]["specific_obs_keys"],
                batch_size=model_config["training"]["batch_size"],
                seq_length=model_config["dataset"]["seq_length"],
                training=True,
                **model_config["algorithm"]["policy"]["params"]
            )  # **动态传参，字典中的键与函数参数名完全匹配
        else:
            return model()
    @staticmethod
    def _setup_optimizer(optimizer_config: dict, model: torch.nn.Module):
        """
        Set up the optimizer.
        """
        optimizer = get_optimizer_cls(optimizer_config["name"])
        return optimizer(model.parameters(), **optimizer_config["params"])

    @staticmethod
    def _setup_criterion(criterion_config_name: dict,seq_length: int,output_dim: int):
        """
        Set up the criterion.
        """

        def composed_loss_fn(x, x_hat):
            # x = x.reshape(x.size(0), -1)
            # x_hat = x_hat.reshape(x_hat.size(0), -1)
            loss_fn = get_loss_fn(criterion_config_name["name"],criterion_config_name["weight"],seq_length,output_dim)
            loss_dict = loss_fn(x, x_hat)
            return loss_dict

        return composed_loss_fn


if __name__ == "__main__":
    exp = BehaviorCloningExperiment(config_path="train_mlp.json")
    # exp = BehaviorCloningExperiment(config_path="train_transformer.json")
    # exp = BehaviorCloningExperiment(config_path="train_transformer_single.json")
    exp.run()
