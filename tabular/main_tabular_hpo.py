from tabular.dataset_interface import DatasetInterface
from tabular.tabular_hpo import HPOScikitLearn
from pathlib import Path
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--task_id', help='Parameter of the Top-K loss', type=int)
parser.add_argument('--log_folder', help='where to save the hpo logs', type=str)
parser.add_argument('--sampler', help='random, hpo', type=str)
args = parser.parse_args()

# define the configuration
config = {}

# the log folder
config["log_folder"] = args.log_folder

# HPO config
config["num_trials"] = 50
config["num_seeds"] = 5

# dataset name
config["dataset_task_id"] = args.task_id

# read the dataset
di = DatasetInterface(config)
data_splits = di.get_openml_dataset()

# the dataset
config["x_train"] = data_splits[0][0]
config["y_train"] = data_splits[0][1]
config["x_val"] = data_splits[1][0]
config["y_val"] = data_splits[1][1]
config["x_test"] = data_splits[2][0]
config["y_test"] = data_splits[2][1]

# save the data splits
config["log_folder"] = config["log_folder"] + f"/task_id={config['dataset_task_id']}/"
# create the path if it does not exist
Path(config["log_folder"]).mkdir(parents=True, exist_ok=True)

np.save(config["log_folder"]+"x_train.npy", config["x_train"])
np.save(config["log_folder"]+"y_train.npy", config["y_train"])
np.save(config["log_folder"]+"x_val.npy", config["x_val"])
np.save(config["log_folder"]+"y_val.npy", config["y_val"])
np.save(config["log_folder"]+"x_test.npy", config["x_test"])
np.save(config["log_folder"]+"y_test.npy", config["y_test"])

# conduct hpo
config["sampler"] = args.sampler
hpoSL = HPOScikitLearn(config=config)
for seed in range(config["num_seeds"]):
    config["current_seed"] = seed
    hpoSL.run_hpo()
