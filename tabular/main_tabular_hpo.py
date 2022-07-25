from tabular.dataset_interface import DatasetInterface
from tabular.tabular_hpo import HPOScikitLearn
from pathlib import Path
import numpy as np


# define the configuration
config = {}

# the log folder
config["log_folder"] = ""

# HPO config
config["num_trials"] = 10
config["num_seeds"] = 5

# dataset name
config["dataset_type"] = "local_collection"
config["dataset_name"] = "ijcnn1"
config["dataset_folder"] = "C:/Users/josif/data/binaryclassification/ijcnn1/0/"

# read the dataset
di = DatasetInterface(config)
if config["dataset_type"] == "sklearn_toy":
    data_splits = di.get_toy_sklearn_dataset()
elif config["dataset_type"] == "local_collection":

    data_splits = di.get_local_collection()
else:
    data_splits = di.get_openml_dataset()
# the dataset
config["x_train"] = data_splits[0][0]
config["y_train"] = data_splits[0][1]
config["x_val"] = data_splits[1][0]
config["y_val"] = data_splits[1][1]
config["x_test"] = data_splits[2][0]
config["y_test"] = data_splits[2][1]

# save the data splits
path = f"logs/dataset={config['dataset_name']}/"
# create the path if it does not exist
Path(path).mkdir(parents=True, exist_ok=True)

np.save(path+"x_train.npy", config["x_train"])
np.save(path+"y_train.npy", config["y_train"])
np.save(path+"x_val.npy", config["x_val"])
np.save(path+"y_val.npy", config["y_val"])
np.save(path+"x_test.npy", config["x_test"])
np.save(path+"y_test.npy", config["y_test"])


# conduct hpo
config["sampler"] = "default" # or "random"
hpoSL = HPOScikitLearn(config=config)
for seed in range(config["num_seeds"]):
    config["current_seed"] = seed
    hpoSL.run_hpo()
