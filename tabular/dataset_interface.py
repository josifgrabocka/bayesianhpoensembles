import sklearn
import sklearn.model_selection
import sklearn.datasets
import sklearn.impute
import numpy as np
import openml


class DatasetInterface:

    def __init__(self, config):
        self.config = config


    def divide_into_train_val_split(self, X, y, train_val_test_fractions=(0.8, 0.1, 0.1)):

        test_fraction = train_val_test_fractions[2]

        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
            X, y, test_size=test_fraction, random_state=0)

        train_fraction = train_val_test_fractions[0]
        val_fraction = train_val_test_fractions[1]
        val_fraction_within_train = train_val_test_fractions[1] / (train_fraction+val_fraction)

        x_train, x_val, y_train, y_val = sklearn.model_selection.train_test_split(
            x_train, y_train, test_size=val_fraction_within_train, random_state=0)

        return (x_train, y_train), (x_val, y_val), (x_test, y_test)


    def get_toy_sklearn_dataset(self, train_val_test_fractions=(0.8, 0.1, 0.1)):

        X, y = None, None

        ds_name = self.config["dataset_name"]

        if ds_name == "digits":
            X, y = sklearn.datasets.load_digits(return_X_y=True)
        elif ds_name == "breast_cancer":
            X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
        else:
            raise ValueError(f'Unknown dataset{ds_name}')

        return self.divide_into_train_val_split(X, y, train_val_test_fractions=train_val_test_fractions)

    # read the openml dataset
    def get_openml_dataset(self, train_val_test_fractions=(0.8, 0.1, 0.1)):

        #task = openml.tasks.get_task(self.config["dataset_task_id"])
        #X, y = task.get_X_and_y()

        # read the data from files, because the cluster does not have internet access to read the data from OpenML
        X = np.load(self.config["openml_data_folder"] + "/" + str(self.config["dataset_task_id"]) + "_x.npy")
        y = np.load(self.config["openml_data_folder"] + "/" + str(self.config["dataset_task_id"]) + "_y.npy")

        imputer = sklearn.impute.SimpleImputer(missing_values=np.nan, strategy='mean')
        X = imputer.fit_transform(X)

        return self.divide_into_train_val_split(X, y, train_val_test_fractions=train_val_test_fractions)


    def get_local_collection(self, train_val_test_fractions=(0.8, 0.1, 0.1)):

        X, y = None, None

        path = self.config["dataset_folder"]

        train_features = np.load(path + "/train_features.npy")
        train_labels = np.load(path + "/train_labels.npy")
        test_features = np.load(path + "/test_features.npy")
        test_labels = np.load(path + "/test_labels.npy")
        # concatenate the features and labels
        X = np.concatenate([train_features, test_features])
        y = np.squeeze(np.concatenate([train_labels, test_labels]))

        return self.divide_into_train_val_split(X, y, train_val_test_fractions=train_val_test_fractions)

    def read_openml_suite(self):

        task_ids = openml.study.get_suite('OpenML-CC18').tasks

        for task_id in task_ids:
            task = openml.tasks.get_task(task_id)
            x, y = task.get_X_and_y()

            np.save(str(task_id) + "_x.npy", x)
            np.save(str(task_id) + "_y.npy", y)
