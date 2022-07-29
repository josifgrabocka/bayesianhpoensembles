import os
import numpy as np
import pickle
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import OneHotEncoder
import pandas as pd


class BayesianHyperEnsembles:

    def __init__(self, config):
        self.config = config

        # load the data splits
        self.x_val = None
        self.y_val = None
        self.x_test = None
        self.y_test = None
        self.num_seeds = -1

        # checkpoints
        self.models = []

        # the validation errors of each model
        self.model_val_accuracy = []

        self.ohe_encoder = OneHotEncoder()
        self.set_labels = set()

        # the test predictions for each model
        self.model_test_predictions = []

    # load the data and the checkpoints for all the different seeds
    def load(self):

        # the path with the seeds sub-folders
        path = self.config["log_folder"] + "/" + f'task_id={self.config["task_id"]}' + "/" + f'sampler={self.config["sampler"]}'
        seed_folders = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name)) and "hpo-seed=" in name]
        self.num_seeds = len(seed_folders)

        # numpy files for the data
        self.x_val = np.load(self.config["log_folder"] + "/" + f'task_id={self.config["task_id"]}/x_val.npy')
        self.y_val = np.load(self.config["log_folder"] + "/" + f'task_id={self.config["task_id"]}/y_val.npy').astype(int)
        self.x_test = np.load(self.config["log_folder"] + "/" + f'task_id={self.config["task_id"]}/x_test.npy')
        self.y_test = np.load(self.config["log_folder"] + "/" + f'task_id={self.config["task_id"]}/y_test.npy').astype(int)


        num_seeds = len(seed_folders)

        for seed_folder in seed_folders:
            seed_folder_path = path + "/" + seed_folder
            checkpoint_files = [name for name in os.listdir(seed_folder_path)
                                if not os.path.isdir(os.path.join(seed_folder_path, name)) and ".pickle" in name]

            seed_models = []

            num_models = len(checkpoint_files)

            for model_file in checkpoint_files:
                model_file_path = seed_folder_path + "/" + model_file
                classifier = pickle.load(open(model_file_path, 'rb'))
                seed_models.append(classifier)

            self.models.append(seed_models)

        # store all the aggregated results
        self.num_posterior_baselines = 4
        self.results = np.zeros((num_models, self.num_posterior_baselines, num_seeds))

    # compute the posteriors
    def compute_model_predictions(self):

        for seed_models in self.models:

            seed_model_val_accuracy = []
            seed_model_test_predictions = []


            for model in seed_models:
                # the validation accuracy
                seed_model_val_accuracy.append(balanced_accuracy_score(self.y_val, model.predict(self.x_val)))
                # the test predictions
                model_test_prediction = model.predict(self.x_test)

                for y in list(model_test_prediction.flat):
                    self.set_labels.add(y)

                seed_model_test_predictions.append(model_test_prediction.astype(int))

            self.model_val_accuracy.append(seed_model_val_accuracy)
            self.model_test_predictions.append(seed_model_test_predictions)


    def convert_to_one_hot(self, y):
        num_classes = len(self.set_labels)
        return np.eye(num_classes)[y]

    def compute_posteriors(self, val_accuracies, posterior_type):

        if posterior_type == "bayesian-likelihood":
            # compute the posteriors from the validation scores
            posteriors = val_accuracies
            posteriors /= np.sum(val_accuracies)

        elif posterior_type == "bayesian-accuracy-rank":
            accuracies_series = pd.Series(val_accuracies)
            accuracies_ranks = accuracies_series.rank().to_numpy()
            K = float(len(val_accuracies))
            posteriors = (2.0 * accuracies_ranks) / (K * (K + 1))

        elif posterior_type == "uniform":
            posteriors = np.ones_like(val_accuracies) / float(len(val_accuracies))

        elif posterior_type == "best":
            posteriors = np.zeros_like(val_accuracies)
            best_model_idx = np.argmax(val_accuracies, axis=-1)
            posteriors[best_model_idx] = 1.0

        else:
            raise ValueError("Unknown posterior type " + posterior_type)

        return posteriors

    # create the bayesian ensemble
    def results_aggregation(self):

        for seed_idx in range(self.num_seeds):

            num_models = len(self.model_val_accuracy[seed_idx])

            for ensemble_size in range(1, num_models+1):

                val_accuracies = self.model_val_accuracy[seed_idx][:ensemble_size]

                results = []

                for posterior_idx, posterior_type in enumerate(["best", "uniform", "bayesian-likelihood", "bayesian-accuracy-rank"]):
                    # compute the posteriors
                    posteriors = self.compute_posteriors(val_accuracies=val_accuracies,
                                                         posterior_type=posterior_type)

                    # the aggregated ensemble predictions, model averaging
                    aggregated_ensemble_prediction = None

                    # compute the predictions for the test instances
                    for model_idx in range(ensemble_size):
                        if aggregated_ensemble_prediction is None:
                            aggregated_ensemble_prediction = posteriors[model_idx] * self.convert_to_one_hot(self.model_test_predictions[seed_idx][model_idx])
                        else:
                            aggregated_ensemble_prediction += posteriors[model_idx] * self.convert_to_one_hot(self.model_test_predictions[seed_idx][model_idx])

                    test_accuracy = balanced_accuracy_score(y_true=self.y_test, y_pred=np.argmax(aggregated_ensemble_prediction, axis=-1))

                    self.results[ensemble_size - 1, posterior_idx, seed_idx] = test_accuracy

                    results.append(test_accuracy)

        # print results
        for ensemble_size in range(1, num_models + 1):
            results =  []
            for baseline_idx in range(self.num_posterior_baselines):
                results.append(np.mean(self.results[ensemble_size - 1, baseline_idx]))
            print(ensemble_size, results)


    def run(self):
        # load the data and the model checkpoints
        self.load()

        # compute posteriors
        self.compute_model_predictions()
        # aggregate the results
        self.results_aggregation()
        
        # save the results to a numpy tensor
        np.save(self.config['results_folder'] + f"results_task_id={self.config['task_id']}_sampler={self.config['sampler']}", self.results)
