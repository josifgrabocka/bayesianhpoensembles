import os
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, log_loss
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

        self.alpha = 0.3

        # checkpoints
        self.models = []

        # the validation errors of each model
        self.model_likelihoods = []

        self.ohe_encoder = OneHotEncoder()
        self.set_labels = set()

        # the test predictions for each model
        self.model_val_predictions = []
        # the test predictions for each model
        self.model_test_predictions = []

    # load the data and the checkpoints for all the different seeds
    def load(self):

        # the path with the seeds sub-folders
        path = self.config["log_folder"] + "/" + f'task_id={self.config["task_id"]}' + "/" + f'sampler={self.config["sampler"]}'
        seed_folders = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name)) and "hpo-seed=" in name]
        self.num_seeds = len(seed_folders)

        # numpy files for the data
        self.x_train = np.load(self.config["log_folder"] + "/" + f'task_id={self.config["task_id"]}/x_train.npy')
        self.y_train = np.load(self.config["log_folder"] + "/" + f'task_id={self.config["task_id"]}/y_train.npy').astype(int)
        self.x_val = np.load(self.config["log_folder"] + "/" + f'task_id={self.config["task_id"]}/x_val.npy')
        self.y_val = np.load(self.config["log_folder"] + "/" + f'task_id={self.config["task_id"]}/y_val.npy').astype(int)

        self.x_train_val = np.concatenate((self.x_train, self.x_val), axis=0)
        self.y_train_val = np.concatenate((self.y_train, self.y_val), axis=0)

        self.x_test = np.load(self.config["log_folder"] + "/" + f'task_id={self.config["task_id"]}/x_test.npy')
        self.y_test = np.load(self.config["log_folder"] + "/" + f'task_id={self.config["task_id"]}/y_test.npy').astype(int)

        num_seeds = len(seed_folders)

        for seed_folder in seed_folders:
            seed_folder_path = path + "/" + seed_folder
            checkpoint_files = [name for name in os.listdir(seed_folder_path)
                                if not os.path.isdir(os.path.join(seed_folder_path, name)) and ".pickle" in name]

            seed_models = []

            self.num_models = len(checkpoint_files)

            for model_file in checkpoint_files:
                model_file_path = seed_folder_path + "/" + model_file
                classifier = pickle.load(open(model_file_path, 'rb'))
                seed_models.append(classifier)

            self.models.append(seed_models)

        # store all the aggregated results
        self.num_posterior_baselines = 5
        self.results = np.zeros((self.num_models, self.num_posterior_baselines, num_seeds))

    # compute the posteriors
    def compute_model_predictions(self):

        for seed_models in self.models:

            seed_model_likelihoods = []
            seed_model_val_predictions = []
            seed_model_test_predictions = []

            for model in seed_models:
                seed_model_likelihoods.append(np.exp(-log_loss(self.y_val, model.predict_proba(self.x_val))))
                seed_model_test_predictions.append(model.predict_proba(self.x_test))
                seed_model_val_predictions.append(model.predict(self.x_val))

            self.model_likelihoods.append(seed_model_likelihoods)
            self.model_test_predictions.append(seed_model_test_predictions)
            self.model_val_predictions.append(seed_model_val_predictions)

    def compute_posteriors(self, model_likelihoods, model_val_predictions, posterior_type):

        if posterior_type == "bayesian-likelihood":
            # compute the posteriors from the validation scores
            posteriors = model_likelihoods
            posteriors /= np.sum(model_likelihoods)

        elif posterior_type == "bayesian-linear":

            if len(model_likelihoods) > 1:
                posteriors = (model_likelihoods - np.min(model_likelihoods))
                posteriors /= np.sum((model_likelihoods - np.min(model_likelihoods)))
            else:
                posteriors = np.array([1.0])

        elif posterior_type == "bayesian-rank":
            accuracies_series = pd.Series(model_likelihoods)
            ranks = accuracies_series.rank().to_numpy()

            scores = np.exp(self.alpha*ranks)
            posteriors = scores / np.sum(scores)

        elif posterior_type == "uniform":
            posteriors = np.ones_like(model_likelihoods) / float(len(model_likelihoods))

        elif posterior_type == "best":
            posteriors = np.zeros_like(model_likelihoods)
            best_model_idx = np.argmax(model_likelihoods, axis=-1)
            posteriors[best_model_idx] = 1.0

        else:
            raise ValueError("Unknown posterior type " + posterior_type)

        return posteriors

    # create the bayesian ensemble
    def results_aggregation(self):

        for seed_idx in range(self.num_seeds):

            num_models = len(self.model_likelihoods[seed_idx])

            for ensemble_size in range(1, num_models+1):

                model_likelihoods = self.model_likelihoods[seed_idx][:ensemble_size]
                model_val_predictions = self.model_val_predictions[seed_idx][:ensemble_size]

                results = []

                for posterior_idx, posterior_type in enumerate(["best", "uniform", "bayesian-likelihood", "bayesian-rank", "bayesian-linear"]):
                    # compute the posteriors
                    posteriors = self.compute_posteriors(model_likelihoods=model_likelihoods,
                                                         model_val_predictions=model_val_predictions,
                                                         posterior_type=posterior_type)

                    # the aggregated ensemble predictions, model averaging
                    aggregated_ensemble_prediction = None

                    # compute the predictions for the test instances
                    for model_idx in range(ensemble_size):
                        if aggregated_ensemble_prediction is None:
                            aggregated_ensemble_prediction = posteriors[model_idx] * self.model_test_predictions[seed_idx][model_idx]
                        else:
                            aggregated_ensemble_prediction += posteriors[model_idx] * self.model_test_predictions[seed_idx][model_idx]

                    y_test_pred_hard = np.argmax(aggregated_ensemble_prediction, axis=-1)
                    test_accuracy = accuracy_score(y_true=self.y_test, y_pred=y_test_pred_hard)
                    self.results[ensemble_size - 1, posterior_idx, seed_idx] = test_accuracy

                    #test_likelihood = self.results[ensemble_size - 1, posterior_idx, seed_idx] = np.exp(-log_loss(y_true=self.y_test, y_pred=aggregated_ensemble_prediction))
                    #results.append(test_likelihood)

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
