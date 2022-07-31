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

        # checkpoints
        self.models = []

        # the validation errors of each model
        self.model_val_likelihoods = []

        self.ohe_encoder = OneHotEncoder()
        self.set_labels = set()

        # the test predictions for each model
        self.model_test_predictions = []

        self.alpha = 1.0

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
        self.num_posterior_baselines = 5
        self.results = np.zeros((num_models, self.num_posterior_baselines, self.num_seeds))

    # compute the posteriors
    def compute_model_predictions(self):

        for seed_models in self.models:

            seed_model_val_likelihood = []
            seed_model_test_predictions = []


            for model in seed_models:
                nll = log_loss(self.y_val, model.predict_proba(self.x_val))
                likelihood = np.exp(-nll)
                accuracy = accuracy_score(self.y_val, model.predict(self.x_val))

                # the validation accuracy
                seed_model_val_likelihood.append(likelihood)
                # the test predictions
                model_test_prediction = model.predict_proba(self.x_test)

                seed_model_test_predictions.append(model_test_prediction)

            self.model_val_likelihoods.append(seed_model_val_likelihood)
            self.model_test_predictions.append(seed_model_test_predictions)

    def compute_posteriors(self, val_likelihoods, posterior_type):

        if posterior_type == "bayesian-likelihood":
            # compute the posteriors from the validation scores
            posteriors = val_likelihoods
            posteriors /= np.sum(val_likelihoods)

        elif posterior_type == "bayesian-scaled":
            if len(val_likelihoods) > 1:
                posteriors = val_likelihoods - np.min(val_likelihoods)
                posteriors /= np.sum(val_likelihoods - np.min(val_likelihoods))
            else:
                posteriors = np.array([1.0])

        elif posterior_type == "bayesian-rank":
            accuracies_series = pd.Series(val_likelihoods)
            accuracies_ranks_exp = np.exp(self.alpha*accuracies_series.rank().to_numpy())
            posteriors = accuracies_ranks_exp / np.sum(accuracies_ranks_exp)

        elif posterior_type == "uniform":
            posteriors = np.ones_like(val_likelihoods) / float(len(val_likelihoods))

        elif posterior_type == "best":
            posteriors = np.zeros_like(val_likelihoods)
            best_model_idx = np.argmax(val_likelihoods, axis=-1)
            posteriors[best_model_idx] = 1.0

        else:
            raise ValueError("Unknown posterior type " + posterior_type)

        return posteriors

    # create the bayesian ensemble
    def results_aggregation(self):

        for seed_idx in range(self.num_seeds):

            num_models = len(self.model_val_likelihoods[seed_idx])

            for ensemble_size in range(1, num_models+1):

                val_accuracies = self.model_val_likelihoods[seed_idx][:ensemble_size]

                results = []

                for posterior_idx, posterior_type in enumerate(["best", "uniform", "bayesian-likelihood", "bayesian-rank", "bayesian-scaled"]):
                    # compute the posteriors
                    posteriors = self.compute_posteriors(val_likelihoods=val_accuracies,
                                                         posterior_type=posterior_type)

                    # the aggregated ensemble predictions, model averaging
                    aggregated_ensemble_prediction = None

                    # compute the predictions for the test instances
                    for model_idx in range(ensemble_size):
                        if aggregated_ensemble_prediction is None:
                            aggregated_ensemble_prediction = posteriors[model_idx] * self.model_test_predictions[seed_idx][model_idx]
                        else:
                            aggregated_ensemble_prediction += posteriors[model_idx] * self.model_test_predictions[seed_idx][model_idx]

                    #test_likelihood = np.exp(-log_loss(y_true=self.y_test, y_pred=aggregated_ensemble_prediction))

                    test_likelihood = accuracy_score (y_true=self.y_test, y_pred=np.argmax(aggregated_ensemble_prediction, axis=-1))

                    self.results[ensemble_size - 1, posterior_idx, seed_idx] = test_likelihood

                    results.append(test_likelihood)

        # print results
        for ensemble_size in range(1, num_models + 1):
            results =  []
            for baseline_idx in range(self.num_posterior_baselines):
                results.append(np.mean(self.results[ensemble_size - 1, baseline_idx]))

            print(self.config['task_id'], ensemble_size, results)


    def run(self):
        # load the data and the model checkpoints
        self.load()

        # compute posteriors
        self.compute_model_predictions()
        # aggregate the results
        self.results_aggregation()
        
        # save the results to a numpy tensor
        np.save(self.config['results_folder'] + f"results_task_id={self.config['task_id']}_sampler={self.config['sampler']}", self.results)
