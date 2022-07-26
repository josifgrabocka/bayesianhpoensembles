import optuna
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import balanced_accuracy_score
import pickle
from pathlib import Path

class HPOScikitLearn:

    def __init__(self, config):

        self.config = config

    def evaluate(self, trial):

        classifier_name = trial.suggest_categorical("classifier", ["SVM", "RandomForest", "GBDT", "MLP"])

        # the data
        x_train = self.config["x_train"]
        y_train = self.config["y_train"]
        x_val = self.config["x_val"]
        y_val = self.config["y_val"]

        classifier = None

        if classifier_name == "SVM":
            svm_c = trial.suggest_float("svm_c", 1e0, 1e2, log=True)
            svm_kernel = trial.suggest_categorical("svm_kernel", ["rbf", "poly"])
            svm_degree = trial.suggest_int("svm_degree", 1, 5)
            classifier = SVC(C=svm_c, kernel=svm_kernel, degree=svm_degree)

        elif classifier_name == "RandomForest":
            rf_num_estimators = trial.suggest_int("rf_num_estimators", 16, 128, log=True)
            classifier = RandomForestClassifier(n_estimators=rf_num_estimators)

        elif classifier_name == "GBDT":
            gbdt_num_estimators = trial.suggest_int("gbdt_num_estimators", 100, 200)
            gbdt_learning_rate = trial.suggest_float("gbdt_learning_rate", 1e-2, 1e-1, log=True)
            gbdt_max_depth = trial.suggest_int("gbdt_max_depth", 3, 5)
            classifier = GradientBoostingClassifier(n_estimators=gbdt_num_estimators, learning_rate=gbdt_learning_rate,
                                                    max_depth=gbdt_max_depth)

        elif classifier_name == "NearestNeighbors":
            nn_num_neighbors = trial.suggest_int("nn_num_neighbors", 1, 10)
            classifier = KNeighborsClassifier(n_neighbors=nn_num_neighbors)

        elif classifier_name == "MLP":
            mlp_learning_rate_init = trial.suggest_float("mlp_learning_rate", 1e-3, 1e-2, log=True)
            mlp_activation = trial.suggest_categorical("mlp_activation", ['logistic', 'tanh', 'relu'])
            mlp_l2_penalty = trial.suggest_float("mlp_l2_penalty", 1e-5, 1e-3, log=True)
            mlp_num_layers = trial.suggest_int("mlp_num_layers", 2, 6)
            mlp_neurons_per_layer = trial.suggest_int("mlp_neurons_per_layer", 8, 64, log=True)
            mlp_max_iter = trial.suggest_int("mlp_max_iter", 500, 2000, log=True)


            hidden_layer_sizes = []
            for _ in range(mlp_num_layers):
                hidden_layer_sizes.append(mlp_neurons_per_layer)

            classifier = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, learning_rate_init=mlp_learning_rate_init,
                                       activation=mlp_activation, alpha=mlp_l2_penalty,
                                       max_iter=mlp_max_iter,
                                       n_iter_no_change=mlp_max_iter)

        # fit the classifier
        classifier.fit(x_train, y_train)

        # checkpoint the model
        path = self.config["log_folder"] + f"sampler={self.config['sampler']}/hpo-seed={self.config['current_seed']}/"
        # create the path if it does not exist
        Path(path).mkdir(parents=True, exist_ok=True)

        with open(path + f"model_{trial.number}.pickle", "wb") as fout:
            pickle.dump(classifier, fout)

        # evaluate on the validation set
        return 1.0 - balanced_accuracy_score(classifier.predict(x_val), y_val)


    def run_hpo(self):

        sampler = None
        if self.config['sampler'] == "random":
            sampler = optuna.samplers.RandomSampler()
        elif self.config['sampler'] == "tpe":
            sampler = optuna.samplers.TPESampler()
        else:
            raise ValueError("Unknown sampler " + self.config['sampler'])

        study = optuna.create_study(sampler=sampler)
        study.optimize(self.evaluate, n_trials=self.config['num_trials'])

        print(study.best_params)
