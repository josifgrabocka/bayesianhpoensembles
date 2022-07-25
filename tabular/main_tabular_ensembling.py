from BayesianHyperEnsembles import BayesianHyperEnsembles

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--task_id', help='The OpenML task id', type=int)
parser.add_argument('--log_folder', help='Where to save the hpo logs', type=str)
parser.add_argument('--results_folder', help='Where to save the baseline results', type=str)
parser.add_argument('--sampler', help='random, hpo', type=str)
args = parser.parse_args()


# define the configuration
config = {}

config["task_id"] = args.task_id
config["sampler"] = args.sampler
config["log_folder"] = args.log_folder
config["results_folder"] = args.results_folder



bhe = BayesianHyperEnsembles(config=config)


