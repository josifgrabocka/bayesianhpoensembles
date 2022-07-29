from BayesianHyperEnsembles import BayesianHyperEnsembles

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--log_folder', help='Where to save the hpo logs', type=str)
parser.add_argument('--results_folder', help='Where to save the baseline results', type=str)
args = parser.parse_args()

config = {}
config["log_folder"] = args.log_folder
config["results_folder"] = args.results_folder

tasks_folders = [name for name in os.listdir(config["log_folder"])
                 if os.path.isdir(os.path.join(config["log_folder"], name)) and "task_id=" in name]

# aggregate the results for that task
for task_folder in tasks_folders:

    task_id = int(task_folder.split("=")[1])

    config["task_id"] = task_id

    for sampler in ['tpe', 'random']:
        config["sampler"] = sampler

        print('Run', config)

        bhe = BayesianHyperEnsembles(config=config)
        bhe.run()

