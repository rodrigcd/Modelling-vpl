import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd
import pickle as pkl
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from scrips.curr_flexnet import all_regimes
from vpl_model.utils import plot_config


def get_loss(data, model_id):
    loss = data[model_id]["loss"]
    return loss

def get_accuracy(data, model_id):
    accuracy = data[model_id]["accuracy"]
    return accuracy


def apply_func_to_all_files(save_path, all_model_ids, n_runs, func):
    all_results = {}
    for model_id in all_model_ids:
        for run in range(n_runs):
            file_name = save_path + model_id + "_run_" + str(run) + ".pkl"
            data = pd.read_pickle(file_name)
            all_results[model_id] = func(data, model_id)


def main():
    plot_config()
    # Load the data
    data_path = "../all_results/neural_data_net_slurm/"
    all_files = glob.glob(data_path + "*.pkl")
    n_runs = 5

    all_model_ids = list(all_regimes.keys())

    # Plot loss function
    f, ax = plt.subplots(3, 3, figsize=(10, 10))
    ax = ax.flatten()
    for i, model_id in enumerate(all_model_ids):
        for run in range(n_runs):
            file_name = data_path + model_id + "_run_" + str(run) + ".pkl"
            data = pd.read_pickle(file_name)
            loss = get_loss(data, model_id)
            ax[i].plot(loss, label="run_" + str(run))
        ax[i].set_title(model_id)
    plt.show()

    # TODO:
    # 1. A few examples of training curves per model id
    # 2. Analysis comparison for all models and for a single seed
    # ---- Add change in slope on each orientation (this is two other rows)
    # ---- Add single cell tuning curve examples

    # 3. Mean and std for all models on a few metrics
    # ---- All loss overlap
    # ---- All accuracy overlap
    # ---- Change in slope PO-TO
    # ---- Time evolution of bandwidth and orientation
    # ---- Change in bandwidth distributions
    # ---- Change in orientation distributions
    # ---- Change in slope at trained orientation
    # ---- Change in slope on each orientation


    # all_means = {}
    # all_stds = {}
    # for key in all_regimes.keys():
    #     key_path_list = [f for f in all_files if key in f]
    #     mean, std = get_all_mean_std(key_path_list, key)
    #     all_means[key] = mean
    #     all_stds[key] = std
    #
    # # Load the data
    # data_ex = pd.read_pickle(all_files[0])
    # print(data_ex.keys())


if __name__ == "__main__":
    main()