import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd
import pickle as pkl
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from scrips.curr_flexnet import all_regimes
from curr_flexnet import *


def get_all_mean_std(key_path_list, model_id):
    loss_list = []
    for key_path in key_path_list:
        data = pd.read_pickle(key_path)
        loss, _ = get_loss(data, make_plot=False)
        loss_list.append(loss[model_id])
    mean = np.mean(loss_list, axis=0)


def main():
    # Load the data
    data_path = "../all_results/flexnet_slurm/"
    all_files = glob.glob(data_path + "*.pkl")
    print(all_files)

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


    all_means = {}
    all_stds = {}
    for key in all_regimes.keys():
        key_path_list = [f for f in all_files if key in f]
        mean, std = get_all_mean_std(key_path_list, key)
        all_means[key] = mean
        all_stds[key] = std

    # Load the data
    data_ex = pd.read_pickle(all_files[0])
    print(data_ex.keys())


if __name__ == "__main__":
    main()