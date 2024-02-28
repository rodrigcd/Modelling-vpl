import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd
import pickle as pkl
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from scrips.curr_flexnet import all_regimes
from vpl_model.utils import plot_config
from curr_flexnet import get_tuning_curves, get_angle_bandwidth_through_time


def get_mean_std(data_array):
    mean = np.mean(data_array, axis=0)
    std = np.std(data_array, axis=0)
    return mean, std


def get_loss(data, model_id):
    loss = data[model_id]["loss"]
    switch_times = data[model_id]["curriculum_switch"]
    return loss, switch_times


def get_accuracy(data, model_id):
    accuracy = data[model_id]["accuracy"]
    switch_times = data[model_id]["curriculum_switch"]
    return accuracy, switch_times


def apply_func_to_all_files(save_path, all_model_ids, n_runs, func):
    all_results =  {}
    for model_id in all_model_ids:
        for run in range(n_runs):
            file_name = save_path + model_id + "_run_" + str(run) + ".pkl"
            data = pd.read_pickle(file_name)
            all_results[model_id] = func(data, model_id)


def plot_loss(all_model_ids, n_runs, data_path):
    # Plot loss function
    f, ax = plt.subplots(3, 3, figsize=(10, 10))
    ax = ax.flatten()
    for i, model_id in enumerate(all_model_ids):
        for run in range(n_runs):
            file_name = data_path + model_id + "_run_" + str(run) + ".pkl"
            try:
                data = pd.read_pickle(file_name)
            except:
                print("file not found", file_name)
                continue
            loss, switches = get_loss(data, model_id)
            ax[i].plot(loss, label="run_" + str(run), color="C" + str(run))
            for switch in switches:
                ax[i].axvline(switch, color="C" + str(run), linestyle="--")
        ax[i].set_title(model_id)
    ax[0].legend()
    plt.tight_layout()
    plt.show()


def plot_accuracy(all_model_ids, n_runs, data_path):
    # Plot accuracy
    f, ax = plt.subplots(3, 3, figsize=(10, 10))
    ax = ax.flatten()
    for i, model_id in enumerate(all_model_ids):
        for run in range(n_runs):
            file_name = data_path + model_id + "_run_" + str(run) + ".pkl"
            try:
                data = pd.read_pickle(file_name)
            except:
                print("file not found", file_name)
                continue
            accuracy, switches = get_accuracy(data, model_id)
            ax[i].plot(accuracy, label="run_" + str(run), color = "C" + str(run))
            for switch in switches:
                ax[i].axvline(switch, color="C" + str(run), linestyle="--")
        ax[i].set_title(model_id)
    ax[0].legend()
    plt.tight_layout()
    plt.show()


def slope_diff_PO_TO(activity, n_training_stages=5,
                     normalize_tuning=True, TO=90.0):
    act = activity
    angle_dslope_per_step = []
    for t in range(n_training_stages):
        angles = np.linspace(0, 180, act.shape[-1])
        PO_index = np.argmax(act[t, ...], axis=1)
        PO = angles[PO_index]
        if normalize_tuning:
            activity = act[t, ...]/np.max(act[t, ...], axis=1)[:, np.newaxis]
        else:
            activity = act[t, ...]
        slopes = np.diff(np.concatenate([activity, activity[:, 1][:, np.newaxis]], axis=1), axis=1)
        x_axis = PO - TO
        TO_index = np.argmin(np.abs(angles - TO))
        y_axis = slopes[:, TO_index]
        sort_index = np.argsort(x_axis)
        x_axis = x_axis[sort_index]
        y_axis = y_axis[sort_index]
        if t == 0:
            base_y = y_axis
        angle_dslope_per_step.append(np.stack([x_axis, y_axis-base_y], axis=0))
    return np.stack(angle_dslope_per_step, axis=0)


def next_slope_diff_PO_TO(activity, n_training_stages=5,
                          normalize_tuning=True, TO=90.0):
        act = activity
        n_cells = act.shape[1]
        colors = plt.cm.viridis(np.linspace(0, 1, n_training_stages+1))
        angles = np.linspace(0, 180, act.shape[-1])
        angle_dslope_per_step = []
        for t in range(1, n_training_stages-1):
            angles = np.linspace(0, 180, act.shape[-1])
            PO_index = np.argmax(act[t, ...], axis=1)
            PO = angles[PO_index]
            if normalize_tuning:
                activity = act[t, ...]/np.max(act[t, ...], axis=1)[:, np.newaxis]
            else:
                activity = act[t, ...]
            slopes = np.diff(np.concatenate([activity, activity[:, 1][:, np.newaxis]], axis=1), axis=1)
            x_axis = PO - TO
            TO_index = np.argmin(np.abs(angles - TO))
            y_axis = slopes[:, TO_index]
            sort_index = np.argsort(x_axis)
            x_axis = x_axis[sort_index]
            y_axis = y_axis[sort_index]

            t2 = t+1
            angles = np.linspace(0, 180, act.shape[-1])
            PO_index = np.argmax(act[t2, ...], axis=1)
            PO = angles[PO_index]
            if normalize_tuning:
                activity = act[t2, ...]/np.max(act[t2, ...], axis=1)[:, np.newaxis]
            else:
                activity = act[t2, ...]
            slopes = np.diff(np.concatenate([activity, activity[:, 1][:, np.newaxis]], axis=1), axis=1)
            x_axis = PO - TO
            TO_index = np.argmin(np.abs(angles - TO))
            y_axis2 = slopes[:, TO_index]
            sort_index = np.argsort(x_axis)
            x_axis = x_axis[sort_index]
            y_axis2 = y_axis2[sort_index]
            angle_dslope_per_step.append(np.stack([x_axis, y_axis2-y_axis], axis=0))
        return np.stack(angle_dslope_per_step, axis=0)


def cosyne_similarities(activity, n_training_stages=5):
    act = activity
    cosyne_sim_steps = []
    for t in range(n_training_stages):
        angles = np.linspace(0, 180, act.shape[-1])
        cosyne_sim_list = []
        for runing_angle in range(90):
            left_activity = act[t, :, runing_angle]
            right_activity = act[t, :, -(runing_angle+1)]
            cosyne_sim = (left_activity*right_activity)/(np.linalg.norm(left_activity)*np.linalg.norm(right_activity))
            cosyne_sim_list.append(np.sum(cosyne_sim))
        cosyne_sim_steps.append(np.stack(cosyne_sim_list, axis=0))
    return np.stack(cosyne_sim_steps, axis=0)


def main():
    plot_config()
    # Load the data
    data_path = "../all_results/random_neural_init/"
    # data_path = "../all_results/neural_data_net_slurm/"
    all_files = glob.glob(data_path + "*.pkl")
    n_runs = 5
    n_training_stages = 5

    include_list = ["gradient_descent",
                    "feedback_alignment",
                    "predictive_coding"]

    all_model_ids = list(all_regimes.keys())

    # plot_loss(all_model_ids, n_runs, data_path)
    # plot_accuracy(all_model_ids, n_runs, data_path)

    f, ax = plt.subplots(2, 3, figsize=(10, 10))

    all_tuning_curves = {}
    for model_id in all_model_ids:
        all_tuning_curves[model_id] = []
        for run in range(n_runs):
            file_name = data_path + model_id + "_run_" + str(run) + ".pkl"
            try:
                data = pd.read_pickle(file_name)
            except:
                print("file not found", file_name)
                continue
            layer1, layer2, layer3 = get_tuning_curves(data, model_id)
            layers_dict = {"layer1": layer1[model_id],
                           "layer2": layer2[model_id],
                           "layer3": layer3[model_id]}
            all_tuning_curves[model_id].append(layers_dict)

    normalize_tuning = False
    """ Plotting slopes change with respect to naive """

    # layer = "layer1"
    # row_index = 0
    # ax[row_index, 0].set_ylabel(r"$\Delta$ slope L1")
    # for i, model_id in enumerate(include_list):
    #     all_runs_dslope = []
    #     ax[row_index, i].set_title(model_id)
    #     for run in range(n_runs):
    #         try:
    #             activity = all_tuning_curves[model_id][run][layer]
    #         except:
    #             print("file not found", model_id, run, layer)
    #             continue
    #         angle_dslope_per_step = slope_diff_PO_TO(activity, normalize_tuning=normalize_tuning)
    #         all_runs_dslope.append(angle_dslope_per_step)
    #     all_runs_dslope = np.stack(all_runs_dslope, axis=0)
    #     mean, std = get_mean_std(all_runs_dslope)
    #     n_training_steps = mean.shape[0]
    #     colors = plt.cm.viridis(np.linspace(0, 1, n_training_steps+1))
    #     for t in range(n_training_steps):
    #         ax[row_index, i].plot(mean[t, 0, :], mean[t, 1, :], label="step_" + str(t), color=colors[t])
    #
    # layer = "layer2"
    # row_index = 1
    # ax[row_index, 0].set_ylabel(r"$\Delta$ slope L2")
    # for i, model_id in enumerate(include_list):
    #     all_runs_dslope = []
    #     for run in range(n_runs):
    #         try:
    #             activity = all_tuning_curves[model_id][run][layer]
    #         except:
    #             print("file not found", model_id, run, layer)
    #             continue
    #         angle_dslope_per_step = slope_diff_PO_TO(activity, normalize_tuning=normalize_tuning)
    #         all_runs_dslope.append(angle_dslope_per_step)
    #     all_runs_dslope = np.stack(all_runs_dslope, axis=0)
    #     mean, std = get_mean_std(all_runs_dslope)
    #     n_training_steps = mean.shape[0]
    #     colors = plt.cm.viridis(np.linspace(0, 1, n_training_steps+1))
    #     for t in range(n_training_steps):
    #         ax[row_index, i].plot(mean[t, 0, :], mean[t, 1, :], label="step_" + str(t), color=colors[t])

    """ Plotting slope one step difference """
    layer = "layer1"
    row_index = 0
    ax[row_index, 0].set_ylabel(r"$\Delta(t+1, t)$ slope L2")
    for i, model_id in enumerate(include_list):
        all_runs_dslope = []
        for run in range(n_runs):
            try:
                activity = all_tuning_curves[model_id][run][layer]
            except:
                print("file not found", model_id, run, layer)
                continue
            angle_dslope_per_step = next_slope_diff_PO_TO(activity, normalize_tuning=normalize_tuning)
            all_runs_dslope.append(angle_dslope_per_step)
        all_runs_dslope = np.stack(all_runs_dslope, axis=0)
        mean, std = get_mean_std(all_runs_dslope)
        n_training_steps = mean.shape[0]
        colors = plt.cm.viridis(np.linspace(0, 1, n_training_steps + 1))
        for t in range(n_training_steps):
            ax[row_index, i].plot(mean[t, 0, :], mean[t, 1, :], label="step_" + str(t), color=colors[t])

    layer = "layer2"
    row_index = 1
    ax[row_index, 0].set_ylabel(r"$\Delta(t+1, t)$ slope L2")
    for i, model_id in enumerate(include_list):
        all_runs_dslope = []
        for run in range(n_runs):
            try:
                activity = all_tuning_curves[model_id][run][layer]
            except:
                print("file not found", model_id, run, layer)
                continue
            angle_dslope_per_step = next_slope_diff_PO_TO(activity, normalize_tuning=normalize_tuning)
            all_runs_dslope.append(angle_dslope_per_step)
        all_runs_dslope = np.stack(all_runs_dslope, axis=0)
        mean, std = get_mean_std(all_runs_dslope)
        n_training_steps = mean.shape[0]
        colors = plt.cm.viridis(np.linspace(0, 1, n_training_steps + 1))
        for t in range(n_training_steps):
            ax[row_index, i].plot(mean[t, 0, :], mean[t, 1, :], label="step_" + str(t), color=colors[t])

    """ Plotting bandwidth distribution """
    use_FWHM = True
    layer1 = "layer1"
    row_index = 2
    ax[row_index, 0].set_ylabel("Bandwidth distr L1")
    for i, model_id in enumerate(include_list):
        all_bandwidths = []
        ax[row_index, i].set_title(model_id)
        for run in range(n_runs):
            try:
                activity = all_tuning_curves[model_id][run][layer1]
            except:
                print("file not found", model_id, run, layer1)
                continue
            aux_dict = {model_id: activity}
            mean, bandwidth = get_angle_bandwidth_through_time(aux_dict)
            bandwidth = bandwidth[model_id]
            histograms = []
            for t in range(n_training_stages):
                if use_FWHM:
                    h, bins = np.histogram(bandwidth[t, :]*2*np.sqrt(2*np.log(2)),
                                           bins=20, density=True)
                else:
                    h, bins = np.histogram(bandwidth[t, :], bins=20, density=True)
                histograms.append(np.stack([bins[:-1], h], axis=0))
            all_bandwidths.append(np.stack(histograms, axis=0))
        all_bandwidths = np.stack(all_bandwidths, axis=0)
        mean, std = get_mean_std(all_bandwidths)
        n_training_steps = mean.shape[0]
        colors = plt.cm.viridis(np.linspace(0, 1, n_training_steps + 1))
        for t in range(n_training_steps):
            ax[row_index, i].plot(mean[t, 0, :], mean[t, 1, :], label="step_" + str(t), color=colors[t])

    layer1 = "layer2"
    row_index = 3
    ax[row_index, 0].set_ylabel("Bandwidth distr L2")
    for i, model_id in enumerate(include_list):
        all_bandwidths = []
        for run in range(n_runs):
            try:
                activity = all_tuning_curves[model_id][run][layer1]
            except:
                print("file not found", model_id, run, layer1)
                continue
            aux_dict = {model_id: activity}
            mean, bandwidth = get_angle_bandwidth_through_time(aux_dict)
            bandwidth = bandwidth[model_id]
            histograms = []
            for t in range(n_training_stages):
                if use_FWHM:
                    h, bins = np.histogram(bandwidth[t, :] * 2 * np.sqrt(2 * np.log(2)),
                                           bins=20, density=True)
                else:
                    h, bins = np.histogram(bandwidth[t, :], bins=20, density=True)
                histograms.append(np.stack([bins[:-1], h], axis=0))
            all_bandwidths.append(np.stack(histograms, axis=0))
        all_bandwidths = np.stack(all_bandwidths, axis=0)
        mean, std = get_mean_std(all_bandwidths)
        n_training_steps = mean.shape[0]
        colors = plt.cm.viridis(np.linspace(0, 1, n_training_steps + 1))
        for t in range(n_training_steps):
            ax[row_index, i].plot(mean[t, 0, :], mean[t, 1, :], label="step_" + str(t), color=colors[t])

    """ Cosyne similarity """
    layer1 = "layer1"
    row_index = 0
    ax[row_index, 0].set_ylabel("Cosyne similarity L1")
    for i, model_id in enumerate(include_list):
        all_cosynes = []
        ax[row_index, i].set_title(model_id)
        for run in range(n_runs):
            try:
                activity = all_tuning_curves[model_id][run][layer1]
            except:
                print("file not found", model_id, run, layer1)
                continue
            cosyne_sim = cosyne_similarities(activity, n_training_stages)
            all_cosynes.append(cosyne_sim)
        all_cosynes = np.stack(all_cosynes, axis=0)
        mean, std = get_mean_std(all_cosynes)
        n_training_steps = mean.shape[0]
        colors = plt.cm.viridis(np.linspace(0, 1, n_training_steps + 1))
        for t in range(n_training_steps):
            ax[row_index, i].plot(mean[t, :], label="step_" + str(t), color=colors[t])

    layer1 = "layer2"
    row_index = 1
    ax[row_index, 0].set_ylabel("Cosyne similarity L2")
    for i, model_id in enumerate(include_list):
        all_cosynes = []
        ax[row_index, i].set_title(model_id)
        for run in range(n_runs):
            try:
                activity = all_tuning_curves[model_id][run][layer1]
            except:
                print("file not found", model_id, run, layer1)
                continue
            cosyne_sim = cosyne_similarities(activity, n_training_stages)
            all_cosynes.append(np.log(cosyne_sim))
        all_cosynes = np.stack(all_cosynes, axis=0)
        mean, std = get_mean_std(all_cosynes)
        n_training_steps = mean.shape[0]
        colors = plt.cm.viridis(np.linspace(0, 1, n_training_steps + 1))
        for t in range(n_training_steps):
            ax[row_index, i].plot(mean[t, :], label="step_" + str(t), color=colors[t])

    plt.tight_layout()
    plt.show()
    print("debugging")

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