import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd
import pickle as pkl
import os
from vpl_model.utils import infer_periodic_gaussian_params, probe_tuning_curve
from vpl_model.networks import sigmoid_output_forward_path


def load_ex_data(results_path, seed=0):
    path_list = glob.glob(results_path + "flexnet_*")
    eg_data = pd.read_pickle(path_list[seed])
    params = pd.read_pickle(results_path + "params_" + str(seed) + ".pkl")
    print("available keys:", eg_data.keys())
    return eg_data, params


def get_loss(data, ax=None, make_plot=True, add_title=False):
    loss_dict = {}
    for i, key in enumerate(data.keys()):
        curriculum_switch = data[key]["curriculum_switch"]
        loss = data[key]["loss"]
        loss_dict[key] = loss
        if make_plot:
            ax[i].plot(loss, label=key)
            for switch in curriculum_switch:
                ax[i].axvline(x=switch, color="black", linestyle="--")
            if add_title:
                ax[i].set_title(key)
            if i == 0:
                ax[i].set_ylabel("Cross entropy loss")
            ax[i].set_xlabel("Training steps")
    return loss_dict, ax


def get_accuracy(data, ax=None, make_plot=True, add_title=False):
    accuracy_dict = {}
    for i, key in enumerate(data.keys()):
        curriculum_switch = data[key]["curriculum_switch"]
        accuracy = data[key]["accuracy"]
        accuracy_dict[key] = accuracy
        if make_plot:
            ax[i].plot(accuracy, label=key)
            for switch in curriculum_switch:
                ax[i].axvline(x=switch, color="black", linestyle="--")
            if add_title:
                ax[i].set_title(key)
            if i == 0:
                ax[i].set_ylabel("Accuracy")
            ax[i].set_xlabel("Training steps")
    return accuracy_dict, ax


def get_tuning_curves(data, pre_switch=True, use_pre_activity=False):
    act1_dict = {}
    act2_dict = {}
    act3_dict = {}
    for key in data.keys():
        if pre_switch:
            W1 = data[key]["switch_W1"]
            W2 = data[key]["switch_W2"]
            W3 = data[key]["switch_W3"]
            W1_0 = data[key]["learned_W1"][0, ...]
            W2_0 = data[key]["learned_W2"][0, ...]
            W3_0 = data[key]["learned_W3"][0, ...]
            W1 = np.concatenate([W1_0[np.newaxis, ...], W1], axis=0)
            W2 = np.concatenate([W2_0[np.newaxis, ...], W2], axis=0)
            W3 = np.concatenate([W3_0[np.newaxis, ...], W3], axis=0)
        else:
            W1 = data[key]["learned_W1"]
            W2 = data[key]["learned_W2"]
            W3 = data[key]["learned_W3"]
        training_steps = W1.shape[0]
        input_dim = W1.shape[-1]
        forward_func = sigmoid_output_forward_path
        act1_list = []
        act2_list = []
        act3_list = []
        for i in range(training_steps):
            probing_angles = np.eye(input_dim)
            W_list = [W1[i, ...], W2[i, ...], W3[i, ...]]
            activity_per_layer, pre_activity = probe_tuning_curve(W_list, probing_angles, forward_func)
            if use_pre_activity:
                act1_list.append(pre_activity[0])
                act2_list.append(pre_activity[1])
                act3_list.append(pre_activity[2])
            else:
                act1_list.append(activity_per_layer[0])
                act2_list.append(activity_per_layer[1])
                act3_list.append(activity_per_layer[2])
        act1_dict[key] = np.stack(act1_list)
        act2_dict[key] = np.stack(act2_list)
        act3_dict[key] = np.stack(act3_list)
    return act1_dict, act2_dict, act3_dict


def plot_tuning_curves(act_dict, n_cells=1, ax=None, ylabel="", add_title=False):
    for i, key in enumerate(act_dict.keys()):
        act = act_dict[key]
        n_training_stages = act.shape[0]
        colors = plt.cm.viridis(np.linspace(0, 1, n_training_stages+1))
        for t in range(n_training_stages):
            label = f"stage {t}"
            for cell in range(n_cells):
                x = np.linspace(0, 180, act.shape[-1])
                ax[i].plot(x, act[t, cell, :], color=colors[t], label=label)
        if add_title:
            ax[i].set_title(key)
        if i == 0:
            ax[i].set_ylabel(ylabel)
            ax[i].legend()
        ax[i].axvline(x=90, color="black", linestyle="--")
        ax[i].set_xlabel("Angle (deg)")


def get_angle_bandwidth_through_time(act_dict):
    bw_dict = {}
    mean_dict = {}
    for key in act_dict.keys():
        act = act_dict[key]
        n_training_stages = act.shape[0]
        n_cells = act.shape[1]
        bw_t = []
        mean_t = []
        for t in range(n_training_stages):
            bw = []
            mean = []
            for cell in range(n_cells):
                mu, sigma = infer_periodic_gaussian_params(np.linspace(0, 180, act.shape[-1]), act[t, cell, :])
                bw.append(sigma)
                mean.append(mu)
            bw_t.append(np.array(bw))
            mean_t.append(np.array(mean))
        bw_dict[key] = np.stack(bw_t)
        mean_dict[key] = np.stack(mean_t)
    return mean_dict, bw_dict


def plot_bandwidth_evolution(bw_dict, data, params, ax=None, add_title=False, ylabel="Bandwidth (deg)"):
    for i, key in enumerate(bw_dict.keys()):
        curriculum_switch = data[key]["curriculum_switch"]
        bw = bw_dict[key]
        n_training_stages = bw.shape[0]
        n_cells = bw.shape[1]
        mean_bw = np.mean(bw, axis=1)
        std_bw = np.std(bw, axis=1)
        ax[i].plot(np.arange(n_training_stages)*params["save_every"], mean_bw)
        ax[i].fill_between(np.arange(n_training_stages)*params["save_every"], mean_bw - 2*std_bw/n_cells,
                           mean_bw + 2*std_bw/n_cells,
                           alpha=0.3)
        for switch in curriculum_switch:
            ax[i].axvline(x=switch, color="black", linestyle="--")
        if add_title:
            ax[i].set_title(key)
        if i == 0:
            ax[i].set_ylabel(ylabel)
        ax[i].set_xlabel("Training 0 steps")


def plot_bandwidth_distr(bw_list, ax=None, add_title=False, ylabel="Norm freq", from_stage=1):
    for i, key in enumerate(bw_list.keys()):
        bw = bw_list[key]
        training_stages = bw.shape[0]
        colors = plt.cm.viridis(np.linspace(0, 1, training_stages+1))
        for t in range(from_stage, training_stages):
            h, bins = np.histogram(bw[t, ...], bins=8, density=True)
            ax[i].plot(bins[:-1], h, color=colors[t], label=f" stage {t}")
        if add_title:
            ax[i].set_title(key)
        if i == 0:
            ax[i].legend()
            ax[i].set_ylabel(ylabel)
        ax[i].set_xlabel("Bandwidth (deg)")


def tuning_curve_per_stage(act_dict, ax=None, ylabel="", add_title=False):
    for i, key in enumerate(act_dict.keys()):
        act = act_dict[key]
        n_training_stages = act.shape[0]
        n_cells = act.shape[1]
        colors = plt.cm.viridis(np.linspace(0, 1, n_training_stages+1))

        for t in range(n_training_stages):
            label = f"stage {t}"
            average_activity = []
            for cell in range(n_cells):
                n_angles = act.shape[-1]
                activity_max = np.argmax(act[t, cell, :])
                shift = int(n_angles/2 - activity_max)
                shifted_activity = np.roll(act[t, cell, :], shift)
                average_activity.append(shifted_activity/np.max(shifted_activity))
            average_activity = np.mean(np.stack(average_activity, axis=0), axis=0)
            x = np.linspace(-90, 90, act.shape[-1])
            ax[i].plot(x, average_activity,
                       color=colors[t], label=label)
        if add_title:
            ax[i].set_title(key)
        if i == 0:
            ax[i].set_ylabel(ylabel)
            ax[i].legend()
        ax[i].axvline(x=0, color="black", linestyle="--")
        ax[i].set_xlabel("Angle (deg)")
    return ax


def slope_PO_TO(act_dict, ax=None, normalize_tuning=False, TO=90.0, add_title=False, ylabel="Slope at TO"):

    for i, key in enumerate(act_dict.keys()):
        act = act_dict[key]
        n_training_stages = act.shape[0]
        n_cells = act.shape[1]
        colors = plt.cm.viridis(np.linspace(0, 1, n_training_stages+1))
        angles = np.linspace(-90, 90, act.shape[-1])
        for t in range(n_training_stages):
            label = f"stage {t}"
            average_activity = []
            all_slopes = []
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
            ax[i].plot(x_axis, y_axis, color=colors[t], label=label)
        ax[i].set_xlabel("PO - TO")
        if add_title:
            ax[i].set_title(key)

        if i == 0:
            ax[i].legend()
            ax[i].set_ylabel(ylabel)


if __name__ == "__main__":
    results_path = "../all_results/curr_flexnet/"
    eg_data, params = load_ex_data(results_path, seed=1)
    re_compute = False

    """ Pre-computed data """
    if re_compute:
        print("updating pre-computed data...")
        act1_dict, act2_dict, act3_dict = get_tuning_curves(eg_data)
        # Change in bandwidth through time
        full_act1_dict, full_act2_dict, full_act3_dict = get_tuning_curves(eg_data, pre_switch=False)
        angle_list, bw_list = [], []

        for i, act_dict in enumerate([full_act1_dict, full_act2_dict, full_act3_dict]):
            print("activity layer", i)
            mean_dict, bw_dict = get_angle_bandwidth_through_time(act_dict)
            angle_list.append(mean_dict)
            bw_list.append(bw_dict)

        curr_angle_list = []
        curr_bw_list = []
        for i, act_dict in enumerate([act1_dict, act2_dict, act3_dict]):
            print("activity layer", i)
            mean_dict, bw_dict = get_angle_bandwidth_through_time(act_dict)
            curr_angle_list.append(mean_dict)
            curr_bw_list.append(bw_dict)

        # Save pre-computed data
        save_path = "../all_results/curr_flexnet"
        data_dict = {"act1_dict": act1_dict, "act2_dict": act2_dict, "act3_dict": act3_dict,
                     "full_act1_dict": full_act1_dict, "full_act2_dict": full_act2_dict, "full_act3_dict": full_act3_dict,
                     "angle_list": angle_list, "bw_list": bw_list,
                     "curr_angle_list": curr_angle_list, "curr_bw_list": curr_bw_list}
        pkl.dump(data_dict, open(os.path.join(results_path, "pre_computed_data.pkl"), "wb"))

    pre_computed_data = pd.read_pickle(os.path.join(results_path, "pre_computed_data.pkl"))
    act1_dict = pre_computed_data["act1_dict"]
    act2_dict = pre_computed_data["act2_dict"]
    act3_dict = pre_computed_data["act3_dict"]
    angle_list = pre_computed_data["angle_list"]
    full_act1_dict = pre_computed_data["full_act1_dict"]
    full_act2_dict = pre_computed_data["full_act2_dict"]
    full_act3_dict = pre_computed_data["full_act3_dict"]
    bw_list = pre_computed_data["bw_list"]
    angle_list = pre_computed_data["angle_list"]
    curr_angle_list = pre_computed_data["curr_angle_list"]
    curr_bw_list = pre_computed_data["curr_bw_list"]

    """ First set of plots """
    make_plot = True
    panel_dim = (3, 3)
    if make_plot:
        f, ax = plt.subplots(panel_dim[0], panel_dim[1], figsize=(15, 7))
    else:
        ax = np.zeros(panel_dim)

    # Loss and accuracy
    loss_dict = get_loss(eg_data, ax[0, :], make_plot=make_plot, add_title=True)
    accuracy_dict = get_accuracy(eg_data, ax[1, :], make_plot=make_plot)

    preact1_dict, preact2_dict, preact3_dict = get_tuning_curves(eg_data, use_pre_activity=True)

    # Tuning curves at the decision layer
    plot_tuning_curves(act3_dict, n_cells=1, ax=ax[2, :], ylabel="Decision layer activity")

    if make_plot:
        plt.tight_layout()
        plt.show()

    """ Second set of plots """
    make_plot = True
    panel_dim = (4, 3)
    if make_plot:
        f, ax = plt.subplots(panel_dim[0], panel_dim[1], figsize=(15, 7))
    else:
        ax = np.zeros(panel_dim)

    tuning_curve_per_stage(act1_dict, ax[0, :], ylabel="Layer 1 tuning curves", add_title=True)
    plot_bandwidth_evolution(angle_list[0], eg_data, params, ax[2, :], add_title=True, ylabel="Layer 1 preferred ori")
    tuning_curve_per_stage(act2_dict, ax[1, :], ylabel="Layer 2 tuning curves", add_title=True)
    plot_bandwidth_evolution(angle_list[1], eg_data, params, ax[3, :], add_title=True, ylabel="Layer 2 preferred ori")

    if make_plot:
        plt.tight_layout()
        plt.show()

    """ Third set of plots """
    make_plot = True
    panel_dim = (4, 3)
    if make_plot:
        f, ax = plt.subplots(panel_dim[0], panel_dim[1], figsize=(15, 7))
    else:
        ax = np.zeros(panel_dim)

    plot_bandwidth_evolution(bw_list[0], eg_data, params, ax[0, :], add_title=True, ylabel="Layer 1 bandwidth (deg)")
    plot_bandwidth_distr(curr_bw_list[0], ax[1, :], ylabel="Layer 1 Norm freq")
    plot_bandwidth_evolution(bw_list[1], eg_data, params, ax[2, :], ylabel="Layer 2 bandwidth (deg)")
    plot_bandwidth_distr(curr_bw_list[1], ax[3, :], ylabel="Layer 2 Norm freq")

    if make_plot:
        plt.tight_layout()
        plt.show()

    """ Fourth set of plots """
    make_plot = True
    panel_dim = (2, 3)
    if make_plot:
        f, ax = plt.subplots(panel_dim[0], panel_dim[1], figsize=(15, 7))
    else:
        ax = np.zeros(panel_dim)

    slope_PO_TO(act1_dict, ax[0, :], normalize_tuning=True, ylabel="Slope at TO layer 1")
    slope_PO_TO(act2_dict, ax[1, :], normalize_tuning=True, ylabel="Slope at TO layer 2")
    if make_plot:
        plt.tight_layout()
        plt.show()

    print("debugging")
