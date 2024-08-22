import numpy as np
from tqdm import tqdm
import pickle
import jax.numpy as jnp
import copy

from vpl_model.networks import FlexibleLearningRuleNet
from vpl_model.utils import generate_tuned_weights, check_dir
from vpl_model.tasks import AngleDiscriminationTask
from vpl_model.neural_data import DataHandler


hebbian_eta = 1e-5
sgd_lr = 2e-4
feedback_alignment_lr = 2e-4
predictive_coding_lr = 0.2
FWHM = 35.0

params = {"training_orientation": 90.0,
          "orientation_diff": 4.0,
          "input_size": 180,
          "signal_amp": 1.0,
          "signal_bandwidth": 10,
          "output_amp": 1.0,
          "task_mode": "regression",
          "L1_reg": 0.0,
          "L2_reg": 0.0,
          "save_every": 10,
          "hidden_dim1": 120,
          "hidden_dim2": 120,
          "pc_inner_loops": 30,
          "pc_inner_lr": 0.1,
          "test_iters": 5000,
          "train_iters": 5000,
          "tuned_neurons_width": FWHM / 2.35482,
          "un_tuned_neurons_width": 1.0,
          "PC_activity_learning": 0.1,
          "lr_W2_W1": 1.0,
          "normalize_weights": True,
          "target_accuracy": 0.75,  # not used in regression task mode
          "minimum_steps_per_stage": 100,  # not used in regression task mode
          "true_sgd_limit": False,
          "local_df_path": "/home/rodrigo/SSD/Projects/tmp_data/20240816203149_orituneData.pkl",
          "cluster_df_path": "/nfs/nhome/live/rcdavis/perception_task/tmp_data/20240816203149_orituneData.pkl",
          }

all_regimes = {
    "gradient_descent": {"sgd_lr": sgd_lr, "eta": 0.0,
                         "feedback_alignment_lr": 0.0,
                         "predictive_coding_lr": 0.0},
    "feedback_alignment": {"sgd_lr": 0.0, "eta": 0.0,
                           "feedback_alignment_lr": feedback_alignment_lr,
                           "predictive_coding_lr": 0.0},
    "predictive_coding": {"sgd_lr": 0.0, "eta": 0.0,
                          "feedback_alignment_lr": 0.0,
                          "predictive_coding_lr": predictive_coding_lr},
    "gradient_descent_hebbian": {"sgd_lr": sgd_lr, "eta": hebbian_eta,
                                 "feedback_alignment_lr": 0.0,
                                 "predictive_coding_lr": 0.0},
    "feedback_alignment_hebbian": {"sgd_lr": 0.0, "eta": hebbian_eta,
                                   "feedback_alignment_lr": feedback_alignment_lr,
                                   "predictive_coding_lr": 0.0},
    "predictive_coding_hebbian": {"sgd_lr": 0.0, "eta": hebbian_eta,
                                  "feedback_alignment_lr": 0.0,
                                  "predictive_coding_lr": predictive_coding_lr},
    "gradient_descent_anti_hebbian": {"sgd_lr": sgd_lr, "eta": -hebbian_eta,
                                      "feedback_alignment_lr": 0.0,
                                      "predictive_coding_lr": 0.0},
    "feedback_alignment_anti_hebbian": {"sgd_lr": 0.0, "eta": -hebbian_eta,
                                        "feedback_alignment_lr": feedback_alignment_lr,
                                        "predictive_coding_lr": 0.0},
    "predictive_coding_anti_hebbian": {"sgd_lr": 0.0, "eta": -hebbian_eta,
                                       "feedback_alignment_lr": 0.0,
                                       "predictive_coding_lr": predictive_coding_lr},
}


def main(model_id, seed=0):
    np.random.seed(seed=seed)
    params["seed"] = seed
    if params["task_mode"] == "classification":
        params["loss_func_id"] = "cross_entropy"
    elif params["task_mode"] == "regression":
        params["loss_func_id"] = "mse"
    else:
        raise ValueError("Invalid task mode")
    data = AngleDiscriminationTask(training_orientation=params["training_orientation"],
                                   orientation_diff=params["orientation_diff"],
                                   input_size=params["input_size"],
                                   signal_amp=params["signal_amp"],
                                   signal_bandwidth=params["signal_bandwidth"],
                                   output_amp=params["output_amp"],
                                   task_mode=params["task_mode"])

    data_handler = DataHandler(params["local_df_path"])

    offset = np.random.normal()
    W1_0, sampled_ori, sampled_bandwidths = data_handler.angle_bandwidth_generated_weights(params["input_size"],
                                                                                           params["hidden_dim1"],
                                                                                           data.angles)

    h_angles = np.linspace(0, 180, num=params["hidden_dim1"])
    W2_0, _, _ = generate_tuned_weights(params["hidden_dim1"],
                                        hidden_dim=params["hidden_dim2"],
                                        angles=h_angles,
                                        tuning_width=params["un_tuned_neurons_width"],
                                        offset=offset)

    W3_0 = np.zeros((int(data.output_size), int(params["hidden_dim2"])))

    params["W1_0"] = W1_0.copy()
    params["W2_0"] = W2_0.copy()
    params["W3_0"] = W3_0.copy()

    params["sampled_ori"] = sampled_ori
    params["sampled_bandwidths"] = sampled_bandwidths
    params["all_regimes"] = all_regimes

    W_list = [W1_0, W2_0, W3_0]
    W_feedback_list = [np.random.normal(size=W.shape) for W in W_list]
    params["W_feedback"] = W_feedback_list

    regime_results = {}
    key = model_id
    values = all_regimes[key]

    print("###### Regime", key, "######")
    aux_dict = {}
    total_loss = []
    grad1_list = []
    grad2_list = []
    grad3_list = []
    W1_list = []
    W2_list = []
    W3_list = []
    switch_W1_list = []
    switch_W2_list = []
    switch_W3_list = []
    accuracy = []
    curriculum_switch = []

    net = FlexibleLearningRuleNet(W_list,
                                  W_feedback_list,
                                  eta=values["eta"],
                                  sgd_lr=values["sgd_lr"],
                                  feedback_alignment_lr=values["feedback_alignment_lr"],
                                  predictive_coding_lr=values["predictive_coding_lr"],
                                  normalize_weights=params["normalize_weights"],
                                  PC_inner_steps=params["pc_inner_loops"],
                                  PC_inner_lr=params["pc_inner_lr"],
                                  L1_reg=params["L1_reg"],
                                  L2_reg=params["L2_reg"],
                                  loss_function=params["loss_func_id"],
                                  true_sgd_limit=params["true_sgd_limit"])

    curr_index = 0
    curr_inner_iters = 0
    switch_W1_list.append(W1_0)
    switch_W2_list.append(W2_0)
    switch_W3_list.append(W3_0)
    for i in tqdm(range(params["train_iters"])):

        if i % params["save_every"] == 0:
            W1_list.append(copy.deepcopy(np.array(net.W_list[0])))
            W2_list.append(copy.deepcopy(np.array(net.W_list[1])))
            W3_list.append(copy.deepcopy(np.array(net.W_list[2])))

        x, y = data.full_batch()
        loss, all_updates = net.update(x, y)
        h = net.forward(x[-1:, ...])
        total_loss.append(loss)

        if i % params["save_every"] == 0:
            grad1_list.append(all_updates[0])
            grad2_list.append(all_updates[1])
            grad3_list.append(all_updates[2])

    aux_dict["loss"] = np.array(total_loss)
    aux_dict["learned_W1"] = np.array(jnp.stack(W1_list))
    aux_dict["learned_W2"] = np.array(jnp.stack(W2_list))
    aux_dict["learned_W3"] = np.array(jnp.stack(W3_list))
    aux_dict["W1_grad"] = np.array(grad1_list)
    aux_dict["W2_grad"] = np.array(grad2_list)
    aux_dict["W3_grad"] = np.array(grad3_list)
    aux_dict["accuracy"] = np.array(accuracy)
    aux_dict["curriculum_switch"] = curriculum_switch
    aux_dict["data"] = data
    regime_results[key] = aux_dict
    return regime_results, params


if __name__ == "__main__":
    n_runs = 3
    save_path = "../../all_results/curr_flexnet/"
    check_dir(save_path)
    for i in range(n_runs):
        print("seed", i)
        results, params = main(seed=i)
        pickle.dump(results, open(save_path + "flexnet_run_" + str(i) + ".pkl", "wb"))
        pickle.dump(params, open(save_path + "params_" + str(i) + ".pkl", "wb"))
