import numpy as np
from tqdm import tqdm
import pickle
import jax.numpy as jnp
import copy

from vpl_model.networks import FlexibleLearningRuleNet
from vpl_model.utils import generate_tuned_weights, check_dir
from vpl_model.tasks import AngleDiscriminationTask
from vpl_model.neural_data import DataHandler


hebbian_eta = 0.0005
sgd_lr = 0.005
feedback_alignment_lr = 0.005
predictive_coding_lr = 0.2
FWHM = 35.0

params = {"training_orientation": 90.0,
          "orientation_diff": 8,
          "input_size": 180,
          "signal_amp": 1.0,
          "signal_bandwidth": 10,
          "output_amp": 1.0,
          "save_every": 10,
          "hidden_dim1": 120,
          "hidden_dim2": 120,
          "pc_inner_loops": 10,
          "test_iters": 5000,
          "train_iters": 5000,
          "tuned_neurons_width": FWHM / 2.35482,
          "un_tuned_neurons_width": 1.0,
          "PC_activity_learning": 0.1,
          "lr_W2_W1": 1.0,
          "normalize_weights": True,
          "output_size": 1,
          "curriculums": [8, 5, 3, 2, 1],  # orientation_diff per training stage
          "target_accuracy": 0.75,
          "minimum_steps_per_stage": 100,
          "df_path": "/nfs/nhome/live/rcdavis/perception_task/tmp_data/20240225031102_orituneData.pkl",
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
    data = AngleDiscriminationTask(training_orientation=params["training_orientation"],
                                   orientation_diff=params["curriculums"][0],
                                   input_size=params["input_size"],
                                   signal_amp=params["signal_amp"],
                                   signal_bandwidth=params["signal_bandwidth"],
                                   output_amp=params["output_amp"],
                                   single_output=True)

    data_handler = DataHandler(params["df_path"])

    offset = np.random.normal()
    W1_0, sampled_ori, sampled_bandwidths = data_handler.angle_bandwidth_generated_weights(params["input_size"],
                                                                                           params["hidden_dim1"],
                                                                                           data.angles)
    # W1_0 = generate_tuned_weights(params["input_size"],
    #                               hidden_dim=params["hidden_dim1"],
    #                               angles=data.angles,
    #                               tuning_width=params["tuned_neurons_width"],
    #                               offset=offset)

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

    W_list = [W1_0, W2_0, W3_0]
    W_feedback_list = [np.random.normal(size=W.shape) for W in W_list]

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
                                  normalize_weights=params["normalize_weights"])

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
        accuracy.append(h[-1][0, 0])
        total_loss.append(loss)
        if i % params["save_every"] == 0:
            grad1_list.append(all_updates[0])
            grad2_list.append(all_updates[1])
            grad3_list.append(all_updates[2])

        if accuracy[-1] > params["target_accuracy"] and curr_inner_iters > params["minimum_steps_per_stage"]:
            curr_inner_iters = 0
            switch_W1_list.append(W1_list[-1])
            switch_W2_list.append(W2_list[-1])
            switch_W3_list.append(W3_list[-1])
            print("target accuracy reached")
            print("iteration", i)
            print("accuracy", accuracy[-1])
            curriculum_switch.append(i)
            if curr_index >= len(params["curriculums"])-1:
                print("curriculum", params["curriculums"][curr_index], "Finished")
                break
            else:
                print("curriculum", params["curriculums"][curr_index], "to", params["curriculums"][curr_index + 1])
            curr_index += 1
            data = AngleDiscriminationTask(training_orientation=params["training_orientation"],
                                           orientation_diff=params["curriculums"][curr_index],
                                           input_size=params["input_size"],
                                           signal_amp=params["signal_amp"],
                                           signal_bandwidth=params["signal_bandwidth"],
                                           output_amp=params["output_amp"],
                                           single_output=True)
            print("iteration", i, "new curriculum", params["curriculums"][curr_index])

        curr_inner_iters += 1

    aux_dict["loss"] = np.array(total_loss)
    aux_dict["learned_W1"] = np.array(jnp.stack(W1_list))
    aux_dict["learned_W2"] = np.array(jnp.stack(W2_list))
    aux_dict["learned_W3"] = np.array(jnp.stack(W3_list))
    aux_dict["W1_grad"] = np.array(grad1_list)
    aux_dict["W2_grad"] = np.array(grad2_list)
    aux_dict["W3_grad"] = np.array(grad3_list)
    aux_dict["switch_W1"] = np.array(jnp.stack(switch_W1_list))
    aux_dict["switch_W2"] = np.array(jnp.stack(switch_W2_list))
    aux_dict["switch_W3"] = np.array(jnp.stack(switch_W3_list))
    aux_dict["accuracy"] = np.array(accuracy)
    aux_dict["curriculum_switch"] = curriculum_switch
    aux_dict["data"] = data
    regime_results[key] = aux_dict
    return regime_results, params


if __name__ == "__main__":
    n_runs = 3
    save_path = "../all_results/curr_flexnet/"
    check_dir(save_path)
    for i in range(n_runs):
        print("seed", i)
        results, params = main(seed=i)
        pickle.dump(results, open(save_path + "flexnet_run_" + str(i) + ".pkl", "wb"))
        pickle.dump(params, open(save_path + "params_" + str(i) + ".pkl", "wb"))
