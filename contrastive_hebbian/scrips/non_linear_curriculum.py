import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pickle
import jax.numpy as jnp
import copy

from vpl_model.networks import ContrastiveNet, MultiLayerContrastiveNet
from vpl_model.utils import generate_tuned_weights, check_dir
from vpl_model.tasks import AngleDiscriminationTask


def main(seed=0):
    np.random.seed(seed=seed)

    params = {"seed": seed,
              "training_orientation": 90.0,
              "orientation_diff": 1,
              "input_size": 80,
              "signal_amp": 1.0,
              "signal_bandwidth": 15,
              "output_amp": 1.0,
              "save_every": 10,
              "hidden_dim": 40,
              "learning_rate": 0.001,
              "test_iters": 10,
              "train_iters": 10000,
              "tuned_neurons_width": 10,
              "lr_W2_W1": 1.0,
              "curriculums": [20, 10, 5, 1]  # orientation_diff per training stage
              }

    all_regimes = {"gradient_descent": {"gamma": 0.0, "eta": 0.0},
                   "contrastive": {"gamma": 1.0, "eta": 0.0},
                   "quasi_predictive": {"gamma": -1.0, "eta": 0.0},
                   "hebbian": {"gamma": 0, "eta": 0.01},
                   "anti_hebbian": {"gamma": 0, "eta": -0.01},
                   "contrastive_hebb": {"gamma": 1.0, "eta": 0.005},
                   "contrastive_anti_hebb": {"gamma": 1.0, "eta": -0.005},
                   "quasi_predictive_hebb": {"gamma": -1.0, "eta": 0.005},
                   "quasi_predictive_anti_hebb": {"gamma": -1.0, "eta": -0.005},}

    data = AngleDiscriminationTask(training_orientation=params["training_orientation"],
                                   orientation_diff=params["curriculums"][0],
                                   input_size=params["input_size"],
                                   signal_amp=params["signal_amp"],
                                   signal_bandwidth=params["signal_bandwidth"],
                                   output_amp=params["output_amp"])

    offset = np.random.normal()
    W1_0 = generate_tuned_weights(params["input_size"],
                                  hidden_dim=params["hidden_dim"],
                                  angles=data.angles,
                                  tuning_width=params["tuned_neurons_width"],
                                  offset=offset)

    W2_0 = np.zeros((int(data.output_size), int(params["hidden_dim"])))
    params["W1_0"] = W1_0.copy()
    params["W2_0"] = W2_0.copy()
    W_list = [W1_0, W2_0]

    regime_results = {}
    for key, values in all_regimes.items():
        aux_dict = {}
        total_loss = []
        grad1_list = []
        grad2_list = []
        W1_list = []
        W2_list = []

        net = MultiLayerContrastiveNet(W_list, gamma=values["gamma"], eta=values["eta"], learning_rate=params["learning_rate"])

        curr_index = 0
        for i in tqdm(range(params["train_iters"])):

            if i % int(params["train_iters"]/len(params["curriculums"])) == 0:
                data = AngleDiscriminationTask(training_orientation=params["training_orientation"],
                                               orientation_diff=params["curriculums"][curr_index],
                                               input_size=params["input_size"],
                                               signal_amp=params["signal_amp"],
                                               signal_bandwidth=params["signal_bandwidth"],
                                               output_amp=params["output_amp"])
                print("iteration", i, "new curriculum", params["curriculums"][curr_index])
                curr_index += 1

            if i % params["save_every"] == 0:
                W1_list.append(copy.deepcopy(np.array(net.W_list[0])))
                W2_list.append(copy.deepcopy(np.array(net.W_list[1])))

            x, y = data.full_batch()
            loss, all_updates = net.update(x, y)
            total_loss.append(loss)
            if i % params["save_every"] == 0:
                grad1_list.append(all_updates[0])
                grad2_list.append(all_updates[1])

        aux_dict["loss"] = np.array(total_loss)
        aux_dict["learned_W1"] = np.array(jnp.stack(W1_list))
        aux_dict["learned_W2"] = np.array(jnp.stack(W2_list))
        aux_dict["W1_grad"] = np.array(grad1_list)
        aux_dict["W2_grad"] = np.array(grad2_list)
        aux_dict["save_every"] = params["save_every"]
        aux_dict["data"] = data
        aux_dict["learning_rate"] = params["learning_rate"]
        aux_dict["tuned_neurons_width"] = params["tuned_neurons_width"]
        regime_results[key] = aux_dict
    return regime_results, params


if __name__ == "__main__":
    n_runs = 3
    save_path = "../all_results/non_linear_tanh_50000/"
    check_dir(save_path)
    for i in range(n_runs):
        print("seed", i)
        results, params = main(seed=i)
        pickle.dump(results, open(save_path + "contrastive_run_" + str(i) + ".pkl", "wb"))
        pickle.dump(params, open(save_path + "params_" + str(i) + ".pkl", "wb"))
