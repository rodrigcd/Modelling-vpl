import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from contrastive_hebbian_net import ContrastiveNet, generate_tuned_weights
from angle_discrimination_task import AngleDiscriminationTask
from multi_layer_contrastive import MultiLayerContrastiveNet
import pickle
import copy


def main(seed):
    # Task parameters
    np.random.seed(seed=seed)
    # Task parameters
    training_orientation = 90.0
    orientation_diff = 1
    input_size = 100
    signal_amp = 1.0
    signal_bandwidth = 15
    output_amp = 1.0
    save_every = 100

    # Model parameters
    hidden_dim = 60
    learning_rate = 1e-3
    test_epochs = 10
    epochs = 7001
    tuned_neurons_width = 10
    weight_reg = 0.1

    curriculums = [30, 20, 10, 1]

    all_regimes = {"gradient_descent": {"gamma": 0.0, "eta": 0.0},
                   "contrastive": {"gamma": 1.0, "eta": 0.0},
                   "quasi_predictive": {"gamma": -1.0, "eta": 0.0},
                   "hebbian": {"gamma": 0, "eta": 0.01},
                   "anti_hebbian": {"gamma": 0, "eta": -0.01},
                   "contrastive_hebb": {"gamma": 1.0, "eta": 0.005},
                   "contrastive_anti_hebb": {"gamma": 1.0, "eta": -0.005},
                   "quasi_predictive_hebb": {"gamma": -1.0, "eta": 0.005},
                   "quasi_predictive_anti_hebb": {"gamma": -1.0, "eta": -0.005},}

    data = AngleDiscriminationTask(training_orientation=training_orientation,
                                   orientation_diff=curriculums[0],
                                   input_size=input_size,
                                   signal_amp=signal_amp,
                                   signal_bandwidth=signal_bandwidth,
                                   output_amp=output_amp)

    h_angles = np.linspace(0, 180, num=hidden_dim)

    W1_0 = generate_tuned_weights(input_size, hidden_dim=hidden_dim, angles=data.angles,
                                  tuning_width=tuned_neurons_width)
    W2_0 = generate_tuned_weights(hidden_dim, hidden_dim=hidden_dim, angles=h_angles, tuning_width=tuned_neurons_width)
    W3_0 = generate_tuned_weights(hidden_dim, hidden_dim=hidden_dim, angles=h_angles, tuning_width=tuned_neurons_width)
    W4_0 = np.zeros((data.output_size, hidden_dim))

    W_list = [W1_0, W2_0, W3_0, W4_0]

    regime_results = {}
    curr_index = 0
    for key, values in all_regimes.items():
        aux_dict = {}
        total_loss = []
        grad_list = []
        weight_list = []

        net = MultiLayerContrastiveNet(W_list, gamma=values["gamma"], eta=values["eta"], learning_rate=learning_rate)

        for i in tqdm(range(epochs)):
            if i % int(epochs/len(curriculums)) == 0 and i != 0:
                curr_index += 1
                data = AngleDiscriminationTask(training_orientation=training_orientation,
                                               orientation_diff=curriculums[curr_index],
                                               input_size=input_size,
                                               signal_amp=signal_amp,
                                               signal_bandwidth=signal_bandwidth,
                                               output_amp=output_amp)
            if i % save_every == 0:
                weight_list.append(copy.deepcopy(net.get_numpy_weights()))
            x, y = data.full_batch()
            loss, all_updates = net.update(x, y)
            total_loss.append(loss)
            if i % save_every == 0:
                grad_list.append(all_updates)

        print("change in weights", np.logical_not(np.equal(weight_list[0][0], weight_list[-1][0]).all()))
        aux_dict["loss"] = np.array(total_loss)
        aux_dict["learned_W"] = weight_list
        aux_dict["W_grad"] = grad_list
        aux_dict["save_every"] = save_every
        aux_dict["data"] = data
        aux_dict["learning_rate"] = learning_rate
        aux_dict["tuned_neurons_width"] = tuned_neurons_width
        regime_results[key] = aux_dict

    return regime_results


if __name__ == "__main__":
    n_runs = 5
    save_path = "long_local/"
    for i in range(n_runs):
        results = main(seed=i)
        pickle.dump(results, open(save_path + "multi_layer_run_" + str(i) + ".pkl", "wb"))