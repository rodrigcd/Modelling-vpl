import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pickle
import jax.numpy as jnp
import copy

from contrastive_hebbian_net import ContrastiveNet, generate_tuned_weights
from angle_discrimination_task import AngleDiscriminationTask


def main(seed=0):
    np.random.seed(seed=seed)

    # Task parameters
    training_orientation = 90.0
    orientation_diff = 1
    input_size = 80
    signal_amp = 1.0
    signal_bandwidth = 15
    output_amp = 1.0
    save_every = 10

    # Model parameters
    hidden_dim = 40
    learning_rate = 0.001
    test_epochs = 10
    epochs = 20000
    tuned_neurons_width = 10
    lr_W2_W1 = 1.0

    curriculums = [20, 10, 5, 1]

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
                                   orientation_diff=orientation_diff,
                                   input_size=input_size,
                                   signal_amp=signal_amp,
                                   signal_bandwidth=signal_bandwidth,
                                   output_amp=output_amp)

    offset = np.random.normal()
    W1_0 = generate_tuned_weights(input_size, hidden_dim=hidden_dim, angles=data.angles,
                                  tuning_width=tuned_neurons_width,
                                  offset=offset)
    W2_0 = np.zeros((int(data.output_size), hidden_dim))

    regime_results = {}
    for key, values in all_regimes.items():
        aux_dict = {}
        total_loss = []
        grad1_list = []
        grad2_list = []
        W1_list = []
        W2_list = []

        net = ContrastiveNet(W1_0=W1_0, W2_0=W2_0, gamma=values["gamma"], eta=values["eta"],
                             learning_rate=learning_rate,
                             lr_W2_W1=lr_W2_W1)

        curr_index = 0
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
                W1_list.append(copy.deepcopy(np.array(net.W1)))
                W2_list.append(copy.deepcopy(np.array(net.W2)))

            x, y = data.full_batch()
            # print(x.shape, y.shape)
            h_ff, y_hat = net.forward(x)
            # print(h_ff.shape, y_hat.shape)
            W1_grad, W2_grad = net.update(x, y, y_hat, h_ff)

            if i % save_every == 0:
                grad1_list.append(W1_grad)
                grad2_list.append(W2_grad)

            total_loss.append(net.loss(y_hat, y))

        #RSM = h_ff[:, :, 0] @ h_ff[:, :, 0].T
        #aux_dict["RSM"] = RSM
        aux_dict["loss"] = np.array(total_loss)
        aux_dict["learned_W1"] = np.array(jnp.stack(W1_list))
        aux_dict["learned_W2"] = np.array(jnp.stack(W2_list))
        aux_dict["W1_grad"] = np.array(grad1_list)
        aux_dict["W2_grad"] = np.array(grad2_list)
        aux_dict["save_every"] = save_every
        aux_dict["data"] = data
        aux_dict["learning_rate"] = learning_rate
        aux_dict["tuned_neurons_width"] = tuned_neurons_width
        regime_results[key] = aux_dict
    return regime_results


if __name__ == "__main__":
    n_runs = 5
    save_path = "long_curriculum/"
    for i in range(n_runs):
        print("seed", i)
        results = main(seed=i)
        pickle.dump(results, open(save_path + "contrastive_run_" + str(i) + ".pkl", "wb"))