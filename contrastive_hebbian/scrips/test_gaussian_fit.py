import numpy as np
import matplotlib.pyplot as plt
from vpl_model.utils import infer_periodic_gaussian_params, generate_tuned_weights, periodic_kernel
from vpl_model.networks import FlexibleLearningRuleNet


def test_kernel_params():

    input_size = 80
    hidden_dim = 4
    angles = np.linspace(0, 180, input_size)
    tuned_neurons_width = 7.5
    offset = 0

    W1_0 = generate_tuned_weights(input_size,
                                  hidden_dim=hidden_dim,
                                  angles=angles,
                                  tuning_width=tuned_neurons_width,
                                  offset=offset)

    print(W1_0.shape)

    for i in range(hidden_dim):
        plt.plot(angles, W1_0[i, :], color="C"+str(i), alpha=0.5)
        inferred_mu, inferred_sigma = infer_periodic_gaussian_params(angles, W1_0[i, :])
        plt.plot(angles, periodic_kernel(angles, inferred_mu, sigma=inferred_sigma, period=180),
                 "--", color="C"+str(i))
    plt.show()


def see_activity_tuning():
    input_size = 160
    hidden_dim1 = 8
    hidden_dim2 = 8
    hidden_dim3 = 8
    angles = np.linspace(0, 180, input_size)
    h_angles = np.linspace(0, 180, num=hidden_dim1)
    tuned_neurons_width = 10.0
    non_tuned_width = 7.5#2.0
    offset = 0

    W1_0 = generate_tuned_weights(input_size,
                                  hidden_dim=hidden_dim1,
                                  angles=angles,
                                  tuning_width=tuned_neurons_width,
                                  offset=offset)

    W2_0 = generate_tuned_weights(hidden_dim1,
                                  hidden_dim=hidden_dim2,
                                  angles=h_angles,
                                  tuning_width=non_tuned_width,
                                  offset=offset)

    W3_0 = generate_tuned_weights(hidden_dim2,
                                  hidden_dim=hidden_dim3,
                                  angles=h_angles,
                                  tuning_width=non_tuned_width,
                                  offset=offset)

    W_list = [W1_0, W2_0, W3_0]
    W_feedback_list = [np.random.normal(size = W.shape) for W in W_list]

    net = FlexibleLearningRuleNet(W_list, W_feedback_list, eta=0.0,
                                   learning_rate=0.001)

    activity_per_layer = net.get_tunning_curves()

    net = MultiLayerContrastiveNet(W_list, gamma=0.0, eta=0.0,
                                   learning_rate=0.001, normalize_weights=True)

    normalized_activity_per_layer = net.get_tunning_curves()

    f, ax = plt.subplots(3, 3, figsize=(15, 10))
    for i, activity in enumerate(activity_per_layer):
        print(activity.shape)
        for h in range(hidden_dim1):
            if i == 0:
                ax[0, i].plot(angles, W_list[i][h, :])
            else:
                ax[0, i].plot(h_angles, W_list[i][h, :])
            ax[1, i].plot(angles, activity[h, :])
            ax[2, i].plot(angles, normalized_activity_per_layer[i][h, :])
    plt.show()

    layer_sigma = []
    for layer in np.arange(3):
        inferred_sigma_list = []
        for i in range(hidden_dim1):
            inferred_mu, inferred_sigma = infer_periodic_gaussian_params(angles,
                                                                         normalized_activity_per_layer[layer][i,:])
            inferred_sigma_list.append(inferred_sigma)
        layer_sigma.append(np.mean(inferred_sigma_list))
    print(layer_sigma)

if __name__ == "__main__":
    # test_kernel_params()
    see_activity_tuning()