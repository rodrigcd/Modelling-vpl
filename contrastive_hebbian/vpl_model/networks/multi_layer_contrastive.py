import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from vpl_model.tasks import SemanticTask


class MultiLayerContrastiveNet(object):

    def __init__(self, W_list, gamma, eta, learning_rate, weight_reg=1.0, normalize_weights=False):
        self.input_dim = W_list[0].shape[1]
        self.output_dim = W_list[-1].shape[0]
        self.gamma = gamma
        self.eta = eta
        self.n_layers = len(W_list)
        self.learning_rate = learning_rate
        self.W_list = [jax.device_put(W) for W in W_list]
        self.sgd_func = grad(mse_loss, argnums=2)
        self.weight_reg = 1.0
        self.normalize_weights = normalize_weights
        if self.normalize_weights:
            self.normalize_activity_weights()

    def forward(self, x):
        h_ff = forward_path(self.W_list, x, self.n_layers)
        return h_ff

    def update(self, x, y):
        h_list = forward_path(self.W_list, x, self.n_layers)
        y_hat = h_list[-1]
        loss = self.loss_func(y_hat, y)

        hebbian_updates = self.get_hebbian_updates(h_list, x)
        sgd_updates = self.sgd_func(x, y, self.W_list, self.n_layers)
        cont_update = vmap(contrastive_update, in_axes=(None, 0, 0, None, None))(self.W_list[-1], y, y_hat, self.gamma,
                                                                                  self.learning_rate)
        all_updates = []
        for i in range(self.n_layers):
            if i == self.n_layers - 1:
                update = jnp.mean(cont_update, axis=0) - sgd_updates[i] * self.learning_rate
                self.W_list[i] = self.W_list[i] + update
                all_updates.append(np.array(update))
            else:
                update = jnp.mean(hebbian_updates[i], axis=0) - sgd_updates[i] * self.learning_rate
                self.W_list[i] = self.W_list[i] + update
                all_updates.append(np.array(update))
        return np.array(loss), all_updates

    def mse_loss(self, y_hat, y):
        return jnp.mean((y_hat - y)**2)/2

    def get_numpy_weights(self):
        return [np.array(W) for W in self.W_list]

    def get_hebbian_updates(self, h_list, x):
        all_updates = []
        h_1 = x
        for i in range(self.n_layers):
            h = h_list[0]
            update = vmap(hebbian_update, in_axes=(None, 0, 0, None))(self.W_list[i], h_list[i], h_1, self.eta)
            all_updates.append(update)
            h_1 = h
        return all_updates

    def get_tunning_curves(self):
        probing_angles = jnp.eye(self.input_dim)
        activity_per_layer = forward_path(self.W_list, probing_angles, self.n_layers)
        return activity_per_layer

    def normalize_activity_weights(self):
        for i, W in enumerate(self.W_list):
            activity_per_layer = self.get_tunning_curves()
            normalized_weights = W/np.amax(activity_per_layer[i], axis=1)[:, np.newaxis]
            self.W_list[i] = normalized_weights


#@jit
def mse_loss(x, y, W_list, n_layers):
    h = forward_path(W_list, x, n_layers)
    y_hat = h[-1]
    return jnp.mean((y_hat - y)**2)/2


def cross_entropy_loss(x, y, W_list, n_layers):
    h = forward_path(W_list, x, n_layers)
    y_hat = h[-1]
    return -jnp.mean(y * jnp.log(y_hat) + (1 - y) * jnp.log(1 - y_hat))


#@jit
def forward_path(W_list, x, n_layers):
    h_list = []
    for i in range(n_layers):
        if i == n_layers - 1:
            x = W_list[i] @ x
        else:
            x = jax.nn.relu(W_list[i] @ x)  #jnp.tanh(W_list[i] @ x)  #jax.nn.sigmoid(W_list[i] @ x)
        h_list.append(x)
    return h_list


@jit
def contrastive_update(W, y, y_hat, gamma, learning_rate):
    second_term = gamma * (y @ y.T - y_hat @ y_hat.T) @ W
    return second_term * learning_rate


@jit
def hebbian_update(W, h, h_1, eta):
    eta_sign = (-1 * jnp.sign(-eta) + 1) / 2#jax.nn.sigmoid(-eta*10) + 1) / 2
    eta_pos = eta * h * (h_1.T - h * W)
    eta_neg = eta * h * h_1.T / (1 + jnp.expand_dims(jnp.sum(W**2, axis=1), axis=-1))
    update = eta_sign * eta_pos + (1 - eta_sign) * eta_neg
    return update


if __name__ == "__main__":
    weight_scale = 0.01
    hidden_dim = 20
    hierarchy_depth = 4
    learning_rate = 0.01
    batch_size = 2048
    test_epochs = 100
    epochs = 10
    run_test = True

    all_regimes = {"contrastive_hebb": {"gamma": 1.0, "eta": 0.0},
                   "gradient_descent": {"gamma": 0.0, "eta": 0.0},
                   "quasi_predictive": {"gamma": -1.0, "eta": 0.0},
                   "hebbian": {"gamma": 0, "eta": 1.0},
                   "anti_hebbian": {"gamma": 0, "eta": -1.0},}

    data = SemanticTask(batch_size=batch_size, h_levels=hierarchy_depth)

    W1_0 = np.random.normal(scale=weight_scale, size=(hidden_dim, data.input_dim))
    W2_0 = np.random.normal(scale=weight_scale, size=(hidden_dim, hidden_dim))
    W3_0 = np.random.normal(scale=weight_scale, size=(hidden_dim, hidden_dim))
    W4_0 = np.random.normal(scale=weight_scale, size=(data.output_dim, hidden_dim))
    W_list = [W1_0, W2_0, W3_0, W4_0]
    # net = ContrastiveNet(W1_0=W1_0, W2_0=W2_0, gamma=0.0, eta=0.0, learning_rate=learning_rate)

    if run_test:
        for key, values in all_regimes.items():
            print(key)
            net = MultiLayerContrastiveNet(W_list=W_list,
                                           gamma=values["gamma"],
                                           eta=values["eta"],
                                           learning_rate=learning_rate)
            for i in range(test_epochs):
                x, y = data.full_batch()
                #print(x.shape, y.shape)
                #h = forward_path(net.W_list, x, net.n_layers)
                loss, _ = net.update(x, y)
                print("loss", loss)
            print(key, "works fine :)")
