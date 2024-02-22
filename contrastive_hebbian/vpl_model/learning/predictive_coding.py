import jax.nn
import jax.numpy as jnp
from jax import grad, vmap, jit
import numpy as np


def predictive_coding_cross_entropy(x, y, W_list, n_layers, inner_steps=60):
    inter_x = [jnp.zeros((W_list[l].shape[-1], 1)) for l in range(n_layers)]
    inter_x[0] = x
    inter_x.append(y)
    for t in range(inner_steps):
        update_x = activity_energy_update(inter_x, W_list, n_layers)
        for i in range(n_layers):
            inter_x[i] = jnp.copy(inter_x[i]) - 0.1 * update_x[i]

    W_updates = weight_energy_update(W_list, inter_x, n_layers)
    return W_updates


def layer_energy_loss(x_l, W_l_1, x_l_1, n_layers, layer_index):
    if layer_index == n_layers - 1:
        return jnp.mean((x_l - jax.nn.sigmoid(W_l_1 @ x_l_1))**2)/2
    else:
        return jnp.mean((x_l - jax.nn.relu(W_l_1 @ x_l_1))**2)/2


def total_energy_loss(x_list, W_list, n_layers):
    total_loss = 0
    for i in range(n_layers):
        total_loss += layer_energy_loss(x_list[i+1], W_list[i], x_list[i], n_layers, i)
    return total_loss


def activity_energy_update(x_list, W_list, n_layers):
    return grad(total_energy_loss, argnums=0)(x_list, W_list, n_layers)


def weight_energy_update(W_list, x_list, n_layers):
    return grad(total_energy_loss, argnums=1)(x_list, W_list, n_layers)


if __name__ == "__main__":
    batch_size = 32
    input_size = 10
    hidden1 = 8
    hidden2 = 6
    hidden3 = 4
    output_size = 1
    n_layers = 4

    W1 = np.random.normal(size=(hidden1, input_size))
    W2 = np.random.normal(size=(hidden2, hidden1))
    W3 = np.random.normal(size=(hidden3, hidden2))
    W4 = np.random.normal(size=(output_size, hidden3))
    W_list = [jnp.array(W1), jnp.array(W2), jnp.array(W3), jnp.array(W4)]

    x = np.random.normal(size=(input_size, 1))
    y = np.random.normal(size=(output_size, 1))

    updates = predictive_coding_cross_entropy(x, y, W_list, n_layers)

    for _ in updates:
        print(_.shape)

    x = np.random.normal(size=(batch_size, input_size, 1))
    y = np.random.normal(size=(batch_size, output_size, 1))

    updates = vmap(predictive_coding_cross_entropy,
                   in_axes=(0, 0, None, None))(x, y, W_list, n_layers)

    for _ in updates:
        print(_.shape)