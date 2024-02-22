import jax.numpy as jnp
from jax import grad, vmap
from vpl_model.networks import forward_path, sigmoid_output_forward_path
import numpy as np


def mse_loss(x, y, W_list, n_layers):
    h, z = forward_path(W_list, x, n_layers)
    y_hat = h[-1]
    return jnp.mean((y_hat - y)**2)/2


def cross_entropy_loss(x, y, W_list, n_layers):
    h, z = sigmoid_output_forward_path(W_list, x, n_layers)
    y_hat = h[-1]
    return -jnp.mean(y * jnp.log(y_hat) + (1 - y) * jnp.log(1 - y_hat))


def grad_cross_entropy(x, y, W_list, n_layers):
    return grad(cross_entropy_loss, argnums=2)(x, y, W_list, n_layers)


def grad_mse(x, y, W_list, n_layers):
    return grad(mse_loss, argnums=2)(x, y, W_list, n_layers)


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

    updates = grad_cross_entropy(x, y, W_list, n_layers)

    for _ in updates:
        print(_.shape)

    x = np.random.normal(size=(batch_size, input_size, 1))
    y = np.random.normal(size=(batch_size, output_size, 1))

    updates = vmap(grad_cross_entropy,
                   in_axes=(0, 0, None, None))(x, y, W_list, n_layers)

    for _ in updates:
        print(_.shape)