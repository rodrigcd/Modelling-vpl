import jax.numpy as jnp
from jax import jit, vmap
from vpl_model.networks import sigmoid_output_forward_path, forward_path
from vpl_model.learning import grad_mse, grad_cross_entropy
import numpy as np
from functools import partial


@partial(jit, static_argnums=(4,))
def feedback_alignment_cross_entropy(x, y, W_list, backward_W_list, n_layers):
    h, z = sigmoid_output_forward_path(W_list, x, n_layers)
    y_hat = h[-1]
    error = -(y - y_hat)
    update_per_layer = []
    for i in range(n_layers):
        layer_index = n_layers - 1 - i
        if i == 0:
            update = error @ h[layer_index-1].T
            update_per_layer.append(update)
            error = (backward_W_list[layer_index].T @ error) * jnp.heaviside(z[layer_index-1], 0.5)
        elif layer_index == 0:
            update = error @ x.T
            update_per_layer.append(update)
        else:
            update = error @ h[layer_index-1].T
            update_per_layer.append(update)
            error = (backward_W_list[layer_index].T @ error) * jnp.heaviside(z[layer_index-1], 0.5)
    update_per_layer.reverse()
    return update_per_layer

@partial(jit, static_argnums=(4,))
def feedback_alignment_mse(x, y, W_list, backward_W_list, n_layers):
    h, z = forward_path(W_list, x, n_layers)
    y_hat = h[-1]
    error = -(y - y_hat)
    update_per_layer = []
    for i in range(n_layers):
        layer_index = n_layers - 1 - i
        if i == 0:
            update = error @ h[layer_index-1].T
            update_per_layer.append(update)
            error = (backward_W_list[layer_index].T @ error) * jnp.heaviside(z[layer_index-1], 0.5)
        elif layer_index == 0:
            update = error @ x.T
            update_per_layer.append(update)
        else:
            update = error @ h[layer_index-1].T
            update_per_layer.append(update)
            error = (backward_W_list[layer_index].T @ error) * jnp.heaviside(z[layer_index-1], 0.5)
    update_per_layer.reverse()
    return update_per_layer



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

    B1 = np.random.normal(size=(hidden1, input_size))
    B2 = np.random.normal(size=(hidden2, hidden1))
    B3 = np.random.normal(size=(hidden3, hidden2))
    B4 = np.random.normal(size=(output_size, hidden3))
    backward_W_list = [jnp.array(B1), jnp.array(B2), jnp.array(B3), jnp.array(B4)]

    x = np.random.normal(size=(input_size, 1))
    y = np.random.normal(size=(output_size, 1))

    true_grad = grad_mse(x, y, W_list, n_layers)
    explicit_grad = feedback_alignment_mse(x, y, W_list, W_list, n_layers)

    print(np.isclose(true_grad[0], explicit_grad[0]))

    true_grad = grad_cross_entropy(x, y, W_list, n_layers)
    explicit_grad = feedback_alignment_cross_entropy(x, y, W_list, W_list, n_layers)

    print(np.isclose(true_grad[0], explicit_grad[0]))
    # print(true_grad)
    # print(explicit_grad)

    # updates = feedback_alignment_cross_entropy(x, y, W_list, backward_W_list, n_layers)
    #
    # for _ in updates:
    #     print(_.shape)
    #
    # x = np.random.normal(size=(batch_size, input_size, 1))
    # y = np.random.normal(size=(batch_size, output_size, 1))
    #
    # updates = vmap(feedback_alignment_cross_entropy,
    #                in_axes=(0, 0, None, None, None))(x, y, W_list, backward_W_list, n_layers)
    #
    # for _ in updates:
    #     print(_.shape)