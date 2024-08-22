import numpy as np
import jax.numpy as jnp
from vpl_model.networks import sigmoid_output_forward_path, forward_path

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

    h_list, z_list = sigmoid_output_forward_path(W_list, x, n_layers)
    print("debug")