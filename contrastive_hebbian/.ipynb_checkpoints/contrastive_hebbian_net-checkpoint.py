import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from angle_discrimination_task import gaussian_func, periodic_kernel


class ContrastiveNet(object):

    def __init__(self, W1_0, W2_0, gamma, eta, learning_rate, lr_W2_W1):
        self.input_dim = W1_0.shape[1]
        self.hidden_dim = W1_0.shape[0]
        self.output_dim = W2_0.shape[0]
        self.gamma = gamma
        self.eta = eta
        self.learning_rate = learning_rate
        self.W1 = jax.device_put(W1_0)
        self.W2 = jax.device_put(W2_0)
        self.lr_W2_W1 = lr_W2_W1

    def forward(self, x):
        h_ff, y_hat = vmap(forward_path, in_axes=(None, None, 0))(self.W1, self.W2, x)
        return h_ff, y_hat

    def update(self, x, y, y_hat, h_ff):
        batch_dW1 = vmap(d_W1, in_axes=(None, None, 0, 0, 0, 0, None, None, None))(self.W1, self.W2, x, y, h_ff, y_hat, self.gamma, self.eta, self.learning_rate)
        batch_dW2 = vmap(d_W2, in_axes=(None, None, 0, 0, 0, None, None, None))(self.W1, self.W2, x, y, y_hat, self.gamma, self.eta, self.learning_rate)
        grad_W1 = jnp.mean(batch_dW1, axis=0)
        grad_W2 = jnp.mean(batch_dW2, axis=0)
        self.W1 += grad_W1
        self.W2 += grad_W2*self.lr_W2_W1
        return jnp.mean(jnp.abs(grad_W1)), jnp.mean(jnp.abs(grad_W2))

    def loss(self, y_hat, y):
        return jnp.mean((y_hat - y)**2)

@jit
def forward_path(W1, W2, x):
    h_ff = W1 @ x
    y_hat = W2 @ h_ff
    return h_ff, y_hat

@jit
def feedback_path(W1, W2, x, y, gamma):
    h = W1 @ x + gamma * W2.T @ y
    return h

@jit
def contrastive_W1_update(W2, y, y_hat, x, learning_rate):
    return (W2.T @ (y - y_hat) @ x.T) * learning_rate

@jit
def contrastive_W2_update(W1, W2, x, y, y_hat, gamma, learning_rate):
    first_term = (y - y_hat) @ x.T @ W1.T
    second_term = gamma * (y @ y.T - y_hat @ y_hat.T) @ W2
    return (first_term + second_term) * learning_rate

@jit
def hebbian_update(W1, h, x, eta):
    eta_sign = (-1 * jnp.sign(-eta) + 1) / 2
    eta_pos = eta * h * (x.T - h * W1)
    eta_neg = eta * h * x.T / (1 + jnp.expand_dims(jnp.sum(W1**2, axis=1), axis=-1))
    update = eta_sign * eta_pos + (1 - eta_sign) * eta_neg
    return update

@jit
def d_W1(W1, W2, x, y, h_ff, y_hat, gamma, eta, learning_rate):
    W1_hebbian = hebbian_update(W1, h_ff, x, eta)
    W1_contrastive = contrastive_W1_update(W2, y, y_hat, x, learning_rate)
    return W1_hebbian + W1_contrastive

@jit
def d_W2(W1, W2, x, y, y_hat, gamma, eta, learning_rate):
    W2_contrastive = contrastive_W2_update(W1, W2, x, y, y_hat, gamma, learning_rate)
    return W2_contrastive


def generate_tuned_weights(input_dim, hidden_dim, angles, tuning_width=10):
    W = np.zeros((hidden_dim, input_dim))
    for i in range(hidden_dim):
        W[i, :] = periodic_kernel(angles, 180/hidden_dim*i, sigma=tuning_width, period=180)
    return W


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    Nh = 25
    input_dim = 10
    output_dim = 5
    gamma = 0.5
    eta = -1
    W1 = jax.random.normal(subkey, (Nh, input_dim))
    W2 = jax.random.normal(subkey, (output_dim, Nh))
    x = jax.random.normal(subkey, (input_dim, 1))
    y = jax.random.normal(subkey, (output_dim, 1))
    h_ff, y_hat = forward_path(W1, W2, x)
    update = hebbian_update(W1, h_ff, x, eta)