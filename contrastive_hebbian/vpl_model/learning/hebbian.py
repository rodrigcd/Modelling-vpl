import jax.numpy as jnp
from jax import jit


@jit
def hebbian_update(W, h, h_1, eta):
    eta_sign = (-1 * jnp.sign(-eta) + 1) / 2
    eta_pos = eta * h * (h_1.T - h * W)
    eta_neg = eta * h * h_1.T / (1 + jnp.expand_dims(jnp.sum(W**2, axis=1), axis=-1))
    update = eta_sign * eta_pos + (1 - eta_sign) * eta_neg
    return update


@jit
def contrastive_update(W, y, y_hat, gamma, learning_rate):
    second_term = gamma * (y @ y.T - y_hat @ y_hat.T) @ W
    return second_term * learning_rate