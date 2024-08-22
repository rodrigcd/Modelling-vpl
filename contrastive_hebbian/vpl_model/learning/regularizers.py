import jax.numpy as jnp
from jax import grad, vmap, jit

@jit
def L2_norm(W):
    return jnp.mean(W**2)/2

@jit
def L1_norm(W):
    return jnp.mean(jnp.abs(W))

@jit
def L2_norm_grad(W):
    return grad(L2_norm)(W)

@jit
def L1_norm_grad(W):
    return grad(L1_norm)(W)