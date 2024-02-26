import numpy as np
from scipy.stats import norm
import os
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt


def gaussian_func(x, mu, sigma):
    pdf = norm.pdf(x, mu, sigma)
    return pdf/np.max(pdf)


def periodic_kernel(x, mu, sigma=1.0, period=1.0):
    distance = np.stack([np.abs(x - mu), np.abs(x - (mu + period)), np.abs(x - (mu - period))])
    distance = np.min(distance, axis=0)
    kernel = np.exp(-distance**2/(2*sigma**2))
    return kernel/np.max(kernel)


def generate_tuned_weights(input_dim, hidden_dim, angles, tuning_width=10, offset=0):
    W = np.zeros((hidden_dim, input_dim))
    for i in range(hidden_dim):
        mu = 180/hidden_dim*(i+offset) % 180
        W[i, :] = periodic_kernel(angles, mu, sigma=tuning_width, period=180)
    return W, mu, tuning_width


def check_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def infer_periodic_gaussian_params(x, p, use_argmax=True):
    """
    Infer the gaussian parameters from an array representing probability distribution
    :param x: variable space
    :param y: probability
    :return: mean and standard deviation of the inferred gaussian
    """
    if use_argmax:
        mu = x[np.argmax(p)]
    else:
        cos_x, sin_x = jnp.cos(jnp.deg2rad(x)*2), jnp.sin(jnp.deg2rad(x)*2)
        total_vector = jnp.array([jnp.sum(p * cos_x), jnp.sum(p * sin_x)])
        mu = jnp.rad2deg(np.arctan2(total_vector[1], total_vector[0]))/2
    distance = jnp.stack([jnp.abs(x - mu), jnp.abs(x - (mu + 180)), jnp.abs(x - (mu - 180))])
    distance = jnp.min(distance, axis=0)
    sigma = jnp.sqrt(jnp.sum(p * distance**2)/jnp.sum(p))
    return mu, sigma


def probe_tuning_curve(W_list, probing_angles, forward_path_func):
    activity_per_layer, pre_activity = forward_path_func(W_list, probing_angles, len(W_list))
    return activity_per_layer, pre_activity


def plot_config():
    # Configure figure
#     matplotlib.rc('font', **{'family': 'Helvetica'})
    SMALL_SIZE = 18
    MEDIUM_SIZE = 18
    BIGGER_SIZE = 18
    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    plt.rc('xtick', direction='out')
    plt.rc('ytick', direction='out')
    plt.rc('xtick', top=False)
    plt.rc('ytick', right=False)
    plt.rc('axes', labelpad=5)
    plt.rc('axes.spines', right=False)
    plt.rc('axes.spines', top=False)
    plt.rc('patch', facecolor='None')
    plt.rc('axes', facecolor='None')
    plt.rc('axes', linewidth=2)
    plt.rc('ytick.major', size=10)
    plt.rc('xtick.major', width=1.5)
    plt.rc('xtick.major', size=10)
    plt.rc('ytick.major', width=1.5)
