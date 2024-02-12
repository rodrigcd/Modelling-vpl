import numpy as np
from scipy.stats import norm
import os


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
    return W

def check_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)