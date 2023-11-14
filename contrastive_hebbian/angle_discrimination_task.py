from semantic_task import BaseTask
from scipy.stats import norm
import numpy as np


class AngleDiscriminationTask(BaseTask):

    def __init__(self, training_orientation=90.0, orientation_diff=45, input_size=30,
                 signal_amp=1.0, signal_bandwidth=20, output_amp=1.0):

        self.training_orientation = training_orientation
        self.orientation_diff = orientation_diff
        self.input_size = input_size
        self.signal_amp = signal_amp
        self.signal_bandwidth = signal_bandwidth
        self.output_amp = output_amp
        self.output_size = 1.0
        self._generate_input()

    def _generate_input(self):
        self.angles = np.linspace(0, 180, self.input_size)
        self.pos_orientation = self.training_orientation + self.orientation_diff/2
        self.neg_orientation = self.training_orientation - self.orientation_diff/2
        self.pos_gaussian = gaussian_func(self.angles, self.pos_orientation, self.signal_bandwidth)
        self.neg_gaussian = gaussian_func(self.angles, self.neg_orientation, self.signal_bandwidth)

    def full_batch(self):
        y = np.array([self.output_amp, -self.output_amp])[..., np.newaxis, np.newaxis]
        x = np.stack([self.pos_gaussian, self.neg_gaussian], axis=0)[..., np.newaxis]
        return x, y


def gaussian_func(x, mu, sigma):
    pdf = norm.pdf(x, mu, sigma)
    return pdf/np.max(pdf)


def periodic_kernel(x, mu, sigma=1.0, period=1.0):
    distance = np.stack([np.abs(x - mu), np.abs(x - (mu + period)), np.abs(x - (mu - period))])
    distance = np.min(distance, axis=0)
    kernel = np.exp(-distance**2/(2*sigma**2))
    return kernel/np.max(kernel)


if __name__ == "__main__":
    task = AngleDiscriminationTask()
    task._generate_input()
    print(task.pos_gaussian)
    print(task.neg_gaussian)
    print(task.angles)
    print(task.pos_orientation)
    print(task.neg_orientation)