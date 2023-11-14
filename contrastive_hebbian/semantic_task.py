import numpy as np
import matplotlib.pyplot as plt


class BaseTask(object):

    def sample_batch(self):
        pass

    def get_correlation_matrix(self, training=True):
        return [[] for i in range(5)]

    def get_linear_regression_solution(self, reg_coef=0.0):
        correlation_matrices = self.get_correlation_matrix()
        input_corr = correlation_matrices[0]
        input_output_corr = correlation_matrices[2]
        linear_mapping = input_output_corr.T @ np.linalg.inv(input_corr)
        return linear_mapping

    def get_best_possible_error(self, training=True):
        input_corr, output_corr, input_output_corr, expected_y, expected_x = self.get_correlation_matrix(training=training)
        min_loss = (np.trace(output_corr) - np.trace(input_output_corr.T @ np.linalg.inv(input_corr) @ input_output_corr))/2.0
        return min_loss

    def set_random_seed(self, seed):
        np.random.seed(seed=seed)


class SemanticTask(BaseTask):

    def __init__(self, batch_size, h_levels=3, affine_data=True):
        self.batch_size = batch_size
        self.h_level = h_levels
        self.affine_data = affine_data
        self._generate_hierarchy()

    def _generate_hierarchy(self):
        self.input_dim = 2**(self.h_level-1)
        self.output_dim = 2**self.h_level - 1

        self.input_matrix = np.identity(n=self.input_dim)
        self.input_corr = np.identity(n=self.input_dim)/self.input_dim
        self.h_matrix = generate_hierarchy_matrix(self.h_level).T
        self.input_output_corr = self.h_matrix/self.input_dim
        self.output_corr = self.input_output_corr.T @ self.input_output_corr * self.input_dim

    def get_correlation_matrix(self, training=None):
        expected_x = np.mean(self.input_matrix, axis=0)
        expected_y = np.mean(self.h_matrix, axis=0)
        return self.input_corr, self.output_corr, self.input_output_corr, expected_y, expected_x

    def sample_batch(self, training=None):
        batch_idx = np.random.choice(np.arange(self.input_dim), size=self.batch_size, replace=True)
        x = self.input_matrix[batch_idx, :]
        y = self.h_matrix[batch_idx, :]
        return x, y

    def full_batch(self):
        return self.input_matrix[..., np.newaxis], self.h_matrix[..., np.newaxis]


def generate_hierarchy_matrix(h):
    h = h - 1
    if h < 0:
        raise Exception("Hierarchy < 0")
    else:
        cov_matrix = np.ones((1, 1))
        for i in range(h):
            new_matrix = []
            for j in range(cov_matrix.shape[1]):
                new_matrix.append(cov_matrix[:, j])
                new_matrix.append(cov_matrix[:, j])
            new_matrix = np.stack(new_matrix, axis=1)
            new_matrix = np.concatenate([new_matrix, np.identity(new_matrix.shape[1])])
            cov_matrix = new_matrix
    return cov_matrix
