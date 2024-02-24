import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from vpl_model.tasks import SemanticTask
from .forward_paths import sigmoid_output_forward_path
from vpl_model.learning import (grad_cross_entropy, feedback_alignment_cross_entropy,
                                predictive_coding_cross_entropy, hebbian_update, cross_entropy_loss)
from vpl_model.utils import probe_tuning_curve


class FlexibleLearningRuleNet(object):

    def __init__(self, W_list, feedback_W_list, eta=0.0, sgd_lr=0.0, feedback_alignment_lr=0.0,
                 predictive_coding_lr=0.0, normalize_weights=False):

        self.input_dim = W_list[0].shape[1]
        self.output_dim = W_list[-1].shape[0]
        self.eta = eta
        self.n_layers = len(W_list)
        self.sgd_lr = sgd_lr
        self.feedback_alignment_lr = feedback_alignment_lr
        self.predictive_coding_lr = predictive_coding_lr
        self.W_list = [jax.device_put(W) for W in W_list]
        self.feedback_W_list = [jax.device_put(W) for W in feedback_W_list]
        self.sgd_updates_func = vmap(grad_cross_entropy, in_axes=(0, 0, None, None))
        self.feedback_alignment_func = vmap(feedback_alignment_cross_entropy,
                                            in_axes=(0, 0, None, None, None))
        self.predictive_coding_func = vmap(predictive_coding_cross_entropy,
                                           in_axes=(0, 0, None, None))
        # self.forward = sigmoid_output_forward_path
        self.weight_reg = 1.0
        self.normalize_weights = normalize_weights
        if self.normalize_weights:
            self.normalize_activity_weights()

    def forward(self, x):
        h_ff, z = sigmoid_output_forward_path(self.W_list, x, self.n_layers)
        return h_ff

    def update(self, x, y):
        h_list, z = sigmoid_output_forward_path(self.W_list, x, self.n_layers)
        y_hat = h_list[-1]
        loss = self.loss_func(y_hat, y)

        hebbian_updates = self.get_hebbian_updates(h_list, x)
        if self.sgd_lr != 0:
            sgd_updates = self.sgd_updates_func(x, y, self.W_list, self.n_layers)
        else:
            sgd_updates = [jnp.zeros(self.W_list[_].shape) for _ in range(self.n_layers)]
        if self.feedback_alignment_lr != 0:
            feedback_alignment_updates = self.feedback_alignment_func(x, y, self.W_list, self.feedback_W_list, self.n_layers)
        else:
            feedback_alignment_updates = [jnp.zeros(self.W_list[_].shape) for _ in range(self.n_layers)]
        if self.predictive_coding_lr != 0:
            predictive_coding_updates = self.predictive_coding_func(x, y, self.W_list, self.n_layers)
        else:
            predictive_coding_updates = [jnp.zeros(self.W_list[_].shape) for _ in range(self.n_layers)]

        all_updates = []
        for i in range(self.n_layers):
            if i == self.n_layers - 1:
                update = - jnp.mean(sgd_updates[i], axis=0) * self.sgd_lr \
                         - jnp.mean(feedback_alignment_updates[i], axis=0) * self.feedback_alignment_lr \
                         - jnp.mean(predictive_coding_updates[i], axis=0) * self.predictive_coding_lr
                self.W_list[i] = self.W_list[i] + update
                all_updates.append(np.array(update))
            else:
                update = - jnp.mean(sgd_updates[i], axis=0) * self.sgd_lr \
                         - jnp.mean(feedback_alignment_updates[i], axis=0) * self.feedback_alignment_lr \
                         - jnp.mean(predictive_coding_updates[i], axis=0) * self.predictive_coding_lr + \
                         + jnp.mean(hebbian_updates[i], axis=0)
                self.W_list[i] = self.W_list[i] + update
                all_updates.append(np.array(update))
        return np.array(loss), all_updates

    def loss_func(self, y_hat, y):
        return -jnp.mean(y * jnp.log(y_hat) + (1 - y) * jnp.log(1 - y_hat))

    def mse_loss(self, y_hat, y):
        return jnp.mean((y_hat - y)**2)/2

    def get_numpy_weights(self):
        return [np.array(W) for W in self.W_list]

    def get_hebbian_updates(self, h_list, x):
        all_updates = []
        h_1 = x
        for i in range(self.n_layers):
            h = h_list[0]
            update = vmap(hebbian_update, in_axes=(None, 0, 0, None))(self.W_list[i], h_list[i], h_1, self.eta)
            all_updates.append(update)
            h_1 = h
        return all_updates

    def get_tuning_curves(self):
        probing_angles = jnp.eye(self.input_dim)
        activity_per_layer, z = probe_tuning_curve(self.W_list, probing_angles, sigmoid_output_forward_path)
        return activity_per_layer

    def normalize_activity_weights(self):
        for i, W in enumerate(self.W_list):
            activity_per_layer = self.get_tuning_curves()
            normalized_weights = W/np.amax(activity_per_layer[i], axis=1)[:, np.newaxis]
            self.W_list[i] = normalized_weights
