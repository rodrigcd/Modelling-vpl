import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from vpl_model.tasks import SemanticTask
from . import forward_path
from .forward_paths import sigmoid_output_forward_path, forward_path
from vpl_model.learning import (grad_cross_entropy, feedback_alignment_cross_entropy,
                                predictive_coding_cross_entropy, cross_entropy_loss,
                                grad_mse, feedback_alignment_mse, predictive_coding_mse, mse_loss,
                                hebbian_update)
from vpl_model.learning.regularizers import L2_norm_grad, L1_norm_grad
from vpl_model.utils import probe_tuning_curve


class FlexibleLearningRuleNet(object):

    def __init__(self, W_list, feedback_W_list, eta=0.0, sgd_lr=0.0, L2_reg=0.0, L1_reg=0.0, feedback_alignment_lr=0.0,
                 predictive_coding_lr=0.0, normalize_weights=False, PC_inner_steps=30, PC_inner_lr=0.1,
                 loss_function="cross_entropy", true_sgd_limit=False):
        self.input_dim = W_list[0].shape[1]
        self.output_dim = W_list[-1].shape[0]
        self.eta = eta
        self.n_layers = len(W_list)
        self.sgd_lr = sgd_lr
        self.feedback_alignment_lr = feedback_alignment_lr
        self.predictive_coding_lr = predictive_coding_lr
        self.PC_inner_steps = PC_inner_steps
        self.PC_inner_lr = PC_inner_lr
        self.L2_reg = L2_reg
        self.L1_reg = L1_reg
        self.true_sgd_limit = true_sgd_limit
        self.W_list = [jax.device_put(W) for W in W_list]
        self.feedback_W_list = [jax.device_put(W) for W in feedback_W_list]
        self.loss_func_id = loss_function
        if loss_function == "cross_entropy":
            self.sgd_updates_func = vmap(grad_cross_entropy, in_axes=(0, 0, None, None))
            self.feedback_alignment_func = vmap(feedback_alignment_cross_entropy,
                                                in_axes=(0, 0, None, None, None))
            self.predictive_coding_func = vmap(predictive_coding_cross_entropy,
                                               in_axes=(0, 0, None, None, None, None))
            self.forward_func = sigmoid_output_forward_path
        elif loss_function == "mse":
            self.sgd_updates_func = vmap(grad_mse, in_axes=(0, 0, None, None))
            self.feedback_alignment_func = vmap(feedback_alignment_mse,
                                                in_axes=(0, 0, None, None, None))
            self.predictive_coding_func = vmap(predictive_coding_mse,
                                               in_axes=(0, 0, None, None, None, None))
            self.forward_func = forward_path
        else:
            raise ValueError("loss function not recognized")

        self.normalize_weights = normalize_weights
        if self.normalize_weights:
            self.normalize_activity_weights()

        if not true_sgd_limit:
            self.sgd_lr = self.sgd_lr + 1e-8

    def forward(self, x):
        h_ff, z = self.forward_func(self.W_list, x, self.n_layers)
        return h_ff

    def update(self, x, y):
        h_list, z = self.forward_func(self.W_list, x, self.n_layers)
        y_hat = h_list[-1]
        loss = self.loss_func(y_hat, y)

        if self.eta != 0:
            hebbian_updates = self.get_hebbian_updates(h_list, x)
        else:
            hebbian_updates = [jnp.zeros(self.W_list[_].shape) for _ in range(self.n_layers)]
        if self.sgd_lr != 0:
            sgd_updates = self.sgd_updates_func(x, y, self.W_list, self.n_layers)
        else:
            sgd_updates = [jnp.zeros(self.W_list[_].shape) for _ in range(self.n_layers)]
        if self.feedback_alignment_lr != 0:
            feedback_alignment_updates = self.feedback_alignment_func(x, y, self.W_list, self.feedback_W_list, self.n_layers)
        else:
            feedback_alignment_updates = [jnp.zeros(self.W_list[_].shape) for _ in range(self.n_layers)]
        if self.predictive_coding_lr != 0:
            predictive_coding_updates = self.predictive_coding_func(x, y, self.W_list, self.n_layers,
                                                                    self.PC_inner_steps, self.PC_inner_lr)
        else:
            predictive_coding_updates = [jnp.zeros(self.W_list[_].shape) for _ in range(self.n_layers)]
        if self.L2_reg != 0:
            l2_updates = [L2_norm_grad(W, self.L2_reg) for W in self.W_list]
        else:
            l2_updates = [jnp.zeros(self.W_list[_].shape) for _ in range(self.n_layers)]
        if self.L1_reg != 0:
            l1_updates = [L1_norm_grad(W, self.L1_reg) for W in self.W_list]
        else:
            l1_updates = [jnp.zeros(self.W_list[_].shape) for _ in range(self.n_layers)]

        all_updates = []
        for i in range(self.n_layers):
            if i == self.n_layers - 1:
                update = - jnp.mean(sgd_updates[i], axis=0) * self.sgd_lr \
                         - jnp.mean(feedback_alignment_updates[i], axis=0) * self.feedback_alignment_lr \
                         - jnp.mean(predictive_coding_updates[i], axis=0) * self.predictive_coding_lr \
                         - l2_updates[i] * self.L2_reg - l1_updates[i] * self.L1_reg
                self.W_list[i] = self.W_list[i] + update
                all_updates.append(np.array(update))
            else:
                update = - jnp.mean(sgd_updates[i], axis=0) * self.sgd_lr \
                         - jnp.mean(feedback_alignment_updates[i], axis=0) * self.feedback_alignment_lr \
                         - jnp.mean(predictive_coding_updates[i], axis=0) * self.predictive_coding_lr \
                         - l2_updates[i] * self.L2_reg - l1_updates[i] * self.L1_reg \
                         + jnp.mean(hebbian_updates[i], axis=0)
                self.W_list[i] = self.W_list[i] + update
                all_updates.append(np.array(update))
        return np.array(loss), all_updates

    def loss_func(self, y_hat, y):
        if self.loss_func_id == "cross_entropy":
            return -jnp.mean(y * jnp.log(y_hat) + (1 - y) * jnp.log(1 - y_hat))
        elif self.loss_func_id == "mse":
            return jnp.mean((y_hat - y)**2)/2
        else:
            ValueError("loss function not recognized")

    def get_numpy_weights(self):
        return [np.array(W) for W in self.W_list]

    def get_hebbian_updates(self, h_list, x):
        all_updates = []
        h_1 = x
        for i in range(self.n_layers):
            h = h_list[i]
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
