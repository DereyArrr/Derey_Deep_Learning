import numpy as np
from layers.registry_utils import REGISTRY_TYPE
from layers.layer_base import BaseLayerClass


@REGISTRY_TYPE.register_module
class ReLu(BaseLayerClass):

    def __call__(self, x, phase):
        self.x = x
        return np.clip(x, a_min=0, a_max=None)

    def get_grad(self):
        # np.dot, np.zeros_like/np.ones_like, np.zeros(), np.random...,
        self.grads = np.zeros_like(self.x)  # np.zeros((self.x.shape[0], self.x.shape[1]))
        self.grads[self.x > 0] = 1  # np.array
        return self.grads

    def backward(self, dy):
        self.get_grad()
        return dy * self.grads