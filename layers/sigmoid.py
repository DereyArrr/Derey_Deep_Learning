import numpy as np

from layers.registry_utils import REGISTRY_TYPE
from layers.layer_base import BaseLayerClass


@REGISTRY_TYPE.register_module
class Sigmoid(BaseLayerClass):

    def __call__(self, x, phase):
        self.x = x
        out = np.exp(-x)
        return 1 / (1 + out)

    def get_grad(self):
        out = np.exp(- self.x)
        self.grads = out / (1 + out)^2
        return self.grads

    def backward(self, dy):
        return dy * self.grads