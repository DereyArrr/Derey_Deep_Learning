import math

import numpy as np

from layers.registry_utils import REGISTRY_TYPE
from layers.layer_base import BaseLayerClass


@REGISTRY_TYPE.register_module
class FullyConnected(BaseLayerClass):
    def __init__(self, input_size, output_size):   # гиперпараметры слоя
        """
        инициализация обучаемых параметров нейронной сети
        """
        super().__init__()
        self.bias = np.zeros(output_size)
        self.weight = np.random.normal(0, math.sqrt(2/output_size), size = (input_size, output_size))   # иницализируем матрицу


    def __call__(self, x, phase):  #фаза обучения или валидации
        self.x = x   # батч
        return np.dot(self.x, self.weight) + self.bias

    @property
    def trainable(self):
        """
        является ли слой обучаемым
        """
        return True

    def get_grad(self):   # dg_df
        df_dx = self.weight
        df_dw = self.x
        df_db = 1
        self.grads = [df_dx, df_dw, df_db]
        return self.grads

    def backward(self, dy):   # градиенты по текущему слою
        self.get_grad()
        self.x_grad = np.dot(self.grads[0], dy.T)
        self.weight_grad = (np.dot(self.grads[1].T, dy))/len(dy)
        self.bias_grad = np.mean(self.grads[2] * dy, 0)
        return self.x_grad.T

    def update_weights(self, update_func):
        """
        обновление обучаемых параметров
        """
        self.weight = self.weight - update_func(self.weight_grad)
        self.bias = self.bias - update_func(self.bias_grad)