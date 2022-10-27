# -*- coding: utf-8 -*-
"""Config class"""

import json

from configs.config import CFG
from utils.config.config_d_m_t import ConfigData, ConfigTrain, ConfigModel


class Config:
    """Config class which contains data, train and model hyperparameters"""

    def __init__(self):
        self.from_json()

    @classmethod
    def from_json(self):
        self.data = ConfigData(CFG['data'])
        self.train = ConfigTrain(CFG['train'])
        self.model = ConfigModel(CFG['model'])



