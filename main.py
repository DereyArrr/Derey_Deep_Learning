# -*- coding: utf-8 -*-
""" main.py """
from distutils.command.config import config

from configs.config import CFG
from dataloader.dataloader import DataLoader
from datasets.dataset import Dataset, DataSetType
from model.ExampleModel import ExampleModel
from ops.norm import Normalize, View
from utils.config import Config





def run():
    conf = Config()

    dataset = Dataset(dataset_type = DataSetType.train,
                                 transforms = [Normalize(),View()],
                                 quant_classes = conf.data.quant_classes)

    dataload = DataLoader(dataset,
                          conf.train.batch_size,
                          None,
                          conf.train.nrof_epoch)
    dataset.read_data()

    next(dataload.batch_generator())
    dataload.show_batch()

if __name__ == '__main__':
    run()
