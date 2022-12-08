import os
from typing import List, Callable
from enum import Enum
import idx2numpy as idx2numpy
import numpy as np
"""здесь пишем класс по считыванию данных, удобнее для каждого набора данных выделять отдельный файл и класс"""

class DataSetType(Enum):
    valid = 10
    train = 20
    test = 30

class Dataset:
    def __init__(self,
                 dataset_type: DataSetType,
                 transforms: List[Callable],
                 quant_classes: int,
                 data_path = "./configs/mnist/",
                 images_name ='train-images.idx3-ubyte',
                 labels_name ='train-labels.idx1-ubyte'):
        self.__data_path = data_path
        self.__dataset_type = dataset_type
        self.__images_name = images_name
        self.__labels_name = labels_name
        self.__labels = []
        self.__images = []
        self.__transforms = transforms
        self.__quant_classes = quant_classes
        self.read_data()



    def read_data(self):
        self.__labels = idx2numpy.convert_from_file(os.path.join(self.__dataset_path, self.__labels_file))
        self.__images = idx2numpy.convert_from_file(os.path.join(self.__dataset_path, self.__images_file))

        self.show_statistics()

    def __len__(self):
        return len(self.__images)

    def one_hot_labels(self, label):
        """
        для 10 классов метка 5-> [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        :param label: метка класса
        :return: one-hot encoding вектор
        """
        result = []
        if(abs(label) > self.__quant_classes):
            raise ValueError('Такого класса нет')

        for i in range(0,self.__quant_classes):
            if(i == label):
                result.append(1)
            else:
                result.append(0)
        return result

    def __getitem__(self, idx:int):
        """
        :param idx: индекс элемента в выборке
        :return: preprocessed image and label
        """
        images = self.__images[idx]
        labels = self.__labels[idx]
        for transform in self.__transforms:
            images = transform(images)
        return images, labels

    def show_statistics(self):
        unique, counts = np.unique(self.__labels, return_counts=True)
        print(f'Датасет: {self.__len__()}'
              f'Классы: {self.__quant_classes}'
              f'Элементы в классе: {dict(zip(unique, counts))}')