from abc import ABC, abstractmethod
from asyncio.windows_events import NULL

class BaseConfig(ABC):

    def __init__(self, model):
        self._fill_model_from_json(model)
    #Заполнили поля модуля
    @abstractmethod
    def _fill_model_from_json(self, model):
        pass

#Определяем поля
class ConfigData(BaseConfig):

    def __init__(self, model):
        self.__path = NULL
        self.__image_size = NULL
        #self.__load_with_info = NULL
        self.__quant_classes = NULL
        self.__shuffle = NULL
        super().__init__(model)

    @property
    def path(self):
        return self.__path


    @property
    def image_size(self):
        return self.__image_size
    @property
    def load_with_info(self):
        return self.__load_with_info
    @property
    def nrof_classes(self):
        return self.__quant_classes
    @property
    def shuffle(self):
        return self.__shuffle


    def _model_from_json(self, section):

        self.__path = section['path']
        self.__image_size = section['image_size']
        self.__load_with_info = section['load_with_info']
        self.__nrof_classes = section['nrof_classes']
        self.__shuffle = section['shuffle']


class ConfigModel(BaseConfig):

    def __init__(self, section):
        self.__input = []
        self.__up_stack = {}
        self.__acivation_function = NULL
        self.__output = NULL
        super().__init__(section)


    @property
    def up_stack(self):
        return self.__up_stack
    @property
    def acivation_function(self):
        return self.__acivation_function
    @property
    def output(self):
        return self.__output


    def _model_from_json(self, model):
        self.__acivation_function = model['acivation_function']
        self.__output = model['output']
        self.__up_stack = self.__up_stack | model['up_stack']
        for inputs in model['input']:
            self.__input.append(inputs)


class ConfigTrain(BaseConfig):

    def __init__(self, section):
        self.__batch_size = NULL
        self.__nrof_epoch = NULL
        self.__optimizer = {}
        self.__metrics = []
        super().__init__(section)

    @property
    def batch_size(self):
        return self.__batch_size
    @property
    def nrof_epoch(self):
        return self.__nrof_epoch
    @property
    def optimizer(self):
        return self.__optimizer
    @property
    def metrics(self):
        return self.__metrics


    def _model_from_json(self, model):
        self.__batch_size = int(model['batch_size'])
        self.__quant_epoch = model['nrof_epoch']
        self.__optimizer = self.__optimizer | model['optimizer']
        for metrics in model['metrics']:
            self.__metrics.append(metrics)