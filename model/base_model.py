class BaseModel(object):
    def __init__(self, parameters):
        ...

    def eval(self):
        self.phase = 'eval'

    def train(self):
        self.phase = 'train'
