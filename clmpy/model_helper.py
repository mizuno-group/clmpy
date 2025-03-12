from collections import defaultdict

class LossContainer():
    def __init__(self):
        self.train_loss = defaultdict(list)
        self.valid_loss = defaultdict(list)

    def train_update(self,key,value):
        self.train_loss[key].append(value)

    def valid_update(self,key,value):
        self.valid_loss[key].append(value)