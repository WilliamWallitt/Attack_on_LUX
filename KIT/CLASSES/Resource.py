import random
from enum import Enum

import numpy as np


class ResourceType(Enum):
    EMPTY = 0
    WATER = 1
    SPICE = 2,
    FACTORY = 3


class Resource:

    def __init__(self, *, position: np.ndarray, amount: int = 200, decay_per_turn: int = 5, resource_type: ResourceType = ResourceType.EMPTY):
        self.initial_amount = amount
        self.resource_type = resource_type
        self.amount = amount
        self.decay_per_turn = decay_per_turn
        self.position = position

    def step(self):

        if self.amount - self.decay_per_turn < 0:
            self.resource_type = random.choice(list(ResourceType))
            self.amount = self.initial_amount
        else:
            self.amount -= self.decay_per_turn

        self.amount -= self.decay_per_turn

    def mine(self, amount) -> int:
        if self.amount < amount:
            self.amount = 0
            return self.amount
        else:
            self.amount -= amount
            return amount
