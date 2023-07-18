import random
import uuid
from enum import Enum

import numpy as np


class ResourceType(Enum):
    EMPTY = 0
    WATER = 1
    SPICE = 2,
    FACTORY = 3


class Resource:

    def __init__(self, *, position: np.ndarray, amount: int = 500, decay_per_turn: int = 1, max_respawn_tick: int = 250, resource_type: ResourceType = ResourceType.EMPTY):
        self.initial_amount = amount
        self.resource_type = resource_type
        self.amount = amount
        self.decay_per_turn = decay_per_turn
        self.max_respawn_tick = max_respawn_tick
        self.position = position
        self.id = uuid.uuid4()
        self.respawn_tick = 0

    def step(self):
        if self.respawn_tick is not None and self.respawn_tick > 0:
            self.respawn_tick -= 1
        elif self.respawn_tick == 0:
            self.resource_type = random.choice([ResourceType.WATER, ResourceType.SPICE])
            self.amount = random.randint(self.initial_amount // 10, self.initial_amount)
            self.respawn_tick = None
        else:
            self.amount -= self.decay_per_turn

            if self.amount <= 0:
                self.resource_type = ResourceType.EMPTY
                self.amount = 0
                self.respawn_tick = random.randint(0, self.max_respawn_tick)

    def mine(self, amount) -> int:
        if self.amount < amount:
            self.amount = 0
            return self.amount
        else:
            self.amount -= amount
            return amount
