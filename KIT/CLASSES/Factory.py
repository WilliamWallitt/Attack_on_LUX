import numpy as np

from KIT.CLASSES.Resource import ResourceType
from KIT.CLASSES.Unit import UnitStats, UnitHealth, UnitType, Unit
from KIT.UTILITIES.utilities import produce_warrior_unit, produce_worker_unit
import uuid


class Factory:
    def __init__(self, *, stats: UnitStats, health: UnitHealth, position: np.ndarray, map_size: int):
        self.id = uuid.uuid4()
        self.stats = stats
        self.health = health
        self.resource_type = ResourceType.FACTORY
        self.position = position
        self.size = map_size

    def step(self):
        if self.health["amount"] - self.health["decay"] < 0 or self.stats["hp"] == 0:
            self.health["amount"] = 0
            self.health["alive"] = False
        else:
            self.health["amount"] -= self.health["decay"]

    def produce_unit(self, unit: UnitType) -> Unit:
        resource_cost = 2
        if unit == UnitType.WARROR:
            resource_cost = 6
        if self.health["amount"] > resource_cost:
            self.health["amount"] -= resource_cost
            # Spawn a warrior adjacent to the agent
            for i in range(max(0, self.position[0] - 1), min(self.size, self.position[0] + 2)):
                for j in range(max(0, self.position[1] - 1), min(self.size, self.position[1] + 2)):
                    if unit == UnitType.WARROR:
                        return produce_warrior_unit(np.ndarray([i, j]))
                    else:
                        return produce_worker_unit(np.ndarray([i, j]))