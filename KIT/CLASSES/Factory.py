from typing import Optional

import numpy as np

from KIT.CLASSES.Resource import ResourceType
from KIT.CLASSES.Unit import UnitStats, UnitHealth, UnitType, Unit, produce_worker_unit, produce_warrior_unit
import uuid


class Factory:
    def __init__(self, *, player: str, stats: UnitStats, health: UnitHealth, position: np.array, map_size: int):
        self.id = uuid.uuid4()
        self.player = player
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

    def produce_unit(self, unit: UnitType) -> Optional[Unit]:
        resource_cost = 2
        if unit == UnitType.WARROR:
            resource_cost = 6
        if self.health["amount"] > resource_cost:
            self.health["amount"] -= resource_cost
            # Spawn a warrior adjacent to the agent
            for i in range(max(0, self.position[0] - 1), min(self.size, self.position[0] + 2)):
                for j in range(max(0, self.position[1] - 1), min(self.size, self.position[1] + 2)):
                    if unit == UnitType.WARROR:
                        return produce_warrior_unit(self.player, np.array([i, j]))
                    else:
                        return produce_worker_unit(self.player, np.array([i, j]))
        return None

def create_factory(player: str, position: np.array, map_size: int) -> Factory:
    return Factory(
        player=player,
        stats={"hp": 50, "attack": 0, "defence": 50},
        health={"amount": 200, "decay": 2, "type": ResourceType.SPICE, "alive": True},
        position=position,
        map_size=map_size)
