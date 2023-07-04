import numpy as np

from KIT.CLASSES.Factory import Factory
from KIT.CLASSES.Resource import ResourceType
from KIT.CLASSES.Unit import Unit, UnitResource


def produce_worker_unit(position: np.ndarray) -> Unit:
    return Unit(stats={"hp": 10, "attack": 2, "defence": 10},
                health={"amount": 100, "decay": 2, "type": ResourceType.WATER, "alive": True},
                position=position,
                mining_options=[
                    {"type": ResourceType.SPICE, "amount": 0, "mine_per_turn": 2},
                    {"type": ResourceType.WATER, "amount": 0, "mine_per_turn": 6}
                ],
            )


def produce_warrior_unit(position: np.ndarray) -> Unit:
    return Unit(stats={"hp": 50, "attack": 20, "defence": 5},
                health={"amount": 100, "decay": 4, "type": ResourceType.WATER, "alive": True},
                position=position,
                mining_options=[
                    {"type": ResourceType.WATER, "amount": 0, "mine_per_turn": 6}
                ],
            )


def create_factory(position: np.ndarray, map_size: int) -> Factory:
    return Factory(stats={"hp": 50, "attack": 0, "defence": 50},
                   health={"amount": 200, "decay": 2, "type": ResourceType.SPICE, "alive": True},
                   position=position,
                   map_size=map_size)