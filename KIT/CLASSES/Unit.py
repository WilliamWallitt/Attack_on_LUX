from typing import TypedDict, List
from enum import Enum
import uuid
import numpy as np
import random
from KIT.CLASSES.Resource import ResourceType, Resource


class UnitType(Enum):
    WORKER = 1,
    WARROR = 2


class UnitHealth(TypedDict):
    type: ResourceType
    amount: int
    alive: bool
    decay: int


class UnitResource(TypedDict):
    type: ResourceType
    amount: int
    mine_per_turn: int


class UnitMovement(Enum):
    CENTER = 0
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4


class UnitStats(TypedDict):
    attack: int
    defence: int


class ProduceUnit(TypedDict):
    type: UnitType
    cost: int


class Unit:
    def __init__(self, player: str, stats: UnitStats, health: UnitHealth, position: np.array,
                 mining_options: List[UnitResource], size: int, unit_type: UnitType = UnitType.WORKER):
        self.id = uuid.uuid4().hex
        self.player = player
        self.stats = stats
        self.unit_type = unit_type
        self.mining_options = mining_options
        self.health = health
        self.position = position
        self.size = size

    def step(self):
        self.health["amount"] -= self.health["decay"]

        if self.health["amount"] < 0:
            self.health["amount"] = 0
            self.health["alive"] = False

    def mine(self, resource: Resource):
        for item in self.mining_options:
            if item['type'] == resource.resource_type and item is not None:
                amount_minded = resource.mine(item["mine_per_turn"])
                if item["type"] == self.health["type"]:
                    self.health["amount"] += amount_minded
                    item["amount"] += amount_minded
                else:
                    item["amount"] += amount_minded

    def move(self, movement: int):
        # a[1] = direction (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)
        move_deltas = np.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]])
        target_pos = self.position + move_deltas[movement]
        if (target_pos[0] >= 0 and target_pos[0] < self.size) and (target_pos[1] >= 0 and target_pos[1] < self.size):
            self.position += move_deltas[movement]

    def transfer(self, factory):
        from KIT.CLASSES.Factory import Factory
        if isinstance(factory, Factory):
            if np.array_equal(factory.position, self.position):
                for item in self.mining_options:
                    if item['type'] == factory.health["type"] and item is not None:
                        factory.health["amount"] += item["amount"]
                        item["amount"] = 0

    def attack(self, unit):
        if isinstance(unit, Unit):
            if np.array_equal(unit.position, self.position):
                # Calculate damage dealt by the attacker, including a random dice roll
                dice_roll = random.randint(1, 6)
                damage = max(0, unit.stats["attack"] + dice_roll - self.stats["defence"])
                self.health["amount"] -= damage

                if self.health["amount"] < 0:
                    self.health["alive"] = False
                    self.health["amount"] = 0

    # maybe be able to heal factory??


def produce_worker_unit(player: str, position: np.array, size) -> Unit:
    return Unit(
        player=player,
        stats={"attack": 2, "defence": 10},
        health={"amount": 100, "decay": 1, "type": ResourceType.WATER, "alive": True},
        position=position,
        mining_options=[
            {"type": ResourceType.SPICE, "amount": 0, "mine_per_turn": 4},
            {"type": ResourceType.WATER, "amount": 0, "mine_per_turn": 12}
        ],
        size=size
    )


def produce_warrior_unit(player: str, position: np.array, size) -> Unit:
    return Unit(
        player=player,
        stats={"attack": 6, "defence": 5},
        health={"amount": 100, "decay": 1, "type": ResourceType.WATER, "alive": True},
        position=position,
        mining_options=[
            {"type": ResourceType.WATER, "amount": 0, "mine_per_turn": 12}
        ],
        size=size
    )
