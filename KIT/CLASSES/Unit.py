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
    CENTER: 0
    UP: 1
    RIGHT: 2
    DOWN: 3
    LEFT: 4


class UnitStats(TypedDict):
    hp: int
    attack: int
    defence: int


class ProduceUnit(TypedDict):
    type: UnitType
    cost: int


class Unit:
    def __init__(self, player: str, stats: UnitStats, health: UnitHealth, position: np.array,
                 mining_options: List[UnitResource], unit_type: UnitType = UnitType.WORKER):
        self.id = uuid.uuid4()
        self.player = player
        self.stats = stats
        self.unit_type = unit_type
        self.mining_options = mining_options
        self.health = health
        self.position = position

    def step(self):
        if self.health["amount"] - self.health["decay"] < 0 or self.stats["hp"] == 0:
            self.health["amount"] = 0
            self.health["alive"] = False
        else:
            self.health["amount"] -= self.health["decay"]

    def mine(self, resource: Resource):

        unit_resource = next((item for item in self.mining_options if item['type'] == resource.resource_type), None)

        if unit_resource is not None:
            amount_minded = resource.mine(unit_resource["mine_per_turn"])
            if unit_resource["type"] == self.health["type"]:
                self.health["amount"] += amount_minded
            else:
                unit_resource["amount"] += amount_minded

    def move(self, movement: UnitMovement):
        # a[1] = direction (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)
        move_deltas = np.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]])
        target_pos = self.position + move_deltas[movement]
        if target_pos[0] < 0 or target_pos[1] < 0:
            self.position += move_deltas[movement]

    def transfer(self, factory):
        from KIT.CLASSES.Factory import Factory
        if isinstance(factory, Factory):
            if factory.position == self.position:
                unit_resource = next((item for item in self.mining_options if item['type'] == factory.health["type"]), None)
                if unit_resource is not None:
                    factory.health["amount"] += unit_resource["amount"]
                    unit_resource["amount"] = 0

    def attack(self, unit):
        if isinstance(unit, Unit):
            if unit.position == self.position:
                # Calculate damage dealt by the attacker, including a random dice roll
                dice_roll = random.randint(1, 6)
                damage = max(0, unit.stats["attack"] + dice_roll - self.stats["defence"])
                self.health["amount"] = damage
                if self.stats["hp"] >= damage:
                    self.health["alive"] = False
                    self.health["amount"] = 0
                else:
                    self.stats["hp"] -= damage

    # maybe be able to heal factory??


def produce_worker_unit(player: str, position: np.array) -> Unit:
    return Unit(
        player=player,
        stats={"hp": 10, "attack": 2, "defence": 10},
        health={"amount": 100, "decay": 2, "type": ResourceType.WATER, "alive": True},
        position=position,
        mining_options=[
            {"type": ResourceType.SPICE, "amount": 0, "mine_per_turn": 2},
            {"type": ResourceType.WATER, "amount": 0, "mine_per_turn": 6}
        ],
    )


def produce_warrior_unit(player: str, position: np.array) -> Unit:
    return Unit(
        player=player,
        stats={"hp": 50, "attack": 20, "defence": 5},
        health={"amount": 100, "decay": 4, "type": ResourceType.WATER, "alive": True},
        position=position,
        mining_options=[
            {"type": ResourceType.WATER, "amount": 0, "mine_per_turn": 6}
        ],
    )
