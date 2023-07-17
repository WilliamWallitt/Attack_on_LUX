import random
from typing import Optional, Union, List, Tuple, TypedDict
import numpy as np
from KIT.CLASSES.Resource import Resource, ResourceType
from KIT.CLASSES.Factory import Factory
from KIT.CLASSES.Unit import Unit, produce_warrior_unit
from KIT.CLASSES.Factory import create_factory
import noise


class Map:

    def __init__(self, size: int, num_agents: int):
        self.size = size
        self.num_agents = num_agents
        self.factories: List[Factory] = []
        self.agents: List[Unit] = []
        self.resources: List[Resource] = []
        self.create_non_random_resource_distribution()
        self.place_factories_and_agents_poisson_disk_sampling()
        self.current_agent = 0

    def create_non_random_resource_distribution(self, scale=100.0, threshold=-0.1):

        for i in range(self.size):
            for j in range(self.size):
                x = i / scale
                y = j / scale
                value = noise.snoise2(x, y, octaves=5, persistence=0.5, lacunarity=2.0, repeatx=self.size,
                                      repeaty=self.size)
                if value > threshold:
                    print(f"Placing SPICE at position {i, j}")
                    self.resources.append(Resource(position=np.array([i, j]), resource_type=ResourceType.SPICE))
                else:
                    print(f"Place WATER at position {i, j}")
                    self.resources.append(Resource(position=np.array([i, j]), resource_type=ResourceType.WATER))

    def place_factories_and_agents(self):
        factories_placed = 0
        min_agent_distance = 5
        while factories_placed < self.num_agents:
            x = np.random.randint(self.size)
            y = np.random.randint(self.size)
            # Check the distance to other agents
            too_close = False
            for i in range(max(0, x - min_agent_distance), min(self.size, x + min_agent_distance + 1)):
                for j in range(max(0, y - min_agent_distance), min(self.size, y + min_agent_distance + 1)):
                    if len([x for x in self.factories if x.position == np.array([i, j])]) != 0:
                        too_close = True
                        break
                if too_close:
                    break
            if not too_close:
                self.factories.append(create_factory(str(factories_placed), np.array([x, y]), self.size))
                # Spawn a worker adjacent to the agent
                placed = False
                for i in range(max(0, x - 1), min(self.size, x + 2)):
                    for j in range(max(0, y - 1), min(self.size, y + 2)):
                        if len([x for x in self.agents if x.position == np.array([x, y])]) != 0 and not placed:
                            self.agents.append(produce_warrior_unit(str(factories_placed), np.array([x, y])))
                            placed = True
            factories_placed += 1

    def place_factories_and_agents_poisson_disk_sampling(self):
        def get_random_point():
            return random.randint(0, self.size - 1), random.randint(0, self.size - 1)

        def calc_distance(p1, p2):
            return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

        def is_valid_point(p, factories):
            for factory in factories:
                if calc_distance(p, factory.position) < min_distance:
                    return False
            return True

        min_distance = int(self.size / 4)

        # Randomly select initial city locations
        for player in range(self.num_agents):
            x, y = get_random_point()
            self.factories.append(create_factory(str(player), np.array([x, y]), self.size))

        active_list = list(self.factories)
        current_player = 0
        while active_list and current_player < self.num_agents:
            city_idx = random.randrange(len(active_list))
            current_city = active_list[city_idx]
            placed = False

            for _ in range(30):  # Maximum number of attempts to find a valid point
                angle = random.uniform(0, 2 * np.pi)
                distance = random.uniform(min_distance, 2 * min_distance)
                x = int(current_city.position[0] + distance * np.cos(angle))
                y = int(current_city.position[1] + distance * np.sin(angle))

                x = max(0, min(self.size - 1, x))
                y = max(0, min(self.size - 1, y))

                new_point = np.array([x, y])

                if is_valid_point(new_point, self.factories):
                    self.factories = list(filter(lambda x: x.player != current_player, self.factories))
                    self.factories.append(create_factory(str(current_player), np.array([x, y]), self.size))
                    active_list.append(self.factories[-1])
                    # Spawn a worker adjacent to the agent
                    placed = False
                    for i in range(max(0, x - 1), min(self.size, x + 2)):
                        for j in range(max(0, y - 1), min(self.size, y + 2)):
                            if not placed and i != x and j != y:
                                self.agents.append(produce_warrior_unit(str(current_player), np.array([i, j])))
                                placed = True
                    placed = True
                    current_player += 1
                    break

            if not placed:
                active_list.pop(city_idx)
