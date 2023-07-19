import numpy as np
from gym.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete

# how many more turns a given factory can survive...
# num turns
# phase (IMPORTANT, disc[0, 4] turns / cycle len)


class AttackOnLuxObsSpec:

    @staticmethod
    def get_obs_spec(num_agents: int, size: int) -> Dict:
        return Dict({
            "worker": MultiBinary((1, size, size)),
            "worker_cargo_spice": Box(0., float("inf"), shape=(1, size, size)),
            "worker_cargo_water": Box(0., float("inf"), shape=(1, size, size)),
            "worker_health": Box(0., float("inf"), shape=(1, size, size)),
            "enemy_worker": MultiBinary((1, size, size)),
            "selectable_worker": Box(0., float("inf"), shape=(1, size, size)),

            "warrior": MultiBinary((1, size, size)),
            "warrior_cargo_water": Box(0., float("inf"), shape=(1, size, size)),
            "warrior_health": Box(0., float("inf"), shape=(1, size, size)),
            "enemy_warrior": MultiBinary((1, size, size)),
            "selectable_warrior": Box(0., float("inf"), shape=(1, size, size)),

            "factory": MultiBinary((1, size, size)),
            "factory_cargo_spice": Box(0., float("inf"), shape=(1, size, size)),
            "factory_health": Box(0., float("inf"), shape=(1, size, size)),

            "water": Box(0., float("inf"), shape=(1, size, size)),
            "spice": Box(0., float("inf"), shape=(1, size, size)),
            "empty": Box(0., float("inf"), shape=(1, size, size)),
        })