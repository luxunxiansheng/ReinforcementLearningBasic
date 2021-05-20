from env.blackjack import BlackjackEnv
from env.cliff_walking import CliffWalkingEnv
from env.grid_world import GridworldEnv
from env.grid_world_with_walls_block import GridWorldWithWallsBlockEnv
from env.mountain_car import MountainCarEnv
from env.random_walking import RandomWalkingEnv
from env.windy_gridworld import WindyGridworldEnv

# A simple factory method
def get_env(env):
    if env == GridworldEnv.__name__:
        return GridworldEnv()

    if env == BlackjackEnv.__name__:
        return BlackjackEnv()

    if env == RandomWalkingEnv.__name__:
        return RandomWalkingEnv()

    if env == WindyGridworldEnv.__name__:
        return WindyGridworldEnv()

    if env == CliffWalkingEnv.__name__:
        return CliffWalkingEnv()

    if env == MountainCarEnv.__name__:
        return MountainCarEnv()