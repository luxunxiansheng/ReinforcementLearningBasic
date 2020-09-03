from env.blackjack import BlackjackEnv
from env.cliff_walking import CliffWalkingEnv
from env.grid_world import GridworldEnv
from env.grid_world_with_walls_block import GridWorldWithWallsBlockEnv
from env.mountain_car import MountainCarEnv
from env.random_walking import RandomWalkingEnv
from env.windy_gridworld import WindyGridworldEnv


def get_env(env):
    if env == 'grid_world':
        return GridworldEnv()

    if env == 'blackjack':
        return BlackjackEnv()

    if env == 'randomwalking':
        return RandomWalkingEnv()

    if env == 'windygridworld':
        return WindyGridworldEnv()

    if env == 'cliffwalking':
        return CliffWalkingEnv()

    if env == 'mountaincar':
        return MountainCarEnv()