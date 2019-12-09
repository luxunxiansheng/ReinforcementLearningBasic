from tqdm import tqdm

from agent.dp_agent import DP_Agent
from env.grid_world import GridworldEnv
from env.grid_world_with_walls_block import GridWorldWithWallsBlockEnv
from env.room import RoomEnv
from policy.policy import (Greedy_Action_Selector, Random_Action_Selector,
                           Tabular_Implicit_Policy, e_Greedy_Action_Selector)


def main():
   
    """
    room_env = RoomEnv()
    room_agent= DP_Agent(room_env)
    room_agent.q_value_iteration() 
    

    grid_world_env=GridworldEnv([4,4])
    grid_word_agent = DP_Agent(grid_world_env)
    grid_word_agent.q_value_iteration()

    """

    grid_world_with_walls_block_env = GridWorldWithWallsBlockEnv()
    grid_world_with_walls_block_agent = DP_Agent(grid_world_with_walls_block_env)
    grid_world_with_walls_block_agent.q_value_iteration()

    

if __name__ == "__main__":
    main()
