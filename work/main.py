from tqdm import tqdm

from agent.dp_agent import DP_Agent
from agent.gamblers_problem_agent import Gamblers_Problem_Agent
from agent.grid_world_agent import Grid_World_Agent
from env.room import RoomEnv
from policy.policy import (Greedy_Action_Selector, Random_Action_Selector,
                           Tabular_Implicit_Policy, e_Greedy_Action_Selector)


def main():
    #gamblers_problem_agent = Gamblers_Problem_Agent()
    #gamblers_problem_agent.value_iteration()
    
    room_env = RoomEnv()
    room_agent= DP_Agent(room_env)
    room_agent.value_iteration()

    

if __name__ == "__main__":
    main()
