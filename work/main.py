from tqdm import tqdm

from env.grid_world_with_walls_block import GridWorldWithWallsBlockEnv
from env.grid_world import GridworldEnv
from policy.policy import Tabular_Policy
from algorithm.dynamic_programming.q_value_iteration_method import Q_Value_Iteration_Method
from algorithm.dynamic_programming.v_value_iteration_method import V_Value_Iteration_Method


def main():
   
    env = GridWorldWithWallsBlockEnv()
    
    q_table = env.build_Q_table()
    v_table = env.build_V_table()
    transition_table= env.P
    policy_table = env.build_policy_table()

    rl_method = V_Value_Iteration_Method(v_table,transition_table)
   
    table_policy= Tabular_Policy(policy_table,rl_method)
    table_policy.improve()
    
    table_policy.show_policy()
   

if __name__ == "__main__":
    main()


