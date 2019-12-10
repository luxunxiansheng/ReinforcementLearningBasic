from tqdm import tqdm


from env.grid_world_with_walls_block import GridWorldWithWallsBlockEnv
from policy.policy import Tabular_Policy
from algorithm.dynamic_programming.q_value_iteration_method import Q_Value_Iteration_Method


def main():
   
    grid_world_with_walls_block_env = GridWorldWithWallsBlockEnv()
    
    q_table = grid_world_with_walls_block_env.build_Q_table()
    transition_table= grid_world_with_walls_block_env.build_transitions()
    policy_table = grid_world_with_walls_block_env.build_policy_table()

    rl_method = Q_Value_Iteration_Method(q_table,transition_table)
    rl_method.improve()
    
    


    table_policy= Tabular_Policy(policy_table,rl_method)
    table_policy.improve()
    


    

if __name__ == "__main__":
    main()
