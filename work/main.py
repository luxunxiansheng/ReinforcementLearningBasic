from tqdm import tqdm

from env.grid_world_with_walls_block import GridWorldWithWallsBlockEnv
from env.grid_world import GridworldEnv
from policy.policy import Tabular_Policy
from algorithm.dynamic_programming.q_value_iteration_method import Q_Value_Iteration_Method
from algorithm.dynamic_programming.v_value_iteration_method import V_Value_Iteration_Method
from algorithm.dynamic_programming.policy_iteration_method  import Policy_Iteration_Method
from lib.utility import create_distribution_randomly


def main():
   
    env = GridWorldWithWallsBlockEnv()
    
    test_policy_iteration(env)


def test_q_value_iteration(env):
    q_table = env.build_Q_table()
    transition_table= env.P
    policy_table = env.build_policy_table()

    rl_method = Q_Value_Iteration_Method(q_table,transition_table)
   
    table_policy= Tabular_Policy(policy_table,rl_method)
    table_policy.improve()
    
    table_policy.show_policy()

def test_v_value_iteration(env):
    v_table = env.build_V_table()
    transition_table= env.P
    policy_table = env.build_policy_table()

    rl_method = Q_Value_Iteration_Method(v_table,transition_table)
   
    table_policy= Tabular_Policy(policy_table,rl_method)
    table_policy.improve()
    
    table_policy.show_policy()

def test_policy_iteration(env):
    v_table = env.build_V_table()
    transition_table = env.P 
    policy_table = env.build_policy_table()

    for state_index, action_probablities in  policy_table.items():
        distribution = create_distribution_randomly()(action_probablities.values())
        for action_index,_ in action_probablities.items():
            policy_table[state_index][action_index]=distribution[action_index]
        
    rl_method = Policy_Iteration_Method(v_table,transition_table)
    table_policy= Tabular_Policy(policy_table,rl_method)

    delta = 1e-3
    while True:
        table_policy.evaluate()
        table_policy.show_policy()

        current_delta= table_policy.improve()
        table_policy.show_policy()
        if current_delta<delta:
            break
    
    
    

if __name__ == "__main__":
    main()


