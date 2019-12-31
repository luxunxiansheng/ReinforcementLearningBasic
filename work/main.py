from algorithm.dynamic_programming.policy_iteration_method import Policy_Iteration_Method
from algorithm.dynamic_programming.q_value_iteration_method import Q_Value_Iteration_Method
from algorithm.dynamic_programming.v_value_iteration_method import V_Value_Iteration_Method
from algorithm.monte_carlo_method.v_monte_carlo_method import V_Monte_Carlo_Method
from algorithm.monte_carlo_method.q_monte_carlo_method import Q_Monte_Carlo_Method

from env.grid_world import GridworldEnv
from env.grid_world_with_walls_block import GridWorldWithWallsBlockEnv
from env.gamblers_problem import GamblersProblemEnv
from env.blackjack  import BlackjackEnv

from lib.utility import create_distribution_randomly
from policy.policy import Tabular_Policy


def main():
    env = BlackjackEnv()
    

    test_q_mc_methond(env)
    #test_v_mc_methond(env)

    #test_q_value_iteration(env)
    #test_v_value_iteration(env)
    #test_policy_iteration(env)



def test_q_mc_methond(env):
    q_table = env.build_Q_table()
    
    policy_table = env.build_policy_table() 

    for state_index, _ in policy_table.items():
        card_sum = state_index[0] 
        if card_sum < 20:
            policy_table[state_index][BlackjackEnv.HIT]= 1.0 
            policy_table[state_index][BlackjackEnv.STICK] = 0.0 
        else:
            policy_table[state_index][BlackjackEnv.HIT]= 0.0 
            policy_table[state_index][BlackjackEnv.STICK] = 1.0
            
    rl_method =  Q_Monte_Carlo_Method(q_table,env)
    table_policy = Tabular_Policy(policy_table,rl_method)

    for _ in range(100):
        
        table_policy.evaluate()
        table_policy.improve()

    env.show_policy(table_policy)
    

def test_v_mc_methond(env):
    v_table = env.build_V_table()
    
    policy_table = env.build_policy_table() 

    for state_index, _ in policy_table.items():
        card_sum = state_index[0] 
        if card_sum < 20:
            policy_table[state_index][BlackjackEnv.HIT]= 1.0 
            policy_table[state_index][BlackjackEnv.STICK] = 0.0 
        else:
            policy_table[state_index][BlackjackEnv.HIT]= 0.0 
            policy_table[state_index][BlackjackEnv.STICK] = 1.0
            
    rl_method =  V_Monte_Carlo_Method(v_table,env)
    table_policy = Tabular_Policy(policy_table,rl_method)

    env.show_policy(table_policy)

    table_policy.evaluate()

    env.show_v_table(v_table)
    


def test_q_value_iteration(env):
    q_table = env.build_Q_table()
    transition_table= env.P
    policy_table = env.build_policy_table()

    rl_method = Q_Value_Iteration_Method(q_table,transition_table)

    table_policy= Tabular_Policy(policy_table,rl_method)
    table_policy.improve()
    
    env.show_policy(table_policy)

def test_v_value_iteration(env):
    v_table = env.build_V_table()
    transition_table= env.P
    policy_table = env.build_policy_table()

    rl_method = V_Value_Iteration_Method(v_table,transition_table)

    table_policy= Tabular_Policy(policy_table,rl_method)
    table_policy.improve()
    
    env.show_policy(table_policy)

def test_policy_iteration(env):
    v_table = env.build_V_table()
    transition_table = env.P 
    policy_table = env.build_policy_table()

    for state_index, action_probablities in  policy_table.items():
        distribution = create_distribution_randomly()(action_probablities)
        policy_table[state_index]=distribution
        
    rl_method = Policy_Iteration_Method(v_table,transition_table)
    table_policy= Tabular_Policy(policy_table,rl_method)

    delta = 1e-5
    while True:
        
        env.show_policy(table_policy)
        table_policy.evaluate()
        

        current_delta= table_policy.improve()
      
        if current_delta<delta:
            break

    

if __name__ == "__main__":
    main()
