
import copy

from tqdm import tqdm

from policy.policy import Tabular_Implicit_Policy


def policy_evaluate(tabular_implict_policy,discrte_env):
    """
    Evaluate the poilcy on Q_table with Bellman Equation
    
    Arguments:
        tabular_implict_policy {Tabular_Implict_Policy} -- policy to be evaulatd 
        discrte_env {Env} -- the env which dynamic is told 
    """
    assert   isinstance(tabular_implict_policy, Tabular_Implicit_Policy) 
   
    new_q_table =  copy.deepcopy(tabular_implict_policy.Q_table)

    for state in range(discrte_env.nS):   #  For each state in env
        for action_index_of_current_state in range(len(tabular_implict_policy.Q_table[state])):  # For each action in current state
            current_env_transition= discrte_env.P[state][action_index_of_current_state]           
            value_of_action = get_value_of_action(current_env_transition, tabular_implict_policy)  #  the reward is also related to the next state
            new_q_table[state][action_index_of_current_state]= value_of_action  
    
    tabular_implict_policy.Q_table = new_q_table  

def get_value_of_action(current_env_transition, tabular_implict_policy):
    value_of_action = 0 
    for transition_prob,next_state,reward,_ in current_env_transition: # For each next state 
        value_of_next_state = get_value_of_state(tabular_implict_policy, next_state)                     
        value_of_action+=(transition_prob*value_of_next_state+reward)  #  the reward is also related to the next state
    return value_of_action

            
def get_value_of_state(tabular_implict_policy, state):
    value_of_state = 0 
    action_values_of_state = tabular_implict_policy.Q_table[state]
    for action_index_of_state in range(len(action_values_of_state)):
        value_of_state+=tabular_implict_policy.get_probability(state,action_index_of_state)*action_values_of_state[action_index_of_state]                     
    return value_of_state
