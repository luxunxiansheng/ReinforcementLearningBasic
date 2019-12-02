from policy.policy import Tabular_Implicit_Policy
from gym.envs import discrte_env

def policy_evaluate(tabular_implict_policy,env_transition):
    """
    Evaluate the poilcy with Bellman Equation
    
    Arguments:
        tabular_implict_policy {Tabular_Implict_Policy} -- policy to be evaulatd 
        discrte_env {Env} -- the env which dynamic is told 
    """
    P[s][UP] = [(1.0, s, reward, True)]


    assert   isinstance(tabular_implict_policy, Tabular_Implicit_Policy) 
   

    for state in discrte_env:   #  For each state in env
        for action_index_of_current_state in range(len(tabular_implict_policy.Q_table[state])):  # For each action in current state
            value_of_action = 0 
            for transition_prob,next_state,reward,_ in env_transition[state][action_index_of_current_state]: # For each next state 
                value_of_next_state = _get_value_of_state(tabular_implict_policy, next_state)                     
                value_of_action+=transition_prob*value_of_next_state 

def _get_value_of_state(tabular_implict_policy, next_state):
    value_of_next_state = 0 
    actions_values_of_next_state = tabular_implict_policy.Q_table[next_state]
    for action_index_of_next_state in range(len(actions_values_of_next_state)):
        value_of_next_state+=tabular_implict_policy.get_probability(next_state,action_index_of_next_state)*actions_values_of_next_state[action_index_of_next_state]                     
    return value_of_next_state
                

                


