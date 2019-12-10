
import copy

import numpy as np
import pysnooper
from tqdm import tqdm




def policy_evaluate(tabular_implict_policy, discrte_env):
    """
    Evaluate the poilcy on Q_table with Bellman Equation

    Arguments:
        tabular_implict_policy {Tabular_Implict_Policy} -- policy to be evaulatd 
        discrte_env {Env} -- the env which dynamic is told 
    """
    assert isinstance(tabular_implict_policy, Tabular_Implicit_Policy)

    new_q_table = copy.deepcopy(tabular_implict_policy.Q_table)
    for state_index,action_values in tabular_implict_policy.Q_table.items():
        for action_index ,_ in action_values.items():
            # the reward is also related to the next state
            value_of_action = get_value_of_action(tabular_implict_policy,discrte_env,state_index,action_index)
            new_q_table[state_index][action_index] = value_of_action
    tabular_implict_policy.Q_table = new_q_table


def get_value_of_action(tabular_implict_policy,discrte_env,state_index,action_index,discount=1.0):
   
    value_of_action = 0
    current_env_transition = discrte_env.P[state_index][action_index]
    for transition_prob, next_state_index, reward, done in current_env_transition:  # For each next state
        value_of_next_state = 0 if done else get_value_of_state(tabular_implict_policy, next_state_index)
        value_of_action += transition_prob*(discount*value_of_next_state+reward)
    return value_of_action


def get_value_of_state(tabular_implict_policy, state_index):
       
    value_of_state = 0
    action_values_of_state = tabular_implict_policy.Q_table[state_index]
    for action_index,action_value in action_values_of_state.items():
        value_of_state += tabular_implict_policy.get_probability(state_index, action_index)*action_value
    return value_of_state


def policy_improve(tabular_implict_policy):
    tabular_implict_policy.set_action_selector(Greedy_Action_Selector())



