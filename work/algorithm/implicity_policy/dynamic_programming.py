
import copy

import numpy as np
from tqdm import tqdm

from policy.policy import Tabular_Implicit_Policy,Greedy_Action_Selector
import pysnooper



def policy_evaluate(tabular_implict_policy, discrte_env):
    """
    Evaluate the poilcy on Q_table with Bellman Equation

    Arguments:
        tabular_implict_policy {Tabular_Implict_Policy} -- policy to be evaulatd 
        discrte_env {Env} -- the env which dynamic is told 
    """
    assert isinstance(tabular_implict_policy, Tabular_Implicit_Policy)

    new_q_table = copy.deepcopy(tabular_implict_policy.Q_table)
    for state_index in range(discrte_env.nS):  # For each state in env
        # For each action in current state
        for action_index_of_current_state in range(len(tabular_implict_policy.Q_table[state_index])):
            # the reward is also related to the next state
            value_of_action = get_value_of_action(tabular_implict_policy,discrte_env,state_index,action_index_of_current_state)
            new_q_table[state_index][action_index_of_current_state] = value_of_action
    tabular_implict_policy.Q_table = new_q_table


def get_value_of_action(tabular_implict_policy,discrte_env,state_index,action_index,discount=1.0):
   
    value_of_action = 0
    current_env_transition = discrte_env.P[state_index][action_index]
    for transition_prob, next_state_index, reward, _ in current_env_transition:  # For each next state
        value_of_next_state = get_value_of_state(tabular_implict_policy, next_state_index)
        value_of_action += transition_prob*(discount*value_of_next_state+reward)
    return value_of_action


def get_value_of_state(tabular_implict_policy, state_index):
       
    value_of_state = 0
    action_values_of_state = tabular_implict_policy.Q_table[state_index]
    for action_index_of_state in range(len(action_values_of_state)):
        value_of_state += tabular_implict_policy.get_probability(state_index, action_index_of_state)*action_values_of_state[action_index_of_state]
    return value_of_state


def policy_improve(tabular_implict_policy):
    tabular_implict_policy.set_action_selector(Greedy_Action_Selector())

def value_iteration(tabular_implict_policy, discrte_env):
    assert isinstance(tabular_implict_policy, Tabular_Implicit_Policy)
    new_q_table = copy.deepcopy(tabular_implict_policy.Q_table)

    for state_index in range(discrte_env.nS):  # For each state in env
        # For each action in current state
        for action_index_of_current_state in range(len(tabular_implict_policy.Q_table[state_index])):
            optimal_value_of_action = get_optimal_value_of_action(tabular_implict_policy,discrte_env,state_index,action_index_of_current_state)  
            new_q_table[state_index][action_index_of_current_state] = optimal_value_of_action
    
    tabular_implict_policy.Q_table = new_q_table


def get_optimal_value_of_action(tabular_implict_policy,discrte_env,state_index,action_index,discount=1.0):
    current_env_transition = discrte_env.P[state_index][action_index]
    optimal_value_of_action = 0
    for transition_prob, next_state_index, reward, _ in current_env_transition:  # For each next state
        optimal_value_of_next_state = get_optimal_value_of_state(tabular_implict_policy, next_state_index)
        # the reward is also related to the next state
        optimal_value_of_action += transition_prob*(discount*optimal_value_of_next_state+reward)
    return optimal_value_of_action


def get_optimal_value_of_state(tabular_implict_policy, state_index):
    action_values_of_state = tabular_implict_policy.Q_table[state_index]

    return action_values_of_state[np.argmax(action_values_of_state)]