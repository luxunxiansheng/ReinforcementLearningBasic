
import sys
sys.path.append("/home/ornot/workspace/ReinforcementLearningBasic/work")

import copy

from algorithm.value_based.tabular_solution_method.monte_carlo_method.monte_carlo_off_policy_evaluation import MonteCarloOffPolicyEvaluation
from algorithm.value_based.tabular_solution_method.monte_carlo_method.monte_carlo_off_policy_control import MonteCarloOffPolicyControl
from policy.policy import TabularPolicy
from test_setup import get_env

real_env = get_env("grid_world")

def test_mc_offpolicy_control_method(env):
    q_table = env.build_Q_table()
    policy_table = env.build_policy_table()
    behavior_policy = TabularPolicy(policy_table)
    target_policy = copy.deepcopy(behavior_policy)
    rl_method = MonteCarloOffPolicyControl(q_table, behavior_policy, target_policy, env,100)
    rl_method.improve()
    env.show_policy(rl_method.get_optimal_policy())


test_mc_offpolicy_control_method(real_env)


from env.blackjack import BlackjackEnv

def test_mc_offpolicy_evaluation_method_for_blackjack():
    env = BlackjackEnv()

    q_table = env.build_Q_table()

    # Random behavior policy
    b_policy_table = env.build_policy_table()
    b_policy = TabularPolicy(b_policy_table)

    # spcific target policy only for blackjack
    t_policy_table = env.build_policy_table()
    for state_index, _ in t_policy_table.items():
        card_sum = state_index[0]
        if card_sum < 20:
            t_policy_table[state_index][BlackjackEnv.HIT] = 1.0
            t_policy_table[state_index][BlackjackEnv.STICK] = 0.0
        else:
            t_policy_table[state_index][BlackjackEnv.HIT] = 0.0
            t_policy_table[state_index][BlackjackEnv.STICK] = 1.0
    t_policy = TabularPolicy(t_policy_table)

    error = {}
    init_state = env.reset(False)
    for episode in range(10000):
        state_value = 0.0
        for _ in range(100):
            rl_method = MonteCarloOffPolicyEvaluation(q_table, b_policy, t_policy, env, episode)
            current_q_value= rl_method.evaluate()
            state_value = state_value + current_q_value[init_state][BlackjackEnv.HIT]+0.27726
        error[episode] = state_value*state_value/100
        print("{}:{:.3f}".format(episode, error[episode]))
    
test_mc_offpolicy_evaluation_method_for_blackjack()



# %%
