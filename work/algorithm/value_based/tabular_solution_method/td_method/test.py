import sys
sys.path.append("/home/ornot/workspace/ReinforcementLearningBasic/work")


from lib.plotting import plot_episode_error
import numpy as np

from tqdm import tqdm
from algorithm.value_based.tabular_solution_method.td_method.double_q_learning import DoubleQLearning
from algorithm.value_based.tabular_solution_method.td_method.dyna_q import (PRIORITY, TRIVAL, DynaQ)
from algorithm.value_based.tabular_solution_method.td_method.expected_sarsa import ExpectedSARSA
from algorithm.value_based.tabular_solution_method.td_method.n_steps_expected_sarsa import NStepsExpectedSARSA
from algorithm.value_based.tabular_solution_method.td_method.n_steps_sarsa import NStepsSARSA
from algorithm.value_based.tabular_solution_method.td_method.off_policy_n_steps_sarsa import OffPolicyNStepsSARSA
from algorithm.value_based.tabular_solution_method.td_method.q_lambda import QLambda
from algorithm.value_based.tabular_solution_method.td_method.q_learning import QLearning
from algorithm.value_based.tabular_solution_method.td_method.sarsa import SARSA
from algorithm.value_based.tabular_solution_method.td_method.sarsa_lambda import SARSALambda
from algorithm.value_based.tabular_solution_method.td_method.td0_evaluation import TD0Evalutaion
from algorithm.value_based.tabular_solution_method.td_method.td_lambda_evaluation import TDLambdaEvalutaion
from algorithm.value_based.tabular_solution_method.td_method.tdn_evaluation import TDNEvalutaion
from lib.plotting import plot_episode_error
from policy.policy import TabularPolicy
from test_setup import get_env
from env.blackjack import BlackjackEnv


num_episodes = 100000
n_steps = 1

real_env = get_env("blackjack")


def test_td0_evaluation_method_for_blackjack():
    env = BlackjackEnv()

    v_table = env.build_V_table()

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

    error = []
    init_state = env.reset(False)
    for episodes in tqdm(range(5000)):
        error_square = 0.0
        rounds = 1
        for _ in range(rounds):
            rl_method = TD0Evalutaion(v_table,t_policy, env,episodes)
            current_value= rl_method.evaluate()
            error_square = error_square+(current_value[init_state] + 0.27726)*(current_value[init_state] + 0.27726)
        
        error.append(error_square/rounds)

    plot_episode_error(error)        


test_td0_evaluation_method_for_blackjack()




def test_tdn_evaluation_method_for_blackjack():
    env = BlackjackEnv()

    v_table = env.build_V_table()

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

    
    error = []
    init_state = env.reset(False)
    for episode in tqdm(range(500)):
        error_square = 0.0
        for _ in range(10):
            rl_method = TDNEvalutaion(v_table,t_policy, env, n_steps,episodes=100)
            current_value= rl_method.evaluate()
            error_square = error_square+(current_value[init_state] + 0.27726)*(current_value[init_state] + 0.27726)
        
        error.append(error_square/100)

    plot_episode_error(error)        

test_tdn_evaluation_method_for_blackjack()


def test_td_lambda_evalution_method(env):
    env = BlackjackEnv()

    v_table = env.build_V_table()

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

    error = []
    init_state = env.reset(False)
    for episode in tqdm(range(500)):
        error_square = 0.0
        for _ in range(10):
            rl_method = TDLambdaEvalutaion(v_table,t_policy, env,episodes=100)
            current_value= rl_method.evaluate()
            error_square = error_square+(current_value[init_state] + 0.27726)*(current_value[init_state] + 0.27726)
        
        error.append(error_square/100)

    plot_episode_error(error)     


test_td_lambda_evalution_method()


def test_sarsa_method(env):
    q_table = env.build_Q_table()
    b_policy_table = env.build_policy_table()
    b_policy = TabularPolicy(b_policy_table)

    sarsa_statistics = plotting.EpisodeStats("sarsa", episode_lengths=np.zeros(num_episodes), episode_rewards=np.zeros(num_episodes),q_value=None)
    sarsa_method = SARSA(q_table, b_policy, 0.1, env, sarsa_statistics, num_episodes)
    sarsa_method.improve()

    return sarsa_statistics

def test_sarsa_lambda_method(env):
    q_table = env.build_Q_table()
    b_policy_table = env.build_policy_table()
    b_policy = TabularPolicy(b_policy_table)

    sarsa_statistics = plotting.EpisodeStats("sarsa_labmda", episode_lengths=np.zeros(num_episodes), episode_rewards=np.zeros(num_episodes),q_value=None)
    sarsa_method = SARSALambda(q_table, b_policy, 0.1, env, sarsa_statistics, num_episodes)
    sarsa_method.improve()

    return sarsa_statistics


def test_q_lambda_method(env):
    q_table = env.build_Q_table()
    b_policy_table = env.build_policy_table()
    b_policy = TabularPolicy(b_policy_table)

    q_lambda_statistics = plotting.EpisodeStats("Q_labmda", episode_lengths=np.zeros(num_episodes), episode_rewards=np.zeros(num_episodes),q_value=None)
    q_lambda_method = QLambda(q_table, b_policy, 0.1, env, q_lambda_statistics, num_episodes)
    q_lambda_method.improve()

    return q_lambda_statistics

def test_qlearning_method(env):
    q_table = env.build_Q_table()
    b_policy_table = env.build_policy_table()
    b_policy = TabularPolicy(b_policy_table)

    q_learning_statistics = plotting.EpisodeStats("Q_Learning", episode_lengths=np.zeros(num_episodes), episode_rewards=np.zeros(num_episodes),q_value=None)

    qlearning_method = QLearning(
        q_table, b_policy, 0.1, env, q_learning_statistics, num_episodes)
    qlearning_method.improve()

    return q_learning_statistics


def test_expected_sarsa_method(env):

    q_table = env.build_Q_table()
    b_policy_table = env.build_policy_table()
    b_policy = TabularPolicy(b_policy_table)

    expectedsarsa_statistics = plotting.EpisodeStats("Expected_Sarsa", episode_lengths=np.zeros(num_episodes), episode_rewards=np.zeros(num_episodes),q_value=None)

    expectedsarsa_method = ExpectedSARSA(
        q_table, b_policy, 0.1, env, expectedsarsa_statistics, num_episodes)
    expectedsarsa_method.improve()

    return expectedsarsa_statistics


def test_double_q_learning_method(env):

    q_table = env.build_Q_table()
    b_policy_table = env.build_policy_table()
    b_policy = TabularPolicy(b_policy_table)

    double_q_learning_statistics = plotting.EpisodeStats("Double_Q_Learning", episode_lengths=np.zeros(num_episodes), episode_rewards=np.zeros(num_episodes),q_value=None)

    double_qlearning_method = DoubleQLearning(
        q_table, b_policy, 0.1, env, double_q_learning_statistics, num_episodes)
    double_qlearning_method.improve()

    return double_q_learning_statistics


def test_tdn_evaluaiton_method(env):
    v_table = env.build_V_table()
    b_policy_table = env.build_policy_table()
    b_policy = TabularPolicy(b_policy_table)
    td0_method = TDNEvalutaion(v_table, b_policy, env, 1)
    td0_method.evaluate()


def test_n_steps_sarsa_method(env):

    q_table = env.build_Q_table()
    b_policy_table = env.build_policy_table()
    b_policy = TabularPolicy(b_policy_table)
    n_sarsa_statistics = plotting.EpisodeStats("N_Steps_Sarsa", episode_lengths=np.zeros(num_episodes), episode_rewards=np.zeros(num_episodes),q_value=None)
    n_sarsa_method = NStepsSARSA(
        q_table, b_policy, 0.1, env,  n_steps, n_sarsa_statistics, num_episodes)
    n_sarsa_method.improve()
    return n_sarsa_statistics


def test_n_setps_expected_sarsa(env):

    q_table = env.build_Q_table()
    b_policy_table = env.build_policy_table()
    b_policy = TabularPolicy(b_policy_table)

    n_steps_expectedsarsa_statistics = plotting.EpisodeStats("N_Steps_Expected_Sarsa", episode_lengths=np.zeros(num_episodes), episode_rewards=np.zeros(num_episodes),q_value=None)

    n_steps_expectedsarsa_method = NStepsExpectedSARSA(
        q_table, b_policy, 0.1, env,  n_steps, n_steps_expectedsarsa_statistics, num_episodes)
    n_steps_expectedsarsa_method.improve()

    return n_steps_expectedsarsa_statistics


def test_off_policy_n_steps_sarsa(env):
    q_table = env.build_Q_table()

    b_policy_table = env.build_policy_table()
    b_policy = TabularPolicy(b_policy_table)

    t_policy_table = env.build_policy_table()
    t_policy = TabularPolicy(t_policy_table)

    n_steps_off_policy_sarsa_statistics = plotting.EpisodeStats("N_Steps_Offpolicy_Sarsa", episode_lengths=np.zeros(num_episodes), episode_rewards=np.zeros(num_episodes),q_value=None)

    n_steps_offpolicy_sarsa_method = OffPolicyNStepsSARSA(
        q_table, b_policy, t_policy, env, n_steps, n_steps_off_policy_sarsa_statistics, num_episodes)

    n_steps_offpolicy_sarsa_method.improve()

    return n_steps_off_policy_sarsa_statistics


def test_dynaQ_method_trival(env):
    q_table = env.build_Q_table()
    b_policy_table = env.build_policy_table()
    b_policy = TabularPolicy(b_policy_table)

    dyna_q_statistics = plotting.EpisodeStats("Dyna_Q", episode_lengths=np.zeros(num_episodes), episode_rewards=np.zeros(num_episodes),q_value=None)

    dyna_q_method = DynaQ(q_table, b_policy, 0.1, env, dyna_q_statistics, num_episodes)
    dyna_q_method.improve()

    return dyna_q_statistics


def test_dynaQ_method_priority(env):
    q_table = env.build_Q_table()
    b_policy_table = env.build_policy_table()
    b_policy = TabularPolicy(b_policy_table)

    dyna_q_statistics = plotting.EpisodeStats("Dyna_Q_PRIORITY", episode_lengths=np.zeros(num_episodes), episode_rewards=np.zeros(num_episodes),q_value=None)

    dyna_q_method = DynaQ(q_table, b_policy, 0.1, env,dyna_q_statistics, num_episodes, mode=PRIORITY)
    dyna_q_method.improve()

    return dyna_q_statistics


def test_td_control_method(env):
    plotting.plot_episode_stats([test_sarsa_method(env),test_sarsa_lambda_method(env),test_q_lambda_method(env)])
