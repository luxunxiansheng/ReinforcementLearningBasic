import sys,os
current_dir= os.path.dirname(os.path.realpath(__file__))
work_folder=current_dir[:current_dir.find('algorithm')]
sys.path.append(work_folder)

import numpy as np

from algorithm.value_based.tabular_solution_method.td_method.sarsa import SARSAAgent
from algorithm.value_based.tabular_solution_method.td_method.q_learning import  QLearningAgent
from algorithm.value_based.tabular_solution_method.td_method.expected_sarsa import ExpectedSARSAAgent
from algorithm.value_based.tabular_solution_method.td_method.double_q_learning import DoubleQLearningAgent
from algorithm.value_based.tabular_solution_method.td_method.q_lambda import QLambdaAgent
from algorithm.value_based.tabular_solution_method.td_method.sarsa_lambda import  SARSALambdaAgent
from algorithm.value_based.tabular_solution_method.td_method.td_n_steps_sarsa import  TDNStepsSARSAAgent
from algorithm.value_based.tabular_solution_method.td_method.td_n_steps_expected_sarsa import  TDNStepsExpectedSARSAAgent
from algorithm.value_based.tabular_solution_method.td_method.double_q_learning import DoubleQLearningAgent
from algorithm.value_based.tabular_solution_method.td_method.dyna_q import DynaQAgent
from lib.plotting import EpisodeStats, plot_episode_stats
from test_setup import get_env


num_episodes = 1000
n_steps = 2


def test_n_setps_expected_sarsa_method(env):

    n_steps_expectedsarsa_statistics = EpisodeStats("N_Steps_Expected_Sarsa", episode_lengths=np.zeros(num_episodes), episode_rewards=np.zeros(num_episodes), q_value=None)

    n_sarsa_method = TDNStepsExpectedSARSAAgent(env,n_steps,n_steps_expectedsarsa_statistics, num_episodes)
    n_sarsa_method.learn()
    return n_steps_expectedsarsa_statistics
    
def test_n_steps_sarsa_method(env):

    n_sarsa_statistics = EpisodeStats("N_Steps_Sarsa", episode_lengths=np.zeros(num_episodes), episode_rewards=np.zeros(num_episodes), q_value=None)

    n_sarsa_method = TDNStepsSARSAAgent(env,n_steps,n_sarsa_statistics, num_episodes)
    n_sarsa_method.learn()
    return n_sarsa_statistics


def test_qlearning_method(env):
    q_learning_statistics = EpisodeStats("Q_Learning", episode_lengths=np.zeros(num_episodes), episode_rewards=np.zeros(num_episodes), q_value=None)
    qlearning_method = QLearningAgent(env, q_learning_statistics, num_episodes)
    qlearning_method.learn()
    qlearning_method.test()

    return q_learning_statistics

def test_sarsa_method(env):
    sarsa_statistics = EpisodeStats("sarsa", episode_lengths=np.zeros(num_episodes), episode_rewards=np.zeros(num_episodes), q_value=None)
    sarsa_method = SARSAAgent(env,sarsa_statistics, num_episodes)
    sarsa_method.learn()
    sarsa_method.test()
    return sarsa_statistics

def test_expected_sarsa_method(env):
    expectedsarsa_statistics = EpisodeStats("Expected_Sarsa", episode_lengths=np.zeros(num_episodes), episode_rewards=np.zeros(num_episodes), q_value=None)
    expectedsarsa_method = ExpectedSARSAAgent(env, expectedsarsa_statistics, num_episodes)
    expectedsarsa_method.learn()
    expectedsarsa_method.test()

    return expectedsarsa_statistics

def test_double_q_learning_method(env):
    double_q_learning_statistics = EpisodeStats("Double_Q_Learning", episode_lengths=np.zeros(num_episodes), episode_rewards=np.zeros(num_episodes), q_value=None)

    double_qlearning_method = DoubleQLearningAgent(env, double_q_learning_statistics, num_episodes)
    double_qlearning_method.learn()
    double_qlearning_method.test()

    return double_q_learning_statistics

def test_q_lambda_method(env):

    q_lambda_statistics = EpisodeStats("Q_labmda", episode_lengths=np.zeros(
        num_episodes), episode_rewards=np.zeros(num_episodes), q_value=None)
    q_lambda_method = QLambdaAgent(env,q_lambda_statistics, num_episodes)
    q_lambda_method.learn()

    return q_lambda_statistics

def test_sarsa_lambda_method(env):
    sarsa_statistics = EpisodeStats("sarsa_labmda", episode_lengths=np.zeros(num_episodes), episode_rewards=np.zeros(num_episodes), q_value=None)
    sarsa_method = SARSALambdaAgent(env,sarsa_statistics, num_episodes)
    sarsa_method.learn()
    return sarsa_statistics


def test_dynaQ_method_trival(env):


    dyna_q_statistics = EpisodeStats("Dyna_Q_trival", episode_lengths=np.zeros(
        num_episodes), episode_rewards=np.zeros(num_episodes), q_value=None)

    dyna_q_method = DynaQAgent(env,dyna_q_statistics,num_episodes)
    dyna_q_method.learn()

    return dyna_q_statistics

def test_dynaQ_method_priority(env):

    dyna_q_statistics = EpisodeStats("Dyna_Q_priority", episode_lengths=np.zeros(
        num_episodes), episode_rewards=np.zeros(num_episodes), q_value=None)

    dyna_q_method = DynaQAgent(env,dyna_q_statistics,num_episodes,model_type=DynaQAgent.PRIORITY)
    dyna_q_method.learn()

    return dyna_q_statistics

def test_td_control_method(env):
    """
    plot_episode_stats([test_expected_sarsa_method(env),test_n_setps_expected_sarsa_method(env),test_off_policy_n_steps_sarsa(env),
                        test_n_steps_sarsa_method(env),test_qlearning_method(env),test_sarsa_lambda_method(env),test_q_lambda_method(env)])
    
    """
    plot_episode_stats([test_qlearning_method(env),test_sarsa_method(env),test_expected_sarsa_method(env)])

real_env = get_env("CliffWalkingEnv")
test_td_control_method(real_env)
