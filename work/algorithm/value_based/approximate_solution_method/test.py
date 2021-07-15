import sys,os
current_dir= os.path.dirname(os.path.realpath(__file__))
work_folder=current_dir[:current_dir.find('algorithm')]
sys.path.append(work_folder)

import numpy as np

from algorithm.value_based.approximate_solution_method.episodic_semi_gradient_expected_sarsa_control import EpisodicSemiGradientExpectedSarsaControl
from algorithm.value_based.approximate_solution_method.episodic_semi_gradient_expected_sarsa_control import ApproximationExpectedSARSACritic
from algorithm.value_based.approximate_solution_method.episodic_semi_gradient_q_learning_control import  EpisodicSemiGradientQLearningControl
from algorithm.value_based.approximate_solution_method.episodic_semi_gradient_sarsa_control import ApproximationSARSACritic, EpisodicSemiGradientSarsaControl
from algorithm.value_based.approximate_solution_method.estimator.q_value_estimator import TileCodingBasesQValueEstimator



from lib import plotting
from policy.policy import ContinuousStateValueBasedPolicy, DiscreteStateValueBasedPolicy
from test_setup import get_env

num_episodes = 100
n_steps = 4


def test_approximation_control_sarsa(env):

    observation_space = env.observation_space

    tile_coding_step_size = 0.3
    estimator = TileCodingBasesQValueEstimator(tile_coding_step_size, observation_space.high[0], observation_space.low[0], observation_space.high[1], observation_space.low[1])

    q_v = plotting.QValue(env.observation_space_name[0], env.observation_space_name[1], estimator)
    approximation_control_statistics = plotting.EpisodeStats("Q_Learning", episode_lengths=np.zeros(num_episodes), episode_rewards=np.zeros(num_episodes), q_value=q_v)

    episodicsemigradsarsacontrol = EpisodicSemiGradientSarsaControl(env,estimator,approximation_control_statistics,num_episodes)
    episodicsemigradsarsacontrol.learn()

    return approximation_control_statistics

def test_approximation_control_expected_sarsa(env):

    observation_space = env.observation_space

    tile_coding_step_size = 0.3
    q_function = TileCodingBasesQValueEstimator(tile_coding_step_size, observation_space.high[0], observation_space.low[0], observation_space.high[1], observation_space.low[1])

    continuous_state_policy = ContinuousStateValueBasedPolicy()
    
    q_v = plotting.QValue(env.observation_space_name[0], env.observation_space_name[1], q_function)
    approximation_control_statistics = plotting.EpisodeStats("Expected SARSA", episode_lengths=np.zeros(num_episodes), episode_rewards=np.zeros(num_episodes), q_value=q_v)

    critic = ApproximationExpectedSARSACritic(env,q_function)
    actor  = ESoftActor(continuous_state_policy,critic)

    episodicsemigradsarsacontrol = EpisodicSemiGradientExpectedSarsaControl(critic,actor,env,approximation_control_statistics,num_episodes)
    episodicsemigradsarsacontrol.explore()

    return approximation_control_statistics

def test_approximation_control_q_learning(env):

    observation_space = env.observation_space

    tile_coding_step_size = 0.3
    estimator = TileCodingBasesQValueEstimator(tile_coding_step_size, observation_space.high[0], observation_space.low[0], observation_space.high[1], observation_space.low[1])

    q_v = plotting.QValue(env.observation_space_name[0], env.observation_space_name[1], estimator)
    approximation_control_statistics = plotting.EpisodeStats("Q_Learning", episode_lengths=np.zeros(num_episodes), episode_rewards=np.zeros(num_episodes), q_value=q_v)

    episodicsemigradsarsacontrol = EpisodicSemiGradientQLearningControl(env,estimator,approximation_control_statistics,num_episodes)
    episodicsemigradsarsacontrol.learn()

    return approximation_control_statistics


def test_approximation_control_method(env):

    episode_stats = [test_approximation_control_sarsa(env)]
    
    plotting.plot_episode_stats(episode_stats)
    plotting.plot_3d_q_value(env, episode_stats)


real_env = get_env("MountainCarEnv")

#test_approximation_evaluation(real_env)

test_approximation_control_method(real_env)




