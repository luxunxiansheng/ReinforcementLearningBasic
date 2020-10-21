import sys
sys.path.append("/home/ornot/workspace/ReinforcementLearningBasic/work")

import numpy as np

from algorithm.value_based.approximate_solution_method.episodic_semi_gradient_expected_sarsa_control import EpisodicSemiGradientExpectedSarsaControl
from algorithm.value_based.approximate_solution_method.episodic_semi_gradient_q_learning_control import EpisodicSemiGradientQLearningControl
from algorithm.value_based.approximate_solution_method.episodic_semi_gradient_sarsa_control import EpisodicSemiGradientSarsaControl
from algorithm.value_based.approximate_solution_method.estimator.q_value_estimator import TileCodingBasesQValueEstimator
from algorithm.value_based.approximate_solution_method.estimator.v_value_estimator import (FourierBasesVValueEsimator, PolynomialBasesVValueEsitmator,
    StateAggregationVValueEstimator)
from algorithm.value_based.approximate_solution_method.gradient_monte_carlo_evaluation import GradientMonteCarloEvaluation
from algorithm.value_based.approximate_solution_method.semi_gradient_td_lambda_evaluation import SemiGradientTDLambdaEvaluation
from algorithm.value_based.approximate_solution_method.semi_gradient_tdn_evaluation import SemiGradientTDNEvalution
from lib import plotting
from lib.utility import create_distribution_epsilon_greedily
from policy.policy import ContinuousStateValueBasedPolicy
from test_setup import get_env

num_episodes = 100
n_steps = 4


def test_approximation_evaluation(env):
    b_policy_table = env.build_policy_table()
    b_policy = ContinuousStateValueBasedPolicy(b_policy_table)

    distribution = np.zeros(env.nS)

    vf = StateAggregationVValueEstimator(env.nS, 3)
    semigradienttdlambdaevalution = SemiGradientTDLambdaEvaluation(vf, b_policy, env, episodes=num_episodes, lamda=0.0)
    semigradienttdlambdaevalution.evaluate()
    semi_grident_tdl_sg = plotting.StateValues('semigradientTDLambdaevalution', vf)

    vf = StateAggregationVValueEstimator(env.nS, 3)
    semigradienttdnevalution = SemiGradientTDNEvalution(vf, b_policy, 5, env, episodes=num_episodes, distribution=distribution)
    semigradienttdnevalution.evaluate()
    semi_graident_tdn_sg = plotting.StateValues('semigradientTDNevalution', vf)

    vf = StateAggregationVValueEstimator(env.nS, 3)
    gradientmcevalution = GradientMonteCarloEvaluation(vf, b_policy, env, episodes=num_episodes, distribution=distribution)
    gradientmcevalution.evaluate()
    mc_sg = plotting.StateValues('gradientMCEvalution', vf)

    plotting.plot_state_value(env, [semi_grident_tdl_sg, semi_graident_tdn_sg, semi_graident_tdn_sg, mc_sg])


def test_approximation_control_sarsa(env):

    action_space = env.action_space

    observation_space = env.observation_space

    tile_coding_step_size = 0.3
    q_function = TileCodingBasesQValueEstimator(tile_coding_step_size, observation_space.high[0], observation_space.low[0], observation_space.high[1], observation_space.low[1])
    
    continuous_state_policy = ContinuousStateValueBasedPolicy(action_space, q_function,create_distribution_epsilon_greedily(0.1))

    q_v = plotting.QValue('Position', 'Speed', q_function)
    approximation_control_statistics = plotting.EpisodeStats("SARSA", episode_lengths=np.zeros(num_episodes), episode_rewards=np.zeros(num_episodes), q_value=q_v)

    episodicsemigradsarsacontrol = EpisodicSemiGradientSarsaControl(q_function, continuous_state_policy, env, approximation_control_statistics, num_episodes)
    episodicsemigradsarsacontrol.improve()

    return approximation_control_statistics


def test_approximation_control_expected_sarsa(env):

    action_space = env.action_space

    observation_space = env.observation_space

    tile_coding_step_size = 0.3
    q_function = TileCodingBasesQValueEstimator(tile_coding_step_size, observation_space.high[0], observation_space.low[0], observation_space.high[1], observation_space.low[1])

    continuous_state_policy = ContinuousStateValueBasedPolicy(action_space, q_function,create_distribution_epsilon_greedily(0.1))
    
    q_v = plotting.QValue('Position', 'Speed', q_function)
    approximation_control_statistics = plotting.EpisodeStats("Expected SARSA", episode_lengths=np.zeros(num_episodes), episode_rewards=np.zeros(num_episodes), q_value=q_v)

    episodicsemigradsarsacontrol = EpisodicSemiGradientExpectedSarsaControl(q_function, continuous_state_policy, env, approximation_control_statistics, num_episodes)
    episodicsemigradsarsacontrol.improve()

    return approximation_control_statistics


def test_approximation_control_q_learning(env):

    action_space = env.action_space

    observation_space = env.observation_space

    tile_coding_step_size = 0.3
    q_function = TileCodingBasesQValueEstimator(tile_coding_step_size, observation_space.high[0], observation_space.low[0], observation_space.high[1], observation_space.low[1])

    continuous_state_policy = ContinuousStateValueBasedPolicy(action_space, q_function,create_distribution_epsilon_greedily(0.1))
    
    q_v = plotting.QValue('Position', 'Speed', q_function)
    approximation_control_statistics = plotting.EpisodeStats("Q_Learning", episode_lengths=np.zeros(num_episodes), episode_rewards=np.zeros(num_episodes), q_value=q_v)

    episodicsemigradsarsacontrol = EpisodicSemiGradientQLearningControl(q_function, continuous_state_policy, env, approximation_control_statistics, num_episodes)
    episodicsemigradsarsacontrol.improve()

    return approximation_control_statistics


def test_approximation_control_method(env):

    episode_stats = [test_approximation_control_sarsa(env),test_approximation_control_expected_sarsa(env),test_approximation_control_q_learning(env)]
    plotting.plot_episode_stats(episode_stats)
    plotting.plot_3d_q_value(env, episode_stats)


real_env = get_env("mountaincar")

test_approximation_control_method(real_env)




