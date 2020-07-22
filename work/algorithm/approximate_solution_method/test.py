import numpy as np

from algorithm.approximate_solution_method.episodic_semi_gradient_expected_sarsa_control import \
    EpisodicSemiGradientExpectedSarsaControl
from algorithm.approximate_solution_method.episodic_semi_gradient_q_learning_control import \
    EpisodicSemiGradientQLearningControl
from algorithm.approximate_solution_method.episodic_semi_gradient_sarsa_control import \
    EpisodicSemiGradientSarsaControl
from algorithm.approximate_solution_method.gradient_monte_carlo_evaluation import \
    GradientMonteCarloEvaluation
from algorithm.approximate_solution_method.q_function import \
    TileCodingBasesQFunction
from algorithm.approximate_solution_method.semi_gradient_td_lambda_evaluation import \
    SemiGradientTDLambdaEvaluation
from algorithm.approximate_solution_method.semi_gradient_tdn_evaluation import \
    SemiGradientTDNEvalution
from algorithm.approximate_solution_method.value_function import (
    FourierBasesValueFunction, PolynomialBasesValueFunction)
from lib import plotting
from lib.utility import create_distribution_epsilon_greedily
from policy.policy import DiscreteActionPolicy, TabularPolicy

num_episodes = 3000
n_steps = 4


def test_approximation_evaluation(env):
    b_policy_table = env.build_policy_table()
    b_policy = TabularPolicy(b_policy_table)

    distribution = np.zeros(env.nS)

    polynomialbasesfunc_semi = PolynomialBasesValueFunction(5)
    semigradienttdnevalution = SemiGradientTDNEvalution(
        polynomialbasesfunc_semi, b_policy, 5, env, episodes=num_episodes, distribution=distribution)
    semigradienttdnevalution.evaluate()
    semi_poly = plotting.StateValues(
        'polynomialbasesfunc_semi', polynomialbasesfunc_semi)


    polynomialbasesfunc = PolynomialBasesValueFunction(5)
    semigradienttdlambdaevalution = SemiGradientTDLambdaEvaluation(
        polynomialbasesfunc, b_policy,env,episodes=num_episodes)
    semigradienttdlambdaevalution.evaluate()
    semi_poly = plotting.StateValues(
        'polynomialbasesfunc_semi_tdlmabda', polynomialbasesfunc_semi)


    polynomialbasesfunc_mc = PolynomialBasesValueFunction(5)
    gradientmcevalution = GradientMonteCarloEvaluation(
        polynomialbasesfunc_mc, b_policy, env,episodes=num_episodes, distribution=distribution)
    gradientmcevalution.evaluate()
    mc_poly = plotting.StateValues(
        'polynomialbasesfunc_mc', polynomialbasesfunc_mc)

    fourierBasesValueFunction_semi = FourierBasesValueFunction(5)
    semigradienttdnevalution = SemiGradientTDNEvalution(
        fourierBasesValueFunction_semi, b_policy, 5, env, episodes=num_episodes,distribution=distribution)
    semigradienttdnevalution.evaluate()
    semi_fourier = plotting.StateValues(
        'fourierBasesValueFunction_semi', fourierBasesValueFunction_semi)

    fourierBasesValueFunction_mc = FourierBasesValueFunction(5)
    gradientmcevalution = GradientMonteCarloEvaluation(
        fourierBasesValueFunction_mc, b_policy, env, episodes=num_episodes,distribution=distribution)
    gradientmcevalution.evaluate()
    mc_fourier = plotting.StateValues(
        'fourierBasesValueFunction_mc', fourierBasesValueFunction_mc)
    plotting.plot_state_value(
        env, [semi_poly, mc_poly, semi_fourier, mc_fourier])


def test_approximation_control_sarsa(env):

    action_space = env.action_space

    observation_space = env.observation_space

    tile_coding_step_size = 0.3
    q_function = TileCodingBasesQFunction(
        tile_coding_step_size, observation_space.high[0], observation_space.low[0], observation_space.high[1], observation_space.low[1])

    discreteactionpolicy = DiscreteActionPolicy(action_space, q_function)
    discreteactionpolicy.create_distribution_fn = create_distribution_epsilon_greedily(
        0.1)

    q_v = plotting.QValue('Position', 'Speed', q_function)
    approximation_control_statistics = plotting.EpisodeStats("SARSA", episode_lengths=np.zeros(
        num_episodes), episode_rewards=np.zeros(num_episodes), q_value=q_v)

    episodicsemigradsarsacontrol = EpisodicSemiGradientSarsaControl(
        q_function, discreteactionpolicy, env, approximation_control_statistics, num_episodes)
    episodicsemigradsarsacontrol.improve()

    return approximation_control_statistics


def test_approximation_control_expected_sarsa(env):

    action_space = env.action_space

    observation_space = env.observation_space

    tile_coding_step_size = 0.3
    q_function = TileCodingBasesQFunction(
        tile_coding_step_size, observation_space.high[0], observation_space.low[0], observation_space.high[1], observation_space.low[1])

    discreteactionpolicy = DiscreteActionPolicy(action_space, q_function)
    discreteactionpolicy.create_distribution_fn = create_distribution_epsilon_greedily(
        0.1)

    q_v = plotting.QValue('Position', 'Speed', q_function)
    approximation_control_statistics = plotting.EpisodeStats("Expected SARSA", episode_lengths=np.zeros(
        num_episodes), episode_rewards=np.zeros(num_episodes), q_value=q_v)

    episodicsemigradsarsacontrol = EpisodicSemiGradientExpectedSarsaControl(
        q_function, discreteactionpolicy, env, approximation_control_statistics, num_episodes)
    episodicsemigradsarsacontrol.improve()

    return approximation_control_statistics


def test_approximation_control_q_learning(env):

    action_space = env.action_space

    observation_space = env.observation_space

    tile_coding_step_size = 0.3
    q_function = TileCodingBasesQFunction(
        tile_coding_step_size, observation_space.high[0], observation_space.low[0], observation_space.high[1], observation_space.low[1])

    discreteactionpolicy = DiscreteActionPolicy(action_space, q_function)
    discreteactionpolicy.create_distribution_fn = create_distribution_epsilon_greedily(
        0.1)

    q_v = plotting.QValue('Position', 'Speed', q_function)
    approximation_control_statistics = plotting.EpisodeStats("Q_Learning", episode_lengths=np.zeros(
        num_episodes), episode_rewards=np.zeros(num_episodes), q_value=q_v)

    episodicsemigradsarsacontrol = EpisodicSemiGradientQLearningControl(
        q_function, discreteactionpolicy, env, approximation_control_statistics, num_episodes)
    episodicsemigradsarsacontrol.improve()

    return approximation_control_statistics


def test_approximation_control_method(env):

    episode_stats = [test_approximation_control_sarsa(env), test_approximation_control_expected_sarsa(
        env), test_approximation_control_q_learning(env)]
    plotting.plot_episode_stats(episode_stats)
    plotting.plot_2d_q_value(env, episode_stats)
