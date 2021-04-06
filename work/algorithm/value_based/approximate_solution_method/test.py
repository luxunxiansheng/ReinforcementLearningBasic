import sys,os
current_dir= os.path.dirname(os.path.realpath(__file__))
work_folder=current_dir[:current_dir.find('algorithm')]
sys.path.append(work_folder)

import numpy as np

from algorithm.value_based.approximate_solution_method.episodic_semi_gradient_expected_sarsa_control import EpisodicSemiGradientExpectedSarsaControl
from algorithm.value_based.approximate_solution_method.episodic_semi_gradient_expected_sarsa_control import ApproximationExpectedSARSACritic
from algorithm.value_based.approximate_solution_method.episodic_semi_gradient_q_learning_control import ApproximationQLearningCritic, EpisodicSemiGradientQLearningControl
from algorithm.value_based.approximate_solution_method.episodic_semi_gradient_sarsa_control import ApproximationSARSACritic, EpisodicSemiGradientSarsaControl
from algorithm.value_based.approximate_solution_method.estimator.q_value_estimator import TileCodingBasesQValueEstimator
from algorithm.value_based.approximate_solution_method.estimator.v_value_estimator import StateAggregationVValueEstimator
from algorithm.value_based.approximate_solution_method.approximation_common import ESoftActor
from algorithm.value_based.approximate_solution_method.gradient_monte_carlo_evaluation import GradientMonteCarloEvaluation
from algorithm.value_based.approximate_solution_method.semi_gradient_td_lambda_evaluation import SemiGradientTDLambdaEvaluation
from algorithm.value_based.approximate_solution_method.semi_gradient_tdn_evaluation import SemiGradientTDNEvalution

from lib import plotting
from policy.policy import ContinuousStateValueBasedPolicy, DiscreteStateValueBasedPolicy
from test_setup import get_env

num_episodes = 100
n_steps = 4


def test_approximation_evaluation(env):
    b_policy_table = env.build_policy_table()
    b_policy = DiscreteStateValueBasedPolicy(b_policy_table)

    distribution = np.zeros(env.nS)

    vf = StateAggregationVValueEstimator(env.nS, 3)
    gradientmcevalution = GradientMonteCarloEvaluation(vf, b_policy, env, episodes=num_episodes, distribution=distribution)
    gradientmcevalution.evaluate()
    mc_sg = plotting.StateValues('gradientMCEvalution', vf)

    vf = StateAggregationVValueEstimator(env.nS, 3)
    semigradienttdlambdaevalution = SemiGradientTDLambdaEvaluation(vf, b_policy, env, episodes=num_episodes, lamda=0.0)
    semigradienttdlambdaevalution.evaluate()
    semi_grident_tdl_sg = plotting.StateValues('semigradientTDLambdaevalution', vf)

    vf = StateAggregationVValueEstimator(env.nS, 3)
    semigradienttdnevalution = SemiGradientTDNEvalution(vf, b_policy, 5, env, episodes=num_episodes, distribution=distribution)
    semigradienttdnevalution.evaluate()
    semi_graident_tdn_sg = plotting.StateValues('semigradientTDNevalution', vf)


    plotting.plot_state_value(env, [semi_grident_tdl_sg, semi_graident_tdn_sg, semi_graident_tdn_sg, mc_sg])


def test_approximation_control_sarsa(env):

    observation_space = env.observation_space

    tile_coding_step_size = 0.3
    q_function = TileCodingBasesQValueEstimator(tile_coding_step_size, observation_space.high[0], observation_space.low[0], observation_space.high[1], observation_space.low[1])
    
    continuous_state_policy = ContinuousStateValueBasedPolicy()

            
    q_v = plotting.QValue(env.observation_space_name[0], env.observation_space_name[1], q_function)
    approximation_control_statistics = plotting.EpisodeStats("SARSA", episode_lengths=np.zeros(num_episodes), episode_rewards=np.zeros(num_episodes), q_value=q_v)

    critic = ApproximationSARSACritic(env,q_function,continuous_state_policy)
    actor  = ESoftActor(continuous_state_policy,critic)

    episodicsemigradsarsacontrol = EpisodicSemiGradientSarsaControl(critic,actor,env,approximation_control_statistics,num_episodes)
    episodicsemigradsarsacontrol.improve()

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
    episodicsemigradsarsacontrol.improve()

    return approximation_control_statistics

def test_approximation_control_q_learning(env):

    observation_space = env.observation_space

    tile_coding_step_size = 0.3
    q_function = TileCodingBasesQValueEstimator(tile_coding_step_size, observation_space.high[0], observation_space.low[0], observation_space.high[1], observation_space.low[1])

    continuous_state_policy = ContinuousStateValueBasedPolicy()
    
    q_v = plotting.QValue(env.observation_space_name[0], env.observation_space_name[1], q_function)
    approximation_control_statistics = plotting.EpisodeStats("Q_Learning", episode_lengths=np.zeros(num_episodes), episode_rewards=np.zeros(num_episodes), q_value=q_v)

    critic = ApproximationQLearningCritic(env,q_function)
    actor  = ESoftActor(continuous_state_policy,critic)

    episodicsemigradsarsacontrol = EpisodicSemiGradientQLearningControl(critic,actor,env,approximation_control_statistics,num_episodes)
    episodicsemigradsarsacontrol.improve()

    return approximation_control_statistics


def test_approximation_control_method(env):

    episode_stats = [test_approximation_control_sarsa(env),test_approximation_control_expected_sarsa(env),test_approximation_control_q_learning(env)]
    
    plotting.plot_episode_stats(episode_stats)
    plotting.plot_3d_q_value(env, episode_stats)


real_env = get_env("mountaincar")

#test_approximation_evaluation(real_env)

test_approximation_control_method(real_env)




