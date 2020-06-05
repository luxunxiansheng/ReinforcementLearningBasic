import numpy as np

from algorithm.approximate_solution_method.value_function import (
    PolynomialBasesValueFunction, FourierBasesValueFunction)
from algorithm.approximate_solution_method.gradient_monte_carlo_evaluation import GradientMonteCarloEvaluation
from algorithm.approximate_solution_method.semi_gradient_tdn_evaluation import SemiGradientTDNEvalution
from policy.policy import TabularPolicy, DiscreteActionPolicy
from lib.utility import create_distribution_epsilon_greedily

from lib import plotting


def test_approximation_evaluation(env):
    b_policy_table = env.build_policy_table()
    b_policy = TabularPolicy(b_policy_table)

    distribution = np.zeros(env.nS)

    polynomialbasesfunc_semi = PolynomialBasesValueFunction(5)
    semigradienttdnevalution = SemiGradientTDNEvalution(polynomialbasesfunc_semi, b_policy, 5, env, distribution=distribution)
    semigradienttdnevalution.evaluate() 
    semi_poly = plotting.StateValues('polynomialbasesfunc_semi', polynomialbasesfunc_semi)

    polynomialbasesfunc_mc = PolynomialBasesValueFunction(5)
    gradientmcevalution = GradientMonteCarloEvaluation(polynomialbasesfunc_mc, b_policy, env, distribution=distribution)
    gradientmcevalution.evaluate()
    mc_poly = plotting.StateValues('polynomialbasesfunc_mc', polynomialbasesfunc_mc)

    fourierBasesValueFunction_semi = FourierBasesValueFunction(5)
    semigradienttdnevalution = SemiGradientTDNEvalution(
        fourierBasesValueFunction_semi, b_policy, 5, env, distribution=distribution)
    semigradienttdnevalution.evaluate()
    semi_fourier = plotting.StateValues('fourierBasesValueFunction_semi', fourierBasesValueFunction_semi)

    fourierBasesValueFunction_mc = FourierBasesValueFunction(5)
    gradientmcevalution = GradientMonteCarloEvaluation(fourierBasesValueFunction_mc, b_policy, env, distribution=distribution)
    gradientmcevalution.evaluate()
    mc_fourier = plotting.StateValues('fourierBasesValueFunction_mc', fourierBasesValueFunction_mc)
    plotting.plot_state_value(env, [semi_poly, mc_poly, semi_fourier, mc_fourier])

def test_approximation_control(env):
    action_space =  env.action_space

    