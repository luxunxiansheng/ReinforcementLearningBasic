import numpy as np 

from algorithm.approximate_solution_method.value_function import (PolynomialBasesValueFunction, FourierBasesValueFunction)
from algorithm.approximate_solution_method.gradient_monte_carlo_evaluation import GradientMonteCarloEvaluation
from policy.policy import TabularPolicy

def test_gradient_mc_evalution(env):
    b_policy_table = env.build_policy_table()

    # build random policy 
    b_policy = TabularPolicy(b_policy_table)
    
    polynomialbasesfunc =  PolynomialBasesValueFunction(5)
    
    gradientmcevalution =  GradientMonteCarloEvaluation(polynomialbasesfunc, b_policy, env,distribution=np.zeros(env.nS))

    gradientmcevalution.evaluate()

