import sys,os
current_dir= os.path.dirname(os.path.realpath(__file__))
work_folder=current_dir[:current_dir.find('algorithm')]
sys.path.append(work_folder)

from test_setup import get_env
from algorithm.value_based.tabular_solution_method.dynamic_programming.policy_iteration  import PolicyIteration 
from algorithm.value_based.tabular_solution_method.dynamic_programming.value_iteration   import ValueIteration

from policy.policy import DiscreteStateValueBasedPolicy


real_env = get_env("GridworldEnv")

def test_policy_iteration(env):
    rl_method = PolicyIteration(env)
    rl_method.learn()
    
test_policy_iteration(real_env)





