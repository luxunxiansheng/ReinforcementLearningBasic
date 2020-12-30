import sys,os
current_dir= os.path.dirname(os.path.realpath(__file__))
work_folder=current_dir[:current_dir.find('algorithm')]
sys.path.append(work_folder)

from test_setup import get_env
from algorithm.value_based.tabular_solution_method.dynamic_programming.policy_iteration  import PolicyIteration
from algorithm.value_based.tabular_solution_method.dynamic_programming.value_iteration   import ValueIteration
from policy.policy import DiscreteStateValueBasedPolicy


real_env = get_env("grid_world")

def test_v_iteration(env):
    v_table = env.build_V_table()
    policy_table = env.build_policy_table()
    table_policy = DiscreteStateValueBasedPolicy(policy_table)
    transition_table = env.P
    rl_method = ValueIteration(v_table,transition_table,table_policy)
    optimal_policy =rl_method.improve()
    env.show_policy(optimal_policy)

test_v_iteration(real_env)

def test_policy_iteration(env):
    v_table = env.build_V_table()
    transition_table = env.P
    policy_table = env.build_policy_table()
    table_policy = DiscreteStateValueBasedPolicy(policy_table)
    rl_method = PolicyIteration(v_table, table_policy,transition_table,delta=1e-5,discount=1.0)
    optimal_policy =rl_method.improve()
    env.show_policy(optimal_policy)

test_policy_iteration(real_env)





