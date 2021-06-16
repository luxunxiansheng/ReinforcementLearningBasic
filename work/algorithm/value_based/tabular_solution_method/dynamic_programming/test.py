import sys,os
current_dir= os.path.dirname(os.path.realpath(__file__))
work_folder=current_dir[:current_dir.find('algorithm')]
sys.path.append(work_folder)

from test_setup import get_env
from algorithm.value_based.tabular_solution_method.dynamic_programming.policy_iteration  import PolicyIteration ,PolicyIterationActor, PoplicyIterationCritic
from algorithm.value_based.tabular_solution_method.dynamic_programming.value_iteration   import ValueIteration, ValueIterationActor

from policy.policy import DiscreteStateValueBasedPolicy


real_env = get_env("grid_world")

def test_v_iteration(env):
    v_table = env.build_V_table()
    policy_table = env.build_policy_table()
    table_policy = DiscreteStateValueBasedPolicy(policy_table)
    transition_table = env.P

    actor= ValueIterationActor(v_table, table_policy,transition_table, 1e-5, 1.0)
    rl_method = ValueIteration(actor)
    optimal_policy =rl_method.explore()
    env.show_policy(optimal_policy)

test_v_iteration(real_env)

def test_policy_iteration(env):
    v_table = env.build_V_table()
    transition_table = env.P
    policy_table = env.build_policy_table()
    table_policy = DiscreteStateValueBasedPolicy(policy_table)

    critic = PoplicyIterationCritic(table_policy,v_table,transition_table,1e-5,1.0)
    actor  = PolicyIterationActor(table_policy,critic,transition_table,1e-5,1.0)

    rl_method = PolicyIteration(critic,actor)
    optimal_policy =rl_method.explore()
    env.show_policy(optimal_policy)

test_policy_iteration(real_env)





