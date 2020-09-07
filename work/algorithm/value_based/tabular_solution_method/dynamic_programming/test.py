
import sys
sys.path.append("/home/ornot/workspace/ReinforcementLearningBasic/work")

from test_setup import get_env

from algorithm.value_based.tabular_solution_method.dynamic_programming.policy_iteration  import PolicyIteration
from algorithm.value_based.tabular_solution_method.dynamic_programming.v_value_iteration import VValueIteration
from algorithm.value_based.tabular_solution_method.dynamic_programming.q_value_iteration import QValueIteration

from policy.policy import TabularPolicy

real_env = get_env("grid_world")

def test_policy_iteration(env):
    v_table = env.build_V_table()
    transition_table = env.P
    policy_table = env.build_policy_table()
    table_policy = TabularPolicy(policy_table)
    rl_method = PolicyIteration(v_table, table_policy,transition_table,delta=1e-5,discount=1.0)
    optimal_policy =rl_method.improve()
    env.show_policy(optimal_policy)

test_policy_iteration(real_env)

def test_q_value_iteration(env):
    q_table = env.build_Q_table()
    transition_table = env.P
    
    rl_method = QValueIteration(q_table, transition_table)
    optimal_policy =rl_method.improve()
    env.show_policy(optimal_policy)

test_q_value_iteration(real_env)

def test_v_iteration(env):
    v_table = env.build_V_table()
    transition_table = env.P
    rl_method = VValueIteration(v_table, transition_table)
    optimal_policy =rl_method.improve()
    env.show_policy(optimal_policy)

test_v_iteration(real_env)




