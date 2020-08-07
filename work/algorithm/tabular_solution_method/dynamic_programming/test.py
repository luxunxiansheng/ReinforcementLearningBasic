from algorithm.tabular_solution_method.dynamic_programming.policy_iteration import PolicyIteration
from algorithm.tabular_solution_method.dynamic_programming.q_value_iteration import QValueIteration
from algorithm.tabular_solution_method.dynamic_programming.v_value_iteration import VValueIteration

from policy.policy import TabularPolicy


def test_policy_iteration(env):
    v_table = env.build_V_table()
    transition_table = env.P
    policy_table = env.build_policy_table()
    table_policy = TabularPolicy(policy_table)

    rl_method = PolicyIteration(v_table, table_policy, transition_table)

    delta = 1e-5

    env.show_policy(table_policy)

    print("---------------------------------")

    while True:
        current_delta = rl_method.improve()
        if current_delta < delta:
            break
    env.show_policy(table_policy)


def test_q_value_iteration(env):
    q_table = env.build_Q_table()
    transition_table = env.P
    
    rl_method = QValueIteration(q_table, transition_table)
    rl_method.improve()
    env.show_policy(rl_method.get_optimal_policy(env))

def test_v_value_iteration(env):
    v_table = env.build_V_table()
    transition_table = env.P
    
    rl_method = VValueIteration(v_table, transition_table)
    rl_method.improve()
    env.show_policy(rl_method.get_optimal_policy())
