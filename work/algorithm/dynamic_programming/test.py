from algorithm.dynamic_programming.policy_iteration import Policy_Iteration
from algorithm.dynamic_programming.q_value_iteration import Q_Value_Iteration
from algorithm.dynamic_programming.v_value_iteration import V_Value_Iteration

from policy.policy import TabularPolicy


def test_policy_iteration(env):
    v_table = env.build_V_table()
    transition_table = env.P
    policy_table = env.build_policy_table()
    table_policy = TabularPolicy(policy_table)

    rl_method = Policy_Iteration(v_table, table_policy, transition_table)

    delta = 1e-5

    env.show_policy(table_policy)

    while True:
        rl_method.evaluate()
        current_delta = rl_method.improve()
        if current_delta < delta:
            break
    env.show_policy(table_policy)


def test_q_value_iteration(env):
    q_table = env.build_Q_table()
    transition_table = env.P
    policy_table = env.build_policy_table()
    table_policy = TabularPolicy(policy_table)
    rl_method = Q_Value_Iteration(q_table, table_policy, transition_table)
    rl_method.improve()
    env.show_policy(table_policy)


def test_v_value_iteration(env):
    v_table = env.build_V_table()
    transition_table = env.P
    policy_table = env.build_policy_table()
    table_policy = TabularPolicy(policy_table)
    rl_method = V_Value_Iteration(table_policy, v_table, transition_table)
    rl_method.improve()
    env.show_policy(table_policy)
