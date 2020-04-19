from algorithm.dynamic_programming.policy_iteration import PolicyIteration
from algorithm.dynamic_programming.q_value_iteration import QValueIteration
from algorithm.dynamic_programming.v_value_iteration import VValueIteration

from policy.policy import TabularPolicy


def test_policy_iteration(env):
    v_table = env.build_V_table()
    transition_table = env.P
    policy_table = env.build_policy_table()
    table_policy = TabularPolicy(policy_table)

    rl_method = PolicyIteration(v_table, table_policy, transition_table)

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
    rl_method = QValueIteration(q_table, table_policy, transition_table)
    rl_method.improve()
    env.show_policy(table_policy)


def test_v_value_iteration(env):
    v_table = env.build_V_table()
    transition_table = env.P
    policy_table = env.build_policy_table()
    table_policy = TabularPolicy(policy_table)
    rl_method = VValueIteration(table_policy, v_table, transition_table)
    rl_method.improve()
    env.show_policy(table_policy)
