from algorithm.td_method.sarsa import SARSA
from algorithm.td_method.td0_evaluation_method import TD0_Evalutaion_Method
from policy.policy import TabularPolicy


def test_td0_evaluation_method(env):

    v_table = env.build_V_table()
    b_policy_table = env.build_policy_table()
    b_policy = TabularPolicy(b_policy_table)
    td0_method = TD0_Evalutaion_Method(v_table, b_policy, env)
    td0_method.evaluate()


def test_sarsa_method(env):
    q_table = env.build_Q_table()
    b_policy_table = env.build_policy_table()
    b_policy = TabularPolicy(b_policy_table)
    sarsa_method = SARSA(q_table, b_policy, 0.1, env)
    sarsa_method.improve()
