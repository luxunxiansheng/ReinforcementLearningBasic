from algorithm.monte_carlo_method.monte_carlo_es_control_method import Monte_Carlo_ES_Control_Method
from algorithm.monte_carlo_method.monte_carlo_off_policy_evaluation_method import Monte_Carlo_Off_Policy_Evaluation_Method
from algorithm.monte_carlo_method.monte_carlo_on_policy_control_method import Monte_Carlo_On_Policy_Control_Method
from algorithm.monte_carlo_method.v_monte_carlo_evaluation_method import V_Monte_Carlo_Evaluation_Method
from policy.policy import TabularPolicy

from env.blackjack import BlackjackEnv


def test_mc_offpolicy_evaluation_method_for_blackjack():
    env = BlackjackEnv()

    q_table = env.build_Q_table()

    # Random behavior policy
    b_policy_table = env.build_policy_table()
    b_policy = TabularPolicy(b_policy_table)

    # spcific target policy only for blackjack
    t_policy_table = env.build_policy_table()
    for state_index, _ in t_policy_table.items():
        card_sum = state_index[0]
        if card_sum < 20:
            t_policy_table[state_index][BlackjackEnv.HIT] = 1.0
            t_policy_table[state_index][BlackjackEnv.STICK] = 0.0
        else:
            t_policy_table[state_index][BlackjackEnv.HIT] = 0.0
            t_policy_table[state_index][BlackjackEnv.STICK] = 1.0
    t_policy = TabularPolicy(t_policy_table)

    error = {}
    init_state = env.reset(False)
    for episode in range(10000):
        state_value = 0.0
        for _ in range(100):
            rl_method = Monte_Carlo_Off_Policy_Evaluation_Method(
                q_table, b_policy, t_policy, env, episode)
            rl_method.evaluate()
            state_value = state_value + \
                q_table[init_state][BlackjackEnv.HIT]+0.27726
        error[episode] = state_value*state_value/100
        print("{}:{:.3f}".format(episode, error[episode]))


def test_q_mc_es_control_method(env):
    q_table = env.build_Q_table()
    policy_table = env.build_policy_table()
    table_policy = TabularPolicy(policy_table)
    rl_method = Monte_Carlo_ES_Control_Method(q_table, table_policy, env)
    rl_method.improve()
    env.show_policy(table_policy)


def test_mc_onpolicy_control_method(env):
    q_table = env.build_Q_table()
    policy_table = env.build_policy_table()
    table_policy = TabularPolicy(policy_table)
    rl_method = Monte_Carlo_On_Policy_Control_Method(
        q_table, table_policy, 0.1, env)
    rl_method.improve()
    env.render()
    env.show_policy(table_policy)


def test_v_mc_evalution_method(env):
    v_table = env.build_V_table()
    policy_table = env.build_policy_table()

    table_policy = TabularPolicy(policy_table)
    rl_method = V_Monte_Carlo_Evaluation_Method(v_table, table_policy, env)
    table_policy = TabularPolicy(policy_table)
    env.show_policy(table_policy)
    rl_method.evaluate()
    env.show_v_table(v_table)
