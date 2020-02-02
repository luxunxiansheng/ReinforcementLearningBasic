# #### BEGIN LICENSE BLOCK #####
# Version: MPL 1.1/GPL 2.0/LGPL 2.1
#
# The contents of this file are subject to the Mozilla Public License Version
# 1.1 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
# http://www.mozilla.org/MPL/
#
# Software distributed under the License is distributed on an "AS IS" basis,
# WITHOUT WARRANTY OF ANY KIND, either express or implied. See the License
# for the specific language governing rights and limitations under the
# License.
#
#
# Contributor(s):
#
#    Bin.Li (ornot2008@yahoo.com)
#
#
# Alternatively, the contents of this file may be used under the terms of
# either the GNU General Public License Version 2 or later (the "GPL"), or
# the GNU Lesser General Public License Version 2.1 or later (the "LGPL"),
# in which case the provisions of the GPL or the LGPL are applicable instead
# of those above. If you wish to allow use of your version of this file only
# under the terms of either the GPL or the LGPL, and not to allow others to
# use your version of this file under the terms of the MPL, indicate your
# decision by deleting the provisions above and replace them with the notice
# and other provisions required by the GPL or the LGPL. If you do not delete
# the provisions above, a recipient may use your version of this file under
# the terms of any one of the MPL, the GPL or the LGPL.
#
# #### END LICENSE BLOCK #####
#
# /

from agent.dp_agent import DP_Agent
from algorithm.dynamic_programming.policy_iteration_method import Policy_Iteration_Method
from algorithm.dynamic_programming.q_value_iteration_method import Q_Value_Iteration_Method
from algorithm.dynamic_programming.v_value_iteration_method import V_Value_Iteration_Method
from algorithm.monte_carlo_method.q_monte_carlo_method import Q_Monte_Carlo_Method
from algorithm.monte_carlo_method.v_monte_carlo_method import V_Monte_Carlo_Method
from env.blackjack import BlackjackEnv
from env.grid_world import GridworldEnv


from lib.utility import create_distribution_randomly
from policy.policy import TabularPolicy


def main():
    env = GridworldEnv()

    # test_q_mc_methond(env)
    # test_v_mc_methond(env)

    # test_q_value_iteration(env)
    # test_v_value_iteration(env)
    test_policy_iteration(env)


def test_policy_iteration(env):
    v_table = env.build_V_table()
    transition_table = env.P
    policy_table = env.build_policy_table()

    for state_index, action_probablities in policy_table.items():
        distribution = create_distribution_randomly()(action_probablities)
        policy_table[state_index] = distribution

    table_policy = TabularPolicy(policy_table)
    rl_method = Policy_Iteration_Method(v_table, table_policy, transition_table)

    delta = 1e-5
    while True:

        env.show_policy(table_policy)
        rl_method.evaluate()

        current_delta = rl_method.improve()

        if current_delta < delta:
            break


def test_q_mc_methond(env):
    q_table = env.build_Q_table()

    policy_table = env.build_policy_table()

    for state_index, _ in policy_table.items():
        card_sum = state_index[0]
        if card_sum < 20:
            policy_table[state_index][BlackjackEnv.HIT] = 1.0
            policy_table[state_index][BlackjackEnv.STICK] = 0.0
        else:
            policy_table[state_index][BlackjackEnv.HIT] = 0.0
            policy_table[state_index][BlackjackEnv.STICK] = 1.0

    rl_method = Q_Monte_Carlo_Method(q_table, env)
    table_policy = TabularPolicy(policy_table, rl_method)

    for _ in range(100):

        table_policy.evaluate()
        table_policy.improve()

    env.show_policy(table_policy)


def test_v_mc_methond(env):
    v_table = env.build_V_table()

    policy_table = env.build_policy_table()

    for state_index, _ in policy_table.items():
        card_sum = state_index[0]
        if card_sum < 20:
            policy_table[state_index][BlackjackEnv.HIT] = 1.0
            policy_table[state_index][BlackjackEnv.STICK] = 0.0
        else:
            policy_table[state_index][BlackjackEnv.HIT] = 0.0
            policy_table[state_index][BlackjackEnv.STICK] = 1.0

    rl_method = V_Monte_Carlo_Method(v_table, env)
    table_policy = TabularPolicy(policy_table, rl_method)

    env.show_policy(table_policy)

    table_policy.evaluate()

    env.show_v_table(v_table)


def test_q_value_iteration(env):
    q_table = env.build_Q_table()
    transition_table = env.P
    policy_table = env.build_policy_table()

    rl_method = Q_Value_Iteration_Method(q_table, transition_table)

    table_policy = TabularPolicy(policy_table, rl_method)
    table_policy.improve()

    env.show_policy(table_policy)


def test_v_value_iteration(env):
    v_table = env.build_V_table()
    transition_table = env.P
    policy_table = env.build_policy_table()

    rl_method = V_Value_Iteration_Method(v_table, transition_table)

    table_policy = TabularPolicy(policy_table, rl_method)
    table_policy.improve()

    env.show_policy(table_policy)


def test_dp_policy_iteration(env):

    policy = TabularPolicy()
    policy_iteration_method = Policy_Iteration_Method(env, p)

    dp_agent = DP_Agent(env)


if __name__ == "__main__":
    main()
