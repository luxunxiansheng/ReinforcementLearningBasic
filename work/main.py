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

import math

from tqdm import tqdm

import fire

from agent.dp_agent import DP_Agent
from algorithm.dynamic_programming.policy_iteration_method import Policy_Iteration_Method
from algorithm.dynamic_programming.q_value_iteration_method import Q_Value_Iteration_Method
from algorithm.dynamic_programming.v_value_iteration_method import V_Value_Iteration_Method
from algorithm.monte_carlo_method.monte_carlo_es_control_method import Monte_Carlo_ES_Control_Method
from algorithm.monte_carlo_method.monte_carlo_on_policy_control_method import Monte_Carlo_On_Policy_Control_Method
from algorithm.monte_carlo_method.v_monte_carlo_evaluation_method import V_Monte_Carlo_Evaluation_Method
from algorithm.monte_carlo_method.monte_carlo_off_policy_evaluation_method import Monte_Carlo_Off_Policy_Evaluation_Method
from algorithm.td_method.td0_evaluation_method import TD0_Evalutaion_Method
from env.blackjack import BlackjackEnv
from env.grid_world import GridworldEnv
from env.grid_world_with_walls_block import GridWorldWithWallsBlockEnv
from env.random_walking import RandomWalkingEnv


from lib.utility import create_distribution_randomly
from policy.policy import TabularPolicy


def get_env(env):
    if env == 'blackjack':
        return BlackjackEnv()

    if env == 'randomwalking':
        return RandomWalkingEnv()




def test_td0_evaluation_method(env):
    environment = get_env(env)

    v_table = environment.build_V_table()

    b_policy_table = environment.build_policy_table()
    b_policy = TabularPolicy(b_policy_table)

    td0_method = TD0_Evalutaion_Method(v_table,b_policy,environment)

    td0_method.evaluate()

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


def test_policy_iteration(en):
    
    env = get_env(en)

    v_table = env.build_V_table()
    transition_table = env.P
    policy_table = env.build_policy_table()
    table_policy = TabularPolicy(policy_table)

    rl_method = Policy_Iteration_Method(
        v_table, table_policy, transition_table)

    delta = 1e-5

    env.show_policy(table_policy)

    while True:
        rl_method.evaluate()
        current_delta = rl_method.improve()
        if current_delta < delta:
            break
    env.show_policy(table_policy)


def test_q_mc_es_control_method(en):

    env = get_env(en)

    q_table = env.build_Q_table()
    policy_table = env.build_policy_table()
    table_policy = TabularPolicy(policy_table)
    rl_method = Monte_Carlo_ES_Control_Method(q_table, table_policy, env)
    rl_method.improve()
    env.show_policy(table_policy)


def test_mc_onpolicy_control_method(en):
    env = get_env(en)
    q_table = env.build_Q_table()
    policy_table = env.build_policy_table()
    table_policy = TabularPolicy(policy_table)
    rl_method = Monte_Carlo_On_Policy_Control_Method(q_table, table_policy, 0.1, env)
    rl_method.improve()
    env.show_policy(table_policy)


def test_v_mc_evalution_method(en):
    env = get_env(en)
    v_table = env.build_V_table()
    policy_table = env.build_policy_table()

    table_policy = TabularPolicy(policy_table)
    rl_method = V_Monte_Carlo_Evaluation_Method(v_table, table_policy, env)
    table_policy = TabularPolicy(policy_table)
    env.show_policy(table_policy)
    rl_method.evaluate()
    env.show_v_table(v_table)


def test_q_value_iteration(en):
    env = get_env(en)
    q_table = env.build_Q_table()
    transition_table = env.P
    policy_table = env.build_policy_table()
    table_policy = TabularPolicy(policy_table)
    rl_method = Q_Value_Iteration_Method(
        q_table, table_policy, transition_table)
    rl_method.improve()
    env.show_policy(table_policy)


def test_v_value_iteration(en):
    env = get_env(en)
    v_table = env.build_V_table()
    transition_table = env.P
    policy_table = env.build_policy_table()
    table_policy = TabularPolicy(policy_table)
    rl_method = V_Value_Iteration_Method(
        table_policy, v_table, transition_table)
    rl_method.improve()
    env.show_policy(table_policy)


if __name__ == "__main__":
    fire.Fire()
