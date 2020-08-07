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

import fire

""" from algorithm.approximate_solution_method.test import (
    test_approximation_control_method, test_approximation_evaluation) """

from algorithm.tabular_solution_method.dynamic_programming.test import (test_policy_iteration, test_q_value_iteration, test_v_value_iteration)

""" from algorithm.tabular_solution_method.td_method.test import (test_double_q_learning_method, test_expected_sarsa_method,
    test_n_setps_expected_sarsa, test_n_steps_sarsa_method,
    test_off_policy_n_steps_sarsa, test_qlearning_method, test_sarsa_method,
    test_td0_evaluation_method, test_td_control_method,
    test_td_lambda_evalution_method, test_tdn_evaluaiton_method) """
from env.blackjack import BlackjackEnv
from env.cliff_walking import CliffWalkingEnv
from env.grid_world import GridworldEnv
from env.grid_world_with_walls_block import GridWorldWithWallsBlockEnv
from env.mountain_car import MountainCarEnv
from env.random_walking import RandomWalkingEnv
from env.windy_gridworld import WindyGridworldEnv


def get_env(env):
    if env == 'grid_world':
        return GridworldEnv()

    if env == 'blackjack':
        return BlackjackEnv()

    if env == 'randomwalking':
        return RandomWalkingEnv()

    if env == 'windygridworld':
        return WindyGridworldEnv()

    if env == 'cliffwalking':
        return CliffWalkingEnv()

    if env == 'mountaincar':
        return MountainCarEnv()


def test_dp(algo,env):
    real_env = get_env(env)

    if algo == "policy_iteration":
        test_policy_iteration(real_env)

    if algo == "v_value_iteration":
        test_v_value_iteration(env)
    
    if algo == "q_value_iteration":
        test_q_value_iteration(env)


""" def test_mc(algo,env):
    pass 


def test_td(algo, env):
    real_env = get_env(env)

    if algo == "TD0_evalutaion":
        test_td0_evaluation_method(real_env)

    if algo == "Sarsa":
        test_sarsa_method(real_env)

    if algo == "Qlearning":
        test_qlearning_method(real_env)

    if algo == 'Expectedsarsa':
        test_expected_sarsa_method(real_env)
    
    if algo == 'Doubleqlearning':
        test_double_q_learning_method(real_env)
    
    if algo == 'TDN_evalutaion':
        test_tdn_evaluaiton_method(real_env)

    if algo == 'TDLambda_evalutaion':
        test_td_lambda_evalution_method(real_env)
    
    if algo == 'N_step_sarsa':
        test_n_steps_sarsa_method(real_env)
    
    if algo == 'N_steps_expected_sarsa':
        test_n_setps_expected_sarsa(real_env)
        
    if algo == 'N_steps_off_policy_sarsa':
        test_off_policy_n_steps_sarsa(real_env)
    
    if algo == 'td_control_method':
        test_td_control_method(real_env)
    
    if algo == 'approximation_evalution':
        test_approximation_evaluation(real_env)
    
    if algo == 'approximation_control':
        test_approximation_control_method(real_env) """


if __name__ == "__main__":
    fire.Fire()
