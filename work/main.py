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
from algorithm.td_method.test import (test_expected_sarsa_method,
                                      test_qlearning_method, test_sarsa_method,
                                      test_td0_evaluation_method)
from env.blackjack import BlackjackEnv
from env.cliff_walking import CliffWalkingEnv
from env.grid_world import GridworldEnv
from env.grid_world_with_walls_block import GridWorldWithWallsBlockEnv
from env.random_walking import RandomWalkingEnv
from env.windy_gridworld import WindyGridworldEnv
from lib.utility import create_distribution_randomly


def get_env(env):
    if env == 'blackjack':
        return BlackjackEnv()

    if env == 'randomwalking':
        return RandomWalkingEnv()

    if env == 'windygridworld':
        return WindyGridworldEnv()

    if env == 'cliffwalking':
        return CliffWalkingEnv()


def test(algo, env):
    if algo == "TD0_Evalutaion_Method":
        test_td0_evaluation_method(get_env(env))

    if algo == "Sarsa":
        test_sarsa_method(get_env(env))

    if algo == "qlearning":
        test_qlearning_method(get_env(env))

    if algo == 'expectedsarsa':
        test_expected_sarsa_method(get_env(env))


if __name__ == "__main__":
    fire.Fire()
