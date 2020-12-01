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

from common import ActorBase
import numpy as np
from tqdm import tqdm


class Actor(ActorBase):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy with approximation of q funciton 
    """

    def __init__(self, estimator, discreteactionpolicy, env, statistics, episodes, step_size=0.1, discount=1.0):
        self.estimator = estimator
        self.policy = discreteactionpolicy
        self.env = env
        self.step_size = step_size
        self.discount = discount
        self.statistics = statistics
        self.episodes = episodes

    def improve(self):
        for episode in tqdm(range(0, self.episodes)):
            self._run_one_episode(episode)
    
    def get_behavior_policy(self):
        return self.policy

    def _run_one_episode(self, episode):
        # S
        current_state = self.env.reset()

        # A
        current_action_index = self.get_behavior_policy().get_action(current_state)

        while True:
            observation = self.env.step(current_action_index)
            # R
            reward = observation[1]
            done = observation[2]

            self.statistics.episode_rewards[episode] += reward
            self.statistics.episode_lengths[episode] += 1

            # S'
            next_state = observation[0]

            # A'
            next_action_index = self.policy.get_action(next_state)

            
            # set the target 
            target = reward + self.discount * self.estimator.predict(next_state, next_action_index)

            # SGD fitting
            self.estimator.update(self.step_size, current_state, current_action_index, target)

            if done:
                break

            current_state = next_state
            current_action_index = next_action_index
    
    def get_optimal_policy(self):
        return self.policy


class EpisodicSemiGradientSarsaControl:
    def __init__(self, estimator, discreteactionpolicy, env, statistics, episodes, step_size=0.1, discount=1.0):
        self.actor= Actor(estimator, discreteactionpolicy, env, statistics, episodes, step_size, discount)
    
    def improve(self):
        self.actor.improve()
        return self.actor.get_optimal_policy()
