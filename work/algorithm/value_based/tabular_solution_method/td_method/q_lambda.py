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
## Contributor(s):
#
#    Bin.Li (ornot2008@yahoo.com)
##
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

from copy import deepcopy
from tqdm import tqdm


class QLambda:
    """
    Q-Learning algorithm with backward view : Off-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy
    """
    def __init__(self,critic,actor, env, statistics, episodes):
        self.env = env
        self.episodes = episodes
            
        self.statistics = statistics

        self.critic = critic
        self.actor  = actor 

    def improve(self):
        for episode in tqdm(range(0, self.episodes)):
            self._run_one_episode(episode)

    def _run_one_episode(self, episode):
        # S
        current_state_index = self.env.reset()
        current_action_index = self.actor.get_behavior_policy().get_action(current_state_index)

        while True:

            # A
            
            observation = self.env.step(current_action_index)

            # R
            reward = observation[1]
            done = observation[2]

            self.statistics.episode_rewards[episode] += reward
            self.statistics.episode_lengths[episode] += 1

            # S'
            next_state_index = observation[0]

            # A'
            next_action_index = self.actor.get_behavior_policy().get_action(next_state_index) 
        

            self.critic.evaluate(current_state_index,current_action_index,reward,next_state_index)
            self.actor.improve(current_state_index,current_action_index)
        

            if done:
                break

            current_state_index  = next_state_index
            current_action_index = next_action_index
