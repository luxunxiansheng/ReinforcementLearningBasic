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

from tqdm import tqdm
from common import  CriticBase

class ApproximationExpectedSARSACritic(CriticBase):
    def __init__(self,env,estimator,step_size=0.01,discount= 1.0):
        self.env = env 
        self.estimator = estimator
        self.discount = discount
        self.step_size = step_size

    def exploit(self, *args):
        current_state_index = args[0]
        current_action_index = args[1]
        reward = args[2]
        next_state_index = args[3]    
        create_distribution_fn= args[4]
        
        q_values = {}
        for action_index in range(self.env.action_space.n):
            q_values[action_index] = self.estimator.predict(next_state_index,action_index)

        distribution = create_distribution_fn(q_values)

        expected_q_value = 0
        for action_index in range(self.env.action_space.n):
            expected_q_value += distribution[action_index]*q_values[action_index]

        # set the target 
        target = reward + self.discount * expected_q_value

        # SGD fitting
        self.estimator.update(current_state_index, current_action_index, target)

    
    def get_value_function(self):
        return self.estimator



class EpisodicSemiGradientExpectedSarsaControl:
    """
    Expected SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy with approximation of q funciton 
    """

    def __init__(self, critic, actor, env, statistics, episodes,discount=1.0):
        self.env = env
        self.discount = discount
        self.statistics = statistics
        self.episodes = episodes
        self.critic = critic 
        self.actor  = actor 

    def explore(self,*args):
        for episode in tqdm(range(0, self.episodes)):
            self._run_one_episode(episode)


    def _run_one_episode(self, episode):
        # S
        current_state_index = self.env.reset()

        while True:
            # A
            self.actor.explore(current_state_index,self.env.action_space)
            current_action_index = self.actor.get_behavior_policy().get_action(current_state_index)
            
            observation = self.env.step(current_action_index)
            self.env.render()
            # R
            reward = observation[1]
            done = observation[2]

            self.statistics.episode_rewards[episode] += reward
            self.statistics.episode_lengths[episode] += 1

            # S'
            next_state_index = observation[0]

            self.critic.exploit(current_state_index,current_action_index,reward,next_state_index,self.actor.get_create_behavior_policy_fn())

            if done:
                break

            current_state_index = next_state_index





