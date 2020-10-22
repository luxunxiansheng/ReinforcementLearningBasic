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


from collections import defaultdict
from tqdm import tqdm
from lib.utility import create_distribution_greedily,create_distribution_epsilon_greedily
from common import ActorBase,CriticBase
from policy.policy import DiscreteStateValueBasedPolicy

class Critic(CriticBase):
    def __init__(self, q_value_function):
        self.state_count=self._init_state_count()
        self.q_value_function=q_value_function

    def evaluate(self,*args):
        state_index= args[0]
        action_index = args[1]
        R= args[2]

        self.state_count[state_index][action_index] = (self.state_count[state_index][action_index][0] + 1, self.state_count[state_index][action_index][1] + R)
        self.q_value_function[state_index][action_index] = self.state_count[state_index][action_index][1] / self.state_count[state_index][action_index][0]
    
    def get_value_function(self):
        return self.q_value_function
    
    def _init_state_count(self):
        state_count = defaultdict(lambda: {})
        for state_index, action_values in self.q_value_function.items():
            for action_index, _ in action_values.items():
                state_count[state_index][action_index] = (0, 0.0)
        return state_count


class Actor(ActorBase):
    def __init__(self,policy,critic):
        self.policy = policy
        self.critic = critic
        self.create_distribution_greedily = create_distribution_greedily()
    
    def improve(self,*args): 
        state_index = args[0]
        q_value_function = self.critic.get_value_function()
        greedy_distibution = self.create_distribution_greedily(q_value_function[state_index])
        self.policy.policy_table[state_index] = greedy_distibution       
    
    def get_optimal_policy(self):
        return self.policy


class MonteCarloESControl:
    def __init__(self, q_value_function, policy, env,episodes=10000, discount=1.0):
        self.env = env
        self.policy = policy
        self.episodes = episodes
        self.discount = discount
        self.critic = Critic(q_value_function) 
        self.actor = Actor(policy,self.critic)

    def improve(self, *args):
        for _ in tqdm(range(0, self.episodes)):
            trajectory = self._run_one_episode()
            R = 0.0
            for state_index, action_index, reward in trajectory[::-1]:
                R = reward+self.discount*R
                self.critic.evaluate(state_index,action_index,R)
                self.actor.improve(state_index)
                
        return self.actor.get_optimal_policy()

    def _run_one_episode(self):
        trajectory = []
        current_state_index = self.env.reset()
        while True:
            action_index = self.policy.get_action(current_state_index)
            observation = self.env.step(action_index)
            reward = observation[1]
            trajectory.append((current_state_index, action_index, reward))
            done = observation[2]
            if done:
                break
            current_state_index = observation[0]
        return trajectory
    
    def _init_state_count(self):
        state_count = defaultdict(lambda: {})
        for state_index, action_values in self.q_value_function.items():
            for action_index, _ in action_values.items():
                state_count[state_index][action_index] = (0, 0.0)
        return state_count
    
