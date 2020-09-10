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
from common import ActorBase, CriticBase

from tqdm import tqdm

from lib.utility import create_distribution_epsilon_greedily
from policy.policy import TabularPolicy

class Critic(CriticBase):
    def __init__(self,q_table,env,discount=1.0):
        self.q_value_function = q_table
        self.env= env
        self.discount = discount

    def evaluate(self, *args):
        returns, count = args[0]
        state_index, action_index =args[1]
        self.q_value_function[state_index][action_index]= returns/count
        
    def _init_state_count(self):
        state_count = defaultdict(lambda: {})
        for state_index, action_values in self.q_value_function.items():
            for action_index, _ in action_values.items():
                state_count[state_index][action_index] = (0, 0.0)
        
        return state_count

    def run_one_episode(self,policy):
        trajectory = []
        current_state_index = self.env.reset(False)
        while True:
            action_index = policy.get_action(current_state_index)
            observation = self.env.step(action_index)
            reward = observation[1]
            trajectory.append((current_state_index, action_index, reward))
            done = observation[2]
            if done:
                break
            current_state_index = observation[0]

        return trajectory

class Actor(ActorBase):
    def __init__(self,epsilon) :
        self.create_distribution_epsilon_greedily = create_distribution_epsilon_greedily(epsilon)
    
    def improve(self, *args):
        policy_table={}
        q_value_function = args[0]
        for state_index, action_values in q_value_function.items():
            distibution = self.create_distribution_epsilon_greedily(action_values)
            policy_table[state_index]= distibution
        
        table_policy = TabularPolicy(policy_table)
        return table_policy

class MonteCarloOnPolicyControl:
    def __init__(self, q_value_function, policy, env, episodes=50000, discount=1.0,epsilon=0.01):
        self.q_value_function = q_value_function
        self.env = env
        self.episodes = episodes
        self.discount = discount
        self.policy = policy
        
        self.critic =Critic(q_value_function,env)
        self.actor = Actor(epsilon)

    def improve(self):
        for _ in tqdm(range(0, self.episodes)):
            q_value_function = self.critic.evaluate(self.policy)
            self.policy= self.actor.improve(q_value_function)
            #self.env.show_policy(self.policy)
        return self.policy


    def improve(self):
        for _ in tqdm(range(0,self.episodes)):
            state_count = self._init_state_count()
            trajectory = self._run_one_episode(self.policy)
            R = 0.0
            for state_index, action_index, reward in trajectory[::-1]:
                R = reward+self.discount*R
                state_count[state_index][action_index] = (state_count[state_index][action_index][0] + 1, state_count[state_index][action_index][1] + R)
                self.q_value_function[state_index][action_index] = self.critic.evaluate(state_count[state_index][action_index][1],state_count[state_index][action_index][0])
                self.actor.improve(self.q_value_function,self.policy,state_index)



