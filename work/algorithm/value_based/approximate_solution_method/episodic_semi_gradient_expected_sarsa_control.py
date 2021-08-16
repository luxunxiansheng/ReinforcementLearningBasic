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
from common import CriticBase
from algorithm.value_based.approximate_solution_method.actor import Actor
from algorithm.value_based.approximate_solution_method.explorer import ESoftExplorer
from algorithm.value_based.approximate_solution_method.continuous_state_value_table_policy import ContinuousStateValueTablePolicy

class ApproximationExpectedSARSACritic(CriticBase):
    def __init__(self,env,estimator,policy,step_size=0.01,discount= 1.0):
        self.env = env 
        self.estimator = estimator
        self.discount = discount
        self.policy = policy
        self.step_size = step_size

    def evaluate(self, *args):
        current_state_index = args[0]
        current_action_index = args[1]
        reward = args[2]
        next_state_index = args[3]    
        
        q_values = {}
        for action_index in range(self.env.action_space.n):
            q_values[action_index] = self.estimator.predict(next_state_index,action_index)

        next_state_distribution = self.policy.get_discrete_distribution(next_state_index)

        expected_q_value = 0
        for action_index in range(self.env.action_space.n):
            expected_q_value += next_state_distribution[action_index]*q_values[action_index]

        # set the target 
        target = reward + self.discount * expected_q_value

        # SGD fitting
        self.estimator.update(current_state_index, current_action_index, target)
    
    def get_value_function(self):
        return self.estimator

    def get_optimal_policy(self):
        pass 


class EpisodicSemiGradientExpectedSarsaControl:
    """
    Expected SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy with approximation of q funciton 
    """

    def __init__(self,env, estimator,statistics, episodes):
        self.env = env
        self.episodes = episodes

        policy =       ContinuousStateValueTablePolicy(estimator,self.env.action_space.n)
        self.critic =  ApproximationExpectedSARSACritic(env,estimator,policy) 
        explorer    =  ESoftExplorer(policy)
        self.actor  =  Actor(env,self.critic,explorer,statistics)

    def learn(self):
        for episode in tqdm(range(0, self.episodes)):
            self.actor.act(episode)