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

import torch
from torch import nn
from common import CriticBase, PolicyEstimator, QValueEstimator


class DeepQValueEstimator(QValueEstimator):

    def __init__(self,model,learning_rate,device):
        self.model = model
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = nn.optim.Adam(self.model.parameters(), learning_rate)


    def predict(self, state, action=None):
        assert action is None
        return self.model(state.to(self.device), action.to(self.device))
    
    def update(self, *args):
        q_values = args[0].to(self.device)
        target_values = args[1].to(self.device)
        done = args[2]
        episode = args[3]
        writer = args[4]

        loss = self.criterion(q_values,target_values)

        if done:
            writer.add_scalar('loss', loss, episode)

        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()
        
        
class DeepQLearningCritic(CriticBase):
    def __init__(self,policy_estimator,target_estimator,action_space,discount):
        self.policy_estimator = policy_estimator
        self.target_estimator = target_estimator
        self.action_space = action_space
        self.discount = discount 

    def evaluate(self,*args):
        samples = args[0]
        done = args[1]
        episode = args[2]
        writer = args[3]
        batch_size = len(samples)
        
        q_values = torch.zeros(batch_size, self.action_space)
        target_values = torch.zeros(batch_size, self.action_space)

        for sample_index in range(0, batch_size):
            state = samples[sample_index][0]
            action = samples[sample_index][1]
            reward = samples[sample_index][2]
            next_state = samples[sample_index][3]
            terminal = samples[sample_index][4]
        
            the_optimal_q_value_of_next_state = torch.max(self.target_estimator.predict(next_state,None).detach())

            target_values[sample_index][int(action)] = reward if terminal else reward + self.discount*the_optimal_q_value_of_next_state
            
            q_values[sample_index][int(action)] = self.policy_estimator.predict(state,int(action))
        
        self.policy_estimator.update(q_values,target_values,done,episode,writer)

    def sync_target_model_with_policy_model(self):
        self.target_estimator.model.load_state_dict(self.policy_estimator.model.state_dict())
        self.target_estimator.model.eval()
    
    def get_optimal_policy(self):
        pass 
    
    def get_value_function(self):
        return self.policy_estimator


class DeepPolicyEstimator(PolicyEstimator):