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
from algorithm.policy_based.actor_critic.actor_critic_common import DeepPolicyEsitmator, DeepValueEstimator, ParameterizedPolicy
from collections import namedtuple
import numpy as np 
from tqdm import tqdm

import torch

import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from common import ActorBase, Agent, ImproverBase,CriticBase


class OnlineCritic(CriticBase):
    def __init__(self,value_estimator,discount=1.0):
        self.estimator = value_estimator
        self.discount  = discount
    
    def evaluate(self,*args):
        current_state_index = args[0]
        reward = args[1]
        next_state_index = args[2]    
        done = args[3]
        episode =args[4]
        writer = args[5]
        
        value_of_next_state = self.estimator.predict(next_state_index)

        # set the target 
        target = reward + self.discount * value_of_next_state
        input  = self.estimator.predict(current_state_index)
        loss  = F.smooth_l1_loss(input,target)

        if done:
            writer.add_scalar('value_loss',loss,episode)
        
        self.estimator.update(loss)
    
    def get_value_function(self):
        return self.estimator
    
    def get_greedy_policy(self):
        pass 
        
class PolicyGridentExplorer(ImproverBase):
    def __init__(self,policy,critic,discount=1.0):
        self.policy = policy 
        self.discount = discount
        self.critic = critic
        
    def explore(self,*args):
        current_state_index = args[0]
        current_action_index = args[1]
        reward = args[2]
        next_state_index = args[3]
        done = args[4]
        episode =args[5]
        writer = args[6]
        
        action_prob=self.policy.get_discrete_distribution_tensor(current_state_index)[current_action_index]
        advantage = torch.tensor(reward) + self.discount* self.critic.estimator.predict(next_state_index) - self.critic.estimator.predict(current_state_index)
        policy_loss = -torch.log(torch.round(action_prob*10**3)/10**3)*advantage.detach()
    
        if done:
            writer.add_scalar('policy_loss',policy_loss.item(),episode)

        self.policy.policy_estimator.update(policy_loss)

    def get_target_policy(self):
        return self.policy

class OnlineActor(ActorBase):
    EPS = np.finfo(np.float32).eps.item()
    MAX_STEPS = 500000
    
    def __init__(self,env,critic,explorer,writer):
        self.env = env
        self.critic=critic 
        self.explorer= explorer
        self.writer = writer


    def act(self, *args):
        episode=args[0]
        # S
        current_state_index = self.env.reset()

        for _ in range(0, OnlineActor.MAX_STEPS):
            # A
            current_action_index = self.explorer.get_behavior_policy().get_action(current_state_index)
            observation = self.env.step(current_action_index)
            self.env.render()

            # R
            reward = observation[1]
            done = observation[2]

            # S'
            next_state_index = observation[0]

            self.critic.evaluate(current_state_index,reward,next_state_index,done,episode,self.writer)
            
            self.explorer.explore(current_state_index,current_action_index,reward,next_state_index,done,episode,self.writer)

            if done:
                break

            current_state_index = next_state_index        


class OnlineCriticActorAgent(Agent):

    def  __init__(self,env,num_episodes):
        self.env = env 
        
        value_esitimator = DeepValueEstimator(self.env.observation_space.shape[0])
        self.critic=OnlineCritic(value_esitimator)

        policy_estimator = DeepPolicyEsitmator(self.env.observation_space.shape[0],self.env.action_space.n)
        policy= ParameterizedPolicy(policy_estimator)
        self.explorer= PolicyGridentExplorer(policy,self.critic)
        
        self.writer = SummaryWriter()
        self.actor = OnlineActor(self.env,self.critic,self.explorer,self.writer)

        self.num_episodes = num_episodes

            
    def learn(self):
        for episode in tqdm(range(0, self.num_episodes)):
            self.actor.act(episode)



