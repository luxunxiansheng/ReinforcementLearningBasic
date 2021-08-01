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
from collections import namedtuple

import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from common import (ActorBase, Agent, CriticBase,ExplorerBase)
from algorithm.policy_based.actor_critic.actor_critic_common import (DeepPolicyEsitmator, DeepValueEstimator, ParameterizedPolicy)


class BatchCritic(CriticBase):
    def __init__(self,value_estimator,discount=1.0):
        self.estimator = value_estimator
        self.discount  = discount
    
    def evaluate(self,*args):
        trajectory = args[0]
        episode = args[1]
        writer = args[2]

        state_values=[]
        returns = []
        value_losses=[]

        G = 0.0 
        
        # reduce the variance  
        for _,state_value,_,_,_,reward in trajectory[::-1]:
            G = reward + self.discount*G
            state_values.insert(0,state_value)
            returns.insert(0,G)

        returns = torch.tensor(returns)
        returns = (returns-returns.mean())/(returns.std()+BatchActor.EPS)

        for value, G in zip(state_values,returns):
            value_losses.append(F.smooth_l1_loss(value,torch.tensor([G])))

        total_loss = torch.stack(value_losses).sum()
        writer.add_scalar('value_loss',total_loss,episode)
        self.estimator.update(total_loss)
    
    def get_value_function(self):
        return self.estimator

    def get_optimal_policy(self):
        pass 

        
class PolicyGridentExplorer(ExplorerBase):
    ENTROY_BETA = 0.0
    
    def __init__(self,policy,discount=1.0):
        self.policy = policy 
        self.discount = discount
        
    def explore(self,*args):
        trajectory = args[0]
        episode = args[1]
        writer  = args[2]
    
        G = 0.0
        log_action_probs=[]
        entroys=[]
        state_values=[]
        returns = []
        policy_losses=[]

        # reduce the variance  
        for _,state_value,_,action_prob,entroy,reward in trajectory[::-1]:
            # estimate the sate value with Monte Carlo target   
            G = reward + self.discount*G
            log_action_probs.insert(0,torch.log(action_prob))
            entroys.insert(0,entroy)
            state_values.insert(0,state_value)
            returns.insert(0,G)
        returns = torch.tensor(returns)
        returns = (returns-returns.mean())/(returns.std()+BatchActor.EPS)

        for log_action_prob,state_value, G in zip(log_action_probs,state_values,returns):
            advantage = G - state_value.detach()
            policy_losses.append(-log_action_prob*advantage)
        
        total_policy_loss = torch.stack(policy_losses).sum()
        total_loss =  total_policy_loss - PolicyGridentExplorer.ENTROY_BETA*torch.stack(entroys).sum()
        writer.add_scalar('policy_loss',total_policy_loss,episode)
        self.policy.policy_estimator.update(total_loss)

    def get_behavior_policy(self):
        return self.policy
    
class BatchActor(ActorBase):
    EPS = np.finfo(np.float32).eps.item()
    MAX_STEPS = 500000
    def  __init__(self,env,critic,explorer,writer):
        self.env = env
        self.critic=critic 
        self.explorer= explorer
        self.writer = writer
            
    def act(self,*args):
        episode=args[0]
        trajectory = self._run_one_episode(episode)    
        if len(trajectory)< BatchActor.MAX_STEPS:
            self.critic.evaluate(trajectory,episode,self.writer)
            self.explorer.explore(trajectory,episode,self.writer)
        
    def _run_one_episode(self,episode):
        trajectory = []
        current_state = self.env.reset()
        
        for step in range(0,BatchActor.MAX_STEPS):
            action_index = self.explorer.get_behavior_policy().get_action(current_state)
            distribution = self.explorer.get_behavior_policy().get_discrete_distribution_tensor(current_state)
            entropy = torch.distributions.Categorical(distribution).entropy()
            action_prob = distribution[action_index]
            state_value  = self.critic.get_value_function().predict(current_state)
            observation = self.env.step(action_index)
            self.env.render()
            reward = observation[1]
            trajectory.append((current_state,state_value,action_index,action_prob,entropy,reward))
            done = observation[2]
            if done:
                self.writer.add_scalar('steps_to_go',step,episode)
                break
            current_state = observation[0]
        return trajectory


class BatchActorCriticAgent(Agent):
    def __init__(self,env,num_episodes):
        self.env = env 
        
        value_esitimator = DeepValueEstimator(self.env.observation_space.shape[0])
        self.critic=BatchCritic(value_esitimator)

        policy_estimator = DeepPolicyEsitmator(self.env.observation_space.shape[0],self.env.action_space.n)
        policy= ParameterizedPolicy(policy_estimator)
        self.explorer= PolicyGridentExplorer(policy)
        
        self.writer = SummaryWriter()
        self.actor = BatchActor(self.env,self.critic,self.explorer,self.writer)

        self.num_episodes = num_episodes
    
    def learn(self):
        for episode in tqdm(range(0, self.num_episodes)):
            self.actor.act(episode)
