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

import numpy as np

import ray
from ray.worker import remote
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from common import ActorBase, Agent, ImproverBase, CriticBase, PolicyEstimator
from algorithm.policy_based.actor_critic.actor_critic_common import ParameterizedPolicy, ValueEstimator
from algorithm.value_based.approximate_solution_method.remoe_env_setup import get_env

import env_setup

class ValueModel(nn.Module):
    def __init__(self,observation_space_size):
        super().__init__()
        self.affine = nn.Linear(observation_space_size, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.value_head = nn.Linear(128, 1)

    def forward(self, state):
        state = torch.from_numpy(state).float()
        state = F.relu(self.dropout(self.affine(state)))
        state_value = self.value_head(state)
        return state_value
    
    def get_weights(self):
        return self.state_dict()
    
    def set_weights(self, weights):
        self.load_state_dict(weights)
    
    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = torch.from_numpy(g)

                
class LocalValueEestimator(ValueEstimator):
    def __init__(self,observation_space_size):
        self.model = ValueModel(observation_space_size)
                
    def predict(self,state):
        value = self.model.forward(state)
        return value
    
    def update(self,*args):
        pass 
    
        
class GlobalValueEestimator(ValueEstimator):
    def __init__(self,observation_space_size):
        self.model = ValueModel(observation_space_size)
        self.optimizer = optim.Adam(self.model.parameters(),lr =1e-3)
        
    def predict(self, state):
        pass      
    
    def update(self, *gradients):
        summed_gradients = [
            np.stack(gradient_zip).sum(axis=0)
            for gradient_zip in zip(*gradients)
        ]

        self.optimizer.zero_grad()
        self.model.set_gradients(summed_gradients)
        self.optimizer.step()

        # return self.model.get_weights()


class LocalBatchCritic(CriticBase):
    EPS = np.finfo(np.float32).eps.item()
    def __init__(self,local_value_estimator,discount=1.0):
        self.value_estimator =  local_value_estimator
        self.discount  = discount
    
    def evaluate(self,*args): 
        trajectory = args[0]
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
        returns = (returns-returns.mean())/(returns.std()+LocalBatchCritic.EPS)

        for value, G in zip(state_values,returns):
            value_losses.append(F.smooth_l1_loss(value,torch.tensor([G])))
        
        total_loss = torch.stack(value_losses).sum()
        
        self.value_estimator.model.zero_grad()
        total_loss.backward()

    def get_value_function(self):
        return self.value_estimator
    
    def get_greedy_policy(self):
        pass 
    
class PolicyModel(nn.Module):
    def __init__(self,observation_space_size,action_space_size):
        super().__init__()
        self.affine = nn.Linear(observation_space_size, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.action_head = nn.Linear(128, action_space_size)
            
    def forward(self, state):
        state = torch.from_numpy(state).float()
        state = F.relu(self.dropout(self.affine(state)))
        actions_prob = F.softmax(self.action_head(state),dim=0)
        return actions_prob
    
    def get_weights(self):
        return self.state_dict()
    
    def set_weights(self, weights):
        self.load_state_dict(weights)
    
    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = torch.from_numpy(g)

class LocalPolicyEsitmator(PolicyEstimator):
    def __init__(self,observation_space_size,action_space_size):
        self.model = PolicyModel(observation_space_size,action_space_size)
    
    def predict(self,state):
        actions_prob=self.model.forward(state)
        return actions_prob
    
    def update(self,*args):
        pass

class GlobalPolicyEsitmator(PolicyEstimator):
    def __init__(self,observation_space_size,action_space_size):
        self.model = PolicyModel(observation_space_size,action_space_size)
        self.optimizer = optim.Adam(self.model.parameters(),lr =1e-3)

    def predict(self, state):
        actions_prob=self.model.forward(state)
        return actions_prob

    def update(self, *gradients):
        summed_gradients = [
            np.stack(gradient_zip).sum(axis=0)
            for gradient_zip in zip(*gradients)
        ]

        self.optimizer.zero_grad()
        self.model.set_gradients(summed_gradients)
        self.optimizer.step()
    
        # return self.model.get_weights()



class LocalBatchPolicyGridentExplorer(ImproverBase):
    EPS = np.finfo(np.float32).eps.item()
    ENTROY_BETA = 0.0
    
    def __init__(self,local_policy,discount=1.0):
        self.policy = local_policy 
        self.discount = discount

    def explore(self,*args):
        trajectory = args[0]
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
        returns = (returns-returns.mean())/(returns.std()+LocalBatchPolicyGridentExplorer.EPS)

        for log_action_prob,state_value, G in zip(log_action_probs,state_values,returns):
            advantage = G - state_value.detach()
            policy_losses.append(-log_action_prob*advantage)
                
        total_policy_loss = torch.stack(policy_losses).sum()
        total_loss =  total_policy_loss - LocalBatchPolicyGridentExplorer.ENTROY_BETA*torch.stack(entroys).sum()
        
        self.policy.policy_estimator.model.zero_grad()
        # calculate the grident for the model of the local policy 
        total_loss.backward()
    
    def get_target_policy(self):
        return self.policy


@ray.remote
class LocalActor(ActorBase):
    EPS = np.finfo(np.float32).eps.item()
    MAX_STEPS = 500000
    
    def  __init__(self,env_class_name,id):
        self.id = id 
        
        #ray.util.pdb.set_trace()

        klass = getattr(env_setup, env_class_name)
        self.env = klass()
        
        value_esitimator = LocalValueEestimator(self.env.observation_space.shape[0])
        self.local_critic=LocalBatchCritic(value_esitimator)

        policy_estimator = LocalPolicyEsitmator(self.env.observation_space.shape[0],self.env.action_space.n)
        policy= ParameterizedPolicy(policy_estimator)
        self.local_explorer= LocalBatchPolicyGridentExplorer(policy)

    def _run_one_episode(self):
        trajectory = []
        current_state = self.env.reset()
        
        for _ in range(0,LocalActor.MAX_STEPS):
            action_index = self.local_explorer.get_target_policy().get_action(current_state)
            distribution = self.local_explorer.get_target_policy().get_discrete_distribution_tensor(current_state)
            entropy = torch.distributions.Categorical(distribution).entropy()
            action_prob = distribution[action_index]
            state_value  = self.local_critic.get_value_function().predict(current_state)
            observation = self.env.step(action_index)
            reward = observation[1]
            self.env.render()
            trajectory.append((current_state,state_value,action_index,action_prob,entropy,reward))
            done = observation[2]
            if done:
                break
            current_state = observation[0]
        return trajectory
    
    def set_weights_for_esitmator(self,value_model_weights,policy_model_weights):
        self.local_critic.value_estimator.model.set_weights(value_model_weights)
        self.local_explorer.policy.policy_estimator.model.set_weights(policy_model_weights)
    
    def get_grident_from_esitmator(self):
        local_value_model_grident = self.local_critic.value_estimator.model.get_gradients()
        local_policy_model_grident = self.local_explorer.policy.policy_estimator.model.get_gradients()

        return local_value_model_grident,local_policy_model_grident

    def act(self, *args):
        trajectory = self._run_one_episode()    
        if len(trajectory)< LocalActor.MAX_STEPS:
            self.local_critic.evaluate(trajectory)
            self.local_explorer.explore(trajectory)
        
            return self.id,self.get_grident_from_esitmator()


class GlobalActor(ActorBase):
    MAX_STEPS = 500000

    def __init__(self,env,global_policy):
        self.env = env
        self.global_policy = global_policy
        self.writer = SummaryWriter(comment="-global_actor")

    def act(self, *args):
        episode = args[0]
        self._run_one_episode(episode)
    
    def _run_one_episode(self,episode):
        current_state = self.env.reset()
        for step in range(0,GlobalActor.MAX_STEPS):
            action_index = self.global_policy.get_action(current_state)
            observation = self.env.step(action_index)
            done = observation[2]
            if done:
                self.writer.add_scalar('steps_to_go',step,episode)
                break
            current_state = observation[0]


class BatchA3CAgent(Agent):
    EPS = np.finfo(np.float32).eps.item()
    NUM_PROCESS = 4

    def  __init__(self,env,num_episodes):
        self.env = env
        self.num_episodes = num_episodes
        self.global_value_esitmator = GlobalValueEestimator(env.observation_space.shape[0])
        self.global_policy_esitmator = GlobalPolicyEsitmator(env.observation_space.shape[0],env.action_space.n)

        global_policy = ParameterizedPolicy(self.global_policy_esitmator)
        self.global_actor = GlobalActor(self.env,global_policy)

        
    def update_global_model_weights(self,value_model_graident,policy_model_graident):
        self.global_value_esitmator.update(value_model_graident)
        self.global_policy_esitmator.update(policy_model_graident)
    

    def learn(self):
        actors =[LocalActor.remote(self.env.__class__.__name__,i) for i in range(BatchA3CAgent.NUM_PROCESS)]
        
        gradient_list=[]

        for actor in actors:
            # sync the local model weights to global model weights
            actor.set_weights_for_esitmator.remote(self.global_value_esitmator.model.get_weights(),self.global_policy_esitmator.model.get_weights())
            
            # start the actor and append the result to the gradient list
            gradient_list.append(actor.act.remote())

        for episode in tqdm(range(0, self.num_episodes)):
            
            # block to wait the actor result 
            done_actor, gradient_list = ray.wait(gradient_list)

        
            # Once one of the actor finished its job, we can get the gradient from the actor
            actor_id, grident =ray.get(done_actor)[0]

            # update the gloabl model weights with the lastest gradient
            self.update_global_model_weights(grident[0],grident[1])

            self.global_actor.act(episode)

            # sync the local model weights to global model weights
            actors[actor_id].set_weights_for_esitmator.remote(self.global_value_esitmator.model.get_weights(),self.global_policy_esitmator.model.get_weights())


            # act again          

            gradient_list.append(actors[actor_id].act.remote())

            




    


