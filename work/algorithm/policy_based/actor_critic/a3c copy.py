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

# http://mcneela.github.io/machine_learning/2019/09/03/Writing-Your-Own-Optimizers-In-Pytorch.html

import numpy as np 

import torch
import torch.nn as nn 
import torch.nn.functional as F

from algorithm.policy_based.actor_critic.actor_critic_common import ValueEstimator
from lib.utility import SharedAdam
from common import ActorBase,CriticBase

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

class GlobalValueEestimator(ValueEstimator):
    def __init__(self,observation_space_size):
        self.model = ValueModel(observation_space_size)
        self.optimizer = SharedAdam(self.model.parameters(),lr=1e-3)

    def _sync_local_grads(local_model, global_model):
        for local_param, global_param in zip(local_model.parameters(),global_model.parameters()):
            if global_param.grad is not None:
                return
            global_param._grad = local_param.grad

    def predict(self,state):
        value = self.model.forward(state)
        return value

    def update(self,*args):
        loss = args[0]
        local_model = args[1]
        
        self._sync_local_grads(self.model,local_model)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class LocalValueEestimator(ValueEstimator):
    def __init__(self,observation_space_size):
        self.model = ValueModel(observation_space_size) 
    
    def predict(self,state):
        value = self.model.forward(state)
        return value
    
    def update(self, *args):
        global_model = args[0]
        self.model.load_state_dict(global_model.state_dict())
        
class OnlineCritic(CriticBase):
    def __init__(self,global_value_estimator,discount=1.0):
        self.global_estimator = global_value_estimator
        self.local_esitimator = LocalValueEestimator()
        self.discount  = discount
    
    def evaluate(self,*args):
        current_state_index = args[0]
        reward = args[1]
        next_state_index = args[2]    
        
        self.local_esitimator.update(self.global_estimator.model)
        value_of_next_state = self.local_esitimator.predict(next_state_index)

        # set the target 
        target = reward + self.discount * value_of_next_state
        input  = self.local_esitimator.predict(current_state_index)
        loss  = F.smooth_l1_loss(input,target)

        self.global_estimator.update(loss,self.local_esitimator.model)
    
    def get_value_function(self):
        return self.estimator

class PolicyModel(nn.Module):
        def __init__(self,observation_space_size,action_space_size):
            super().__init__()
            self.affine = nn.Linear(observation_space_size, 128)
            self.dropout = nn.Dropout(p=0.6)
            self.action_head = nn.Linear(128, action_space_size)
            
        def forward(self, state):
            state = torch.from_numpy(state).float()
            state = self.affine(state)
            state = self.dropout(state)
            state = F.relu(state)
            state = self.action_head(state)
            actions_prob = F.softmax(state,dim=0)
            return actions_prob

class GlobalPolicyEsitmator:
    def __init__(self,observation_space_size,action_space_size):
        self.model = PolicyModel(observation_space_size,action_space_size)
        self.optimizer = SharedAdam(self.model.parameters(),lr =1e-3)
    
    def _sync_local_grads(local_model, global_model):
        for local_param, global_param in zip(local_model.parameters(),global_model.parameters()):
            if global_param.grad is not None:
                return
            global_param._grad = local_param.grad

    def predict(self,state):
        actions_prob=self.model.forward(state)
        return actions_prob

    def update(self,*args):
        loss = args[0]
        local_model = args[1]

        self._sync_local_grads(self.model,local_model)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class LocalPolicyEsitmator:
    def __init__(self,observation_space_size,action_space_size):
        self.model = PolicyModel(observation_space_size,action_space_size)
        
    def predict(self,state):
        actions_prob=self.model.forward(state)
        return actions_prob

    def update(self,*args):
        global_model = args[0]
        self.model.load_state_dict(global_model.state_dict())
        

class OnlineActor(ActorBase):
    def __init__(self,global_policy_esitmator,critic,discount=1.0):
        self.global_policy_estimator =  global_policy_esitmator
        self.discount = discount
        self.critic = critic
        
    def improve(self,*args):
        current_state_index = args[0]
        reward = args[1]
        action_prob = args[2]
        next_state_index = args[3]
    
        advantage = torch.tensor(reward) + self.discount* self.critic.global_estimator.predict(next_state_index) - self.critic.global_estimator.predict(current_state_index)
        policy_loss = -torch.log(torch.round(action_prob*10**3)/10**3)*advantage.detach()
    
        self.global_policy_estimator.update(policy_loss)

    def get_behavior_policy(self):
        return self.policy


class AsynchronousAdvantageActorCritic:
    EPS = np.finfo(np.float32).eps.item()
    MAX_STEPS = 500000
    def  __init__(self,critic,actor,env,num_episodes):
        self.critic=critic 
        self.actor= actor
        self.env = env
        self.num_episodes = num_episodes
            
    def improve(self):
        for episode in range(0, self.num_episodes):
            self._run_one_episode(episode)

    def _run_one_episode(self, episode):
        # S
        current_state_index = self.env.reset()

        for _ in range(0, AsynchronousAdvantageActorCritic.MAX_STEPS):
            # A
            current_action_index = self.actor.get_behavior_policy().get_action(current_state_index)
            observation = self.env.step(current_action_index)
            self.env.render()

            # R
            reward = observation[1]
            done = observation[2]

            # S'
            next_state_index = observation[0]

            self.critic.evaluate(current_state_index,reward,next_state_index,done,episode,self.writer)
            
            action_prob=self.actor.get_behavior_policy().get_discrete_distribution_tensor(current_state_index)[current_action_index]
            self.actor.improve(current_state_index,reward,action_prob,next_state_index,done,episode,self.writer)

            if done:
                break

            current_state_index = next_state_index




























class AsynchronousAdvantageActorCritic:
    def improve(self):
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['CUDA_VISIBLE_DEVICES'] = ""

        args = parser.parse_args()

        torch.manual_seed(args.seed)
        env = create_atari_env(args.env_name)
        shared_model = ActorCritic(env.observation_space.shape[0], env.action_space)
        shared_model.share_memory()

        if args.no_shared:
            optimizer = None
        else:
            optimizer = my_optim.SharedAdam(shared_model.parameters(), lr=args.lr)
            optimizer.share_memory()

        processes = []

        counter = mp.Value('i', 0)
        lock = mp.Lock()

        p = mp.Process(target=test, args=(args.num_processes, args, shared_model, counter))
        p.start()
        processes.append(p)

        for rank in range(0, args.num_processes):
            p = mp.Process(target=train, args=(rank, args, shared_model, counter, lock, optimizer))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
