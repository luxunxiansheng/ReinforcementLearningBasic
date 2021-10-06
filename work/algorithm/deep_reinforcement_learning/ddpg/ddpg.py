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

from genericpath import samefile
from logging import setLoggerClass

from torch.utils.tensorboard import writer
from policy.policy import Policy
import numpy as np 


from torch import nn
from torch.utils.tensorboard.writer import SummaryWriter
from common import ActorBase, CriticBase, ExplorerBase, PolicyEstimator, QValueEstimator
from lib.replay_memory import Replay_Memory

class DeepQValueEstimator(QValueEstimator):
    def __init__(self,model,learning_rate,device):
        self.device = device
        self.model = model
        self.criterion = nn.MSELoss()
        self.optimizer = nn.optim.Adam(self.model.parameters(), learning_rate)

    def predict(self, state, action=None):
        assert action is None
        return self.model(state.to(self.device), action.to(self.device))
    
    def update(self, *args):
        q_values = args[0].to(self.device)
        target_values = args[1].to(self.device)
    
        loss = self.criterion(q_values,target_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class DDPGCritic(CriticBase):
    '''
    The policy is assumed optimal so we can evaluate the optimal value function     
    '''
    def __init__(self,q_value_estimator,q_value_target_estimator,target_policy,policy,discount):
        self.Q_value_estimator = q_value_estimator
        self.Q_value_target_estimator = q_value_target_estimator
        self.target_policy = target_policy
        self.policy = policy 
        self.discount = discount 

    def evaluate(self, *args):
        samples = np.array(args[0])

        current_state_indices = samples[:,0]
        current_action_indices = samples[:,1]
        q_values = self.Q_value_estimator.predict(current_state_indices,current_action_indices)

        rewards = samples[:,2]
        next_state_indices = samples[:,3]
        next_action_indices = self.target_policy.policy_estimator.predict(next_state_indices)
        q_value_of_next_states = self.Q_value_target_estimator.predict(next_state_indices,next_action_indices).detach()

        target_values = rewards + self.discount * q_value_of_next_states

        self.Q_value_estimator.update(q_values,target_values)


    def sync_target_model(self):
        self.Q_value_target_estimator.model.load_state_dict(self.Q_value_estimator.model.state_dict())
        self.Q_value_target_estimator.model.eval()

        self.target_policy.policy_estimator.model.load_state_dict(self.policy.policy_estimator.model.state_dict())
        self.target_policy.policy_estimator.model.eval()
        
    
    def get_optimal_policy(self):
        pass 
    
    def get_value_function(self):
        return self.Q_value_estimator


class DDPGExplorer(ExplorerBase):
    '''
    The policy will be optimal asymptotically with grident accent. 

    '''
    
    def __init__(self,policy,critic,discount=1.0):
        self.policy = policy 
        self.discount = discount
        self.critic = critic

    def explore(self, *args):
        samples = np.array(args[0])
            
        state_indices = samples[:,0]
        action_indices = self.policy.policy_estimator.predict(state_indices)

        # we take the mean of the q_values as the policy gain (and increase the policy grident to make it better)
        policy_loss = -self.critic.estimator.predict(state_indices,action_indices).mean()
        
        self.policy.policy_estimator.update(policy_loss)

    def get_behavior_policy(self):
        return self.policy



class DDPGActor(ActorBase):
    EPS = np.finfo(np.float32).eps.item()
    MAX_STEPS = 500000

    class OUNoise(object):
        def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
            self.mu           = mu
            self.theta        = theta
            self.sigma        = max_sigma
            self.max_sigma    = max_sigma
            self.min_sigma    = min_sigma
            self.decay_period = decay_period
            self.action_dim   = action_space.shape[0]
            self.low          = action_space.low
            self.high         = action_space.high
            self.reset()
            
        def reset(self):
            self.state = np.ones(self.action_dim) * self.mu
            
        def evolve_state(self):
            x  = self.state
            dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
            self.state = x + dx
            return self.state
        
        def get_action(self, action, t=0): 
            ou_state = self.evolve_state()
            self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
            return np.clip(action + ou_state, self.low, self.high)

    def __init__(self,env,critic,explorer,relay_memory_capaticy,init_observations,sync_frequency,batch_size):
        self.env = env
        self.critic = critic
        self.explorer = explorer
        
        self.noise = DDPGActor.OUNoise(self.env.action_space)

        self.relay_memory = Replay_Memory(relay_memory_capaticy)
        
        self.init_observations = init_observations
        self.sync_frequency = sync_frequency
        self.batch_size = batch_size
        self.writer = SummaryWriter()
        self.time_step = 0
    
    def act(self, *args):
        episode = args[0]

        if episode % self.sync_frequency == 0:
            self.critic.sync_target_model()

        current_state_index = self.env.reset()
        for step in range(0,DDPGActor.MAX_STEPS):
            current_action_index = self.noise.get_action(self.explorer.get_behavior_policy().get_action(current_state_index),step)  
            observation = self.env.step(current_state_index,current_action_index)
            next_state_index = observation[0]
            reward = observation[1]
            done = observation[2]
            score = observation[3]
            self.relay_memory.push((current_state_index, current_action_index, reward, next_state_index, done))
            if self.relay_memory.size() > self.init_observations:
                self.critic.evaluate(self.relay_memory.sample(self.batch_size))
                self.explorer.explore(self.relay_memory.sample(self.batch_size))
            if done:
                self.writer.add_scalar('episode_score',score,episode)
                break
            current_state_index = next_state_index

            self.time_step += 1



