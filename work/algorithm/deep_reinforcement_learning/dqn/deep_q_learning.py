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

from PIL.Image import init
import torch
import torch.nn as nn  
import torch.optim as optim
from torchvision import transforms


from common import Agent, CriticBase, ExplorerBase, QValueEstimator,ActorBase
from model.deep_mind_network_base import DeepMindNetworkBase

from lib.replay_memory import Replay_Memory
from lib.utility import create_distribution_boltzmann
from algorithm.deep_reinforcement_learning.dqn.continuous_state_value_table_policy import ContinuousStateValueTablePolicy


class DeepQValueEstimator(QValueEstimator):
    def __init__(self,input_channels,output_size,momentum,learning_rate,weight_decay,device):
        self.device = device
        self.model = DeepMindNetworkBase.create("DeepMindNetwork",input_channels,output_size).to(self.device)
        self.criterion = nn.SmoothL1Loss()
        self.optimizer = optim.RMSProp(self.model.parameters(),learning_rate,momentum,weight_decay)
    
    def predict(self, state, action):
        return self.model(state.to(self.device)[action] if action is not None else self.model(state.to(self.device)))
    
    def update(self, *args):
        q_values = args[0].to(self.device)
        target_values = args[1].to(self.device)

        loss = self.criterion(q_values,target_values)

        self.optimizer.zero_grad()
        loss.backward()

        for param in self.policy_model.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()

class DeepQLearningCritic(CriticBase):
    def __init__(self,policy_estimator,target_estimator,batch_size,action_space,discount):
        self.policy_estimator = policy_estimator
        self.target_estimator = target_estimator
        self.batch_size = batch_size
        self.action_space = action_space
        self.discount = discount 

    def evaluate(self,*args):
        samples = args[0]

        q_values = torch.zeros(self.batch_size, self.action_space)
        target_values = torch.zeros(self.batch_size, self.action_space)

        for sample_index in range(0, self.batch_size):
            state = samples[sample_index][0]
            action = samples[sample_index][1]
            reward = samples[sample_index][2]
            next_state = samples[sample_index][3]
            terminal = samples[sample_index][4]
        
            the_optimal_q_value_of_next_state = torch.max(self.target_estimator.predict(next_state))

            target_values[sample_index][int(action)] = reward if terminal else reward + self.discount*the_optimal_q_value_of_next_state
            
            q_values[sample_index][int(action)] = self.policy_estimator.predict(state)[int(action)]
        
        self.policy_estimator.update(q_values,target_values)

    def sync_target_model_with_policy_model(self):
        self.target_estimator.model.load_state_dict(self.policy_estimator.model.state_dict())
        self.target_estimator.model.eval()
    

class BoltzmannExplorer(ExplorerBase):
    def __init__(self, policy):
        self.policy = policy
        self.policy.create_distribution_fn = create_distribution_boltzmann()
    
    def explore(self, *args):
        pass 
        
    def get_behavior_policy(self):
        return self.policy

class DeepQLearningActor(ActorBase):
    def __init__(self,env,critic,explorer,relay_memory_capaticy,img_rows,img_columns,init_observations,sync_frequency):
        self.env = env
        self.critic = critic
        self.explorer = explorer
        self.relay_memory = Replay_Memory(relay_memory_capaticy)
        self.img_rows = img_rows 
        self.img_columns = img_columns
        self.init_observations = init_observations
        self.sync_frequency = sync_frequency

    def _preprocess_snapshot(self, screenshot):
        transform = transforms.Compose([transforms.CenterCrop((150, 600)),
                                        transforms.Resize((self.img_rows, self.img_columns)),
                                        transforms.Grayscale(),
                                        transforms.ToTensor()])
        return transform(screenshot)

    def _env_reset(self):
        init_screentshot= self._preprocess_snapshot(self.env.reset())
        return torch.stack((init_screentshot,init_screentshot,init_screentshot,init_screentshot))
    
    def _env_step(self,current_state,action):
        screen_shot, reward, terminal, score = self.env.step(action)
        preprocessed_snapshot = self._preprocess_snapshot(screen_shot)
        next_state = current_state.clone()
        next_state[0:-1] = current_state[1:]
        next_state[-1] = preprocessed_snapshot
        return next_state, torch.tensor(reward), torch.tensor(terminal), score

    def act(self, *args):
        self.critic.sync_target_model_with_policy_model()
        
        time_step = 0 
        current_state = self._env_reset()
        while (True):
            action_index = self.explorer.get_behavior_policy().get_action(current_state)
            observation = self._env_step(current_state,action_index)
            next_state = observation[0]
            reward = observation[1]
            done = observation[2]
            self.relay_memory.push((current_state, action_index, reward, next_state, done))
            if self.relay_memory.size() > self.init_observations:
                self.critic.evaluate(self.relay_memory.sample())

                if time_step % self.sync_frequency == 0:
                    self.critic.sync_target_model_with_policy_model()

            if done:
                break
            current_state = next_state
            time_step += 1

class DeepQLearningAgent(Agent):
    def __init__(self,env,config,device):
        self.env = env
        self.device = device
    
        self.final_epsilon = config['GLOBAL'].getfloat('final_epsilon')
        self.init_epsilon = config['GLOBAL'].getfloat('init_epsilon')
        self.discount = config['GLOBAL'].getfloat('discount') 

        self.image_stack_size = config['DQN'].getint('image_stack_size')
        self.action_space = config['DQN'].getint('action_space')
        self.img_rows = config['DQN'].getint('img_rows')
        self.img_columns = config['DQN'].getint('img_columns')

        self.batch_size = config['DQN'].getint('batch')
        self.momentum = config['DQN'].getfloat('momentum')
        self.lr = config['DQN'].getfloat('learning_rate')
        self.weight_decay = config['DQN'].getfloat('weight_decay')

        self.replay_memory_capacity = config['DQN'].getint('replay_memory_capacity')
        self.sync_frequency = config['DQN'].getint('sync_frequency')

        self.init_observations = config['DQN'].getint('init_observations')
        
    
        policy_estimator = DeepQValueEstimator(self.image_stack_size,self.action_space,self.lr,self.momentum,self.weight_decay,self.device)
        target_estimator = DeepQValueEstimator(self.image_stack_size,self.action_space,self.lr,self.momentum,self.weight_decay,self.device)
        
        self.critic = DeepQLearningCritic(policy_estimator,target_estimator,self.batch_size,self.action_space,self.discount)
        policy = ContinuousStateValueTablePolicy(policy_estimator)
        self.explorer = BoltzmannExplorer(policy)

        self.act = DeepQLearningActor(self.env,self.critic,self.explorer,self.replay_memory_capacity,self.img_rows,self.img_columns,self.init_observations,self.sync_frequency)
        

    def learn(self):
        self.act.act()

