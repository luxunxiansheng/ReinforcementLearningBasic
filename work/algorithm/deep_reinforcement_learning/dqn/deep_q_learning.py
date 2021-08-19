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
import torch.nn as nn  
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from torchvision import transforms
from PIL.Image import init
from tqdm import tqdm

from common import Agent, CriticBase, ExplorerBase, QValueEstimator,ActorBase
from model.deep_mind_network_base import DeepMindNetworkBase

from lib.replay_memory import Replay_Memory
from lib.utility import Utilis, create_distribution_boltzmann, create_distribution_epsilon_greedily
from algorithm.deep_reinforcement_learning.dqn.continuous_state_value_table_policy import ContinuousStateValueTablePolicy


class DeepQValueEstimator(QValueEstimator):
    def __init__(self,input_channels,output_size,momentum,learning_rate,weight_decay,device):
        self.device = device
        self.model = DeepMindNetworkBase.create("DeepMindNetwork",input_channels,output_size).to(self.device)
        self.criterion = nn.SmoothL1Loss()
        self.optimizer = optim.RMSprop(self.model.parameters(),learning_rate,momentum,weight_decay)
        
    def predict(self, state, action=None):
    
        return self.model(state.to(self.device))[action] if action is not None  else self.model(state.to(self.device))
    
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

        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()

class DeepQLearningCritic(CriticBase):
    def __init__(self,policy_estimator,target_estimator,batch_size,action_space,discount):
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
    
    
class ESoftExplorer(ExplorerBase):
    def __init__(self, policy,init_epsilon,final_epsilon,decay_rate):
        self.policy = policy
        self.init_epsilon = init_epsilon
        self.final_epsilon = final_epsilon
        self.decay_rate = decay_rate
        self.epsilon = self.init_epsilon
        self.policy.create_distribution_fn = create_distribution_epsilon_greedily(init_epsilon)
    
    def explore(self, *args):
        state = args[0]
        action_index =self.policy.get_action(state)
        
        # reduced the epsilon (exploration parameter) gradually
        if self.epsilon > self.final_epsilon:
            self.epsilon -= (self.init_epsilon - self.final_epsilon) * self.decay_rate
        self.policy.create_distribution_fn = create_distribution_epsilon_greedily(self.epsilon)

        return action_index

    def get_behavior_policy(self):
        return self.policy

class BoltzmannExplorer(ExplorerBase):
    def __init__(self, policy):
        self.policy = policy
        self.policy.create_distribution_fn = create_distribution_boltzmann()
    
    def explore(self, *args):
        return self.policy.get_action(args[0])
        
    def get_behavior_policy(self):
        return self.policy

class DeepQLearningActor(ActorBase):
    def __init__(self,env,critic,explorer,relay_memory_capaticy,img_rows,img_columns,init_observations,sync_frequency,batch_size):
        self.env = env
        self.critic = critic
        self.explorer = explorer
        self.relay_memory = Replay_Memory(relay_memory_capaticy)
        self.img_rows = img_rows 
        self.img_columns = img_columns
        self.init_observations = init_observations
        self.sync_frequency = sync_frequency
        self.batch_size = batch_size
        self.writer = SummaryWriter()

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
        episode = args[0]
        self.critic.sync_target_model_with_policy_model()
        time_step = 0 
        current_state = self._env_reset()
        while (True):
            action_index = self.explorer.explore(current_state)
            observation = self._env_step(current_state,action_index)
            next_state = observation[0]
            reward = observation[1]
            done = observation[2]
            score = observation[3]
            self.relay_memory.push((current_state, action_index, reward, next_state, done))
            if self.relay_memory.size() > self.init_observations:
                self.critic.evaluate(self.relay_memory.sample(self.batch_size),done,episode,self.writer)
                if time_step % self.sync_frequency == 0:
                    self.critic.sync_target_model_with_policy_model()
            if done:
                self.writer.add_scalar('episode_score',score,episode)
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
        self.episodes = config['GLOBAL'].getint('episodes')
        self.check_point_path = config['GLOBAL'].get('check_point_path')
        self.check_frequency = config['GLOBAL'].getint('check_frequency')

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

        self.init_epsilon = config['DQN'].getfloat('init_epsilon')
        self.final_epsilon = config['DQN'].getfloat('final_epsilon')
        self.epsilon_decay_rate = config['DQN'].getfloat('epsilon_decay_rate')

        self.init_observations = config['DQN'].getint('init_observations')


        policy_estimator = DeepQValueEstimator(self.image_stack_size,self.action_space,self.lr,self.momentum,self.weight_decay,self.device)
        target_estimator = DeepQValueEstimator(self.image_stack_size,self.action_space,self.lr,self.momentum,self.weight_decay,self.device)
        
        self.critic = DeepQLearningCritic(policy_estimator,target_estimator,self.batch_size,self.action_space,self.discount)
        policy = ContinuousStateValueTablePolicy(policy_estimator)
        #self.explorer = ESoftExplorer(policy,self.init_epsilon,self.final_epsilon,self.epsilon_decay_rate)
        self.explorer = BoltzmannExplorer(policy)

        self.actor = DeepQLearningActor(self.env,self.critic,self.explorer,self.replay_memory_capacity,self.img_rows,self.img_columns,self.init_observations,self.sync_frequency,self.batch_size)

    def learn(self):
        elapsed_episode = 0
        checkpoint = Utilis.load_checkpoint(self.__class__.__name__,self.check_point_path)
        if checkpoint is not None:
            self.actor.critic.policy_estimator.model.load_state_dict(checkpoint['policy_value_model_state_dict'])
            self.actor.critic.policy_estimator.optimizer.load_state_dict(checkpoint['policy_value_optimizer_state_dict'])
            elapsed_episode = checkpoint['episode']
        
        for episode in tqdm(range(elapsed_episode, self.episodes)):
            self.actor.act(episode)
            if episode % self.check_frequency == 0:
                checkpoint = {'episode': episode,
                                'policy_value_model_state_dict': self.actor.critic.policy_estimator.model.state_dict(),
                                'policy_value_optimizer_state_dict': self.actor.critic.policy_estimator.optimizer.state_dict()}
                Utilis.save_checkpoint(checkpoint,self.__class__.__name__,self.check_point_path)
                    
