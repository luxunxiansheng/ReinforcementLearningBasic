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
from algorithm.value_based.tabular_solution_method.td_method.td_common import TDCritic
import heapq
from abc import abstractmethod

import numpy as np
from tqdm import tqdm
from copy import deepcopy

TRIVAL = 1
PRIORITY = 2

class DynaQTrivalCritic(TDCritic):
    def __init__(self, value_function, model, step_size=0.1, discount=1.0,iterations=5):
        super().__init__(value_function, step_size)
        self.model = model 
        self.iterations = iterations
        self.discount = discount

    def evaluate(self, *args):
        current_state_index  = args[0]
        current_action_index = args[1]
        reward = args[2]
        next_state_index = args[3]
        
        q_values_next_state = self.value_function[next_state_index]
        max_value = max(q_values_next_state.values())
        target = reward + max_value
        self.update(current_state_index,current_action_index,target)

        self.model.feed(current_state_index, current_action_index, next_state_index, reward)
        # Take Q learning several times. 
        for _ in tqdm(range(0, self.iterations)):
            sampled_current_state_index, sampled_current_action_index, sampled_next_state_index, sampled_reward = self.model.sample()
            sampled_q_values_next_state = self.value_function[sampled_next_state_index]
            max_value = max(sampled_q_values_next_state.values())
            sampled_target = sampled_reward + self.discount*max_value
            self.update(sampled_current_state_index,sampled_current_action_index,sampled_target)


class DynaQPriorityCritic(TDCritic):
    def __init__(self, value_function, model, step_size=0.1, discount = 1.0,iterations=5):
        super().__init__(value_function, step_size)
        self.model = model 
        self.iterations = iterations
        self.discount = discount

    def evaluate(self, *args):
        current_state_index  = args[0]
        current_action_index = args[1]
        reward = args[2]
        next_state_index = args[3]
        
        q_values_next_state = self.value_function[next_state_index]
        max_value = max(q_values_next_state.values())
        target = reward + max_value
        self.update(current_state_index,current_action_index,target)

        self.model.feed(current_state_index, current_action_index, next_state_index, reward)
        
        #if the value updated a lot, we put it into the priority queue 
        q_values_next_state = self.value_function[next_state_index]
        max_value = max(q_values_next_state.values())
        delta = reward + self.discount * max_value - self.value_function[current_state_index][current_action_index]
        priority = np.abs(delta)
        if priority > self.model.theta:
            self.model.priority_queue.add_item((current_state_index, current_action_index), -priority)

        # prioritized Sweeping 
        for _ in tqdm(range(0, self.iterations)):
            if self.model.priority_queue.empty():
                return
            _, sampled_current_state_index, sampled_current_action_index, sampled_next_state_index, sampled_reward = self.model.sample()
            sampled_q_values_next_state = self.value_function[sampled_next_state_index]
            max_value = max(sampled_q_values_next_state.values())
            sampled_target = sampled_reward+self.discount*max_value
            self.update(sampled_current_state_index,sampled_current_action_index,sampled_target)

            # deal with all the predecessors of the sample state
            for pre_state_index, pre_action_index, reward in self.model.get_predecessors(sampled_current_state_index):
                q_values_current_state = self.value_function[sampled_current_state_index]
                max_value = max(q_values_current_state.values())
                delta = reward + self.discount * max_value - self.value_function[pre_state_index][pre_action_index]
                priority = np.abs(delta)
                if priority > self.model.theta:
                    self.model.priority_queue.add_item((pre_state_index, pre_action_index), -priority)

class DynaQ:
    def __init__(self, critic,actor, env, statistics, episodes):
        self.env = env
        self.episodes = episodes
        self.critic = critic 
        self.actor  = actor 
        self.statistics = statistics

    def improve(self):
        for episode in tqdm(range(0, self.episodes)):
            self._run_one_episode(episode)

    def _run_one_episode(self, episode):
        # S
        current_state_index = self.env.reset()

        while True:
            # A
            current_action_index = self.actor.get_behavior_policy().get_action(current_state_index)
            observation = self.env.step(current_action_index)

            # R
            reward = observation[1]
            done = observation[2]

            self.statistics.episode_rewards[episode] += reward
            self.statistics.episode_lengths[episode] += 1

            # S'
            next_state_index = observation[0]

            self.critic.evaluate(current_state_index,current_action_index,reward,next_state_index)
            self.actor.improve(current_state_index)

            if done:
                break

            current_state_index = next_state_index



        
#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

class Model():
    # @rand: an instance of np.random.RandomState for sampling
    def __init__(self, rand=np.random):
        self.model = dict()
        self.rand = rand
    
    
    # feed the model with previous experience
    def feed(self, state_index, action_index, next_state_index, reward):
        if state_index not in self.model.keys():
            self.model[state_index] = dict()
        self.model[state_index][action_index] = [next_state_index, reward]

    @abstractmethod
    def sample(self):
        pass


# Trivial model for planning in Dyna-Q
class TrivialModel(Model):
    # randomly sample from previous experience
    def sample(self):
        state_index = list(self.model)[self.rand.choice(range(len(self.model.keys())))]
        action_index = list(self.model[state_index])[self.rand.choice(range(len(self.model[state_index].keys())))]
        next_state_index, reward = self.model[state_index][action_index]
        return state_index, action_index, next_state_index, reward



class PriorityQueue:
    def __init__(self):
        self.pq = []
        self.entry_finder = {}
        self.REMOVED = '<removed-task>'
        self.counter = 0

    def add_item(self, item, priority=0):
        if item in self.entry_finder:
            the_entry = self.entry_finder.pop(item)
            the_entry[-1] = self.REMOVED
        entry = [priority, self.counter, item]
        self.counter += 1
        self.entry_finder[item] = entry
        heapq.heappush(self.pq, entry)

    def pop_item(self):
        while self.pq:
            priority, count, item = heapq.heappop(self.pq)
            if item is not self.REMOVED:
                del self.entry_finder[item]
                return item, priority
        raise KeyError('pop from an empty priority queue')

    def empty(self):
        return not self.entry_finder

# Model containing a priority queue for Prioritized Sweeping


class PriorityModel(Model):
    def __init__(self, rand=np.random, theta=0):
        super().__init__(rand)
        # maintain a priority queue
        self.priority_queue = PriorityQueue()
        # track predecessors for every state
        self.predecessors = dict()
        self.theta = theta

    # get the first item in the priority queue
    def sample(self):
        (state_index, action_index), priority = self.priority_queue.pop_item()
        next_state_index, reward = self.model[state_index][action_index]
        return -priority, state_index, action_index, next_state_index, reward

    # feed the model with previous experience
    def feed(self, state_index, action_index, next_state_index, reward):
        super().feed(state_index, action_index, next_state_index, reward)
        if next_state_index not in self.predecessors.keys():
            self.predecessors[next_state_index] = set()
        self.predecessors[next_state_index].add((state_index, action_index))

    # get all seen predecessors of a state @state
    def get_predecessors(self, state_index):
        if state_index not in self.predecessors.keys():
            return []
        predecessors = []
        for state_index_pre, action_index_pre in list(self.predecessors[state_index]):
            predecessors.append([state_index_pre, action_index_pre,self.model[state_index_pre][action_index_pre][1]])
        return predecessors

