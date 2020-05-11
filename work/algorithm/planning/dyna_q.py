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

import heapq
from copy import deepcopy

import numpy as np
from tqdm import tqdm

from lib.utility import create_distribution_epsilon_greedily


class DynaQ:
    def __init__(self, q_table, behavior_table_policy, epsilon, env, statistics, episodes, iterations= 50, step_size=0.1,  discount=1.0):
        self.q_table = q_table
        self.policy = behavior_table_policy
        self.env = env
        self.episodes = episodes
        self.step_size = step_size
        self.discount = discount
        self.create_distribution_epsilon_greedily = create_distribution_epsilon_greedily(
            epsilon)
        self.statistics = statistics
        self.model = TrivialModel()
        self.iterations = iterations

    def improve(self):
        for episode in tqdm(range(0, self.episodes)):
            self._run_one_episode(episode)

    def _run_one_episode(self, episode):
         # S
        current_state_index = self.env.reset()

        while True:
            # A
            current_action_index = self.policy.get_action(current_state_index)
            observation = self.env.step(current_action_index)

            # R
            reward = observation[1]
            done = observation[2]

            self.statistics.episode_rewards[episode] += reward
            self.statistics.episode_lengths[episode] += 1

            # S'
            next_state_index = observation[0]

            q_values_next_state = self.q_table[next_state_index]
            max_value = max(q_values_next_state.values())
            delta = reward + self.discount * max_value - self.q_table[current_state_index][current_action_index]
            self.q_table[current_state_index][current_action_index] += self.step_size * delta

            
            self.model.feed(current_state_index, current_action_index, next_state_index, reward)
            for _ in tqdm(range(0,self.iterations)):
                current_state_index, current_action_index, next_state_index, reward = self.model.sample()
                q_values_next_state = self.q_table[next_state_index]
                max_value = max(q_values_next_state.values())
                delta = reward + self.discount * max_value - self.q_table[current_state_index][current_action_index]
                self.q_table[current_state_index][current_action_index] += self.step_size * delta

            # update policy softly
            q_values = self.q_table[current_state_index]
            distribution = self.create_distribution_epsilon_greedily(q_values)
            self.policy.policy_table[current_state_index] = distribution


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

class PriorityQueue:
    def __init__(self):
        self.pq = []
        self.entry_finder = {}
        self.REMOVED = '<removed-task>'
        self.counter = 0

    def add_item(self, item, priority=0):
        if item in self.entry_finder:
            self.remove_item(item)
        entry = [priority, self.counter, item]
        self.counter += 1
        self.entry_finder[item] = entry
        heapq.heappush(self.pq, entry)

    def remove_item(self, item):
        entry = self.entry_finder.pop(item)
        entry[-1] = self.REMOVED

    def pop_item(self):
        while self.pq:
            priority, count, item = heapq.heappop(self.pq)
            if item is not self.REMOVED:
                del self.entry_finder[item]
                return item, priority
        raise KeyError('pop from an empty priority queue')

    def empty(self):
        return not self.entry_finder


# Trivial model for planning in Dyna-Q
class TrivialModel:
    # @rand: an instance of np.random.RandomState for sampling
    def __init__(self, rand=np.random):
        self.model = dict()
        self.rand = rand

    # feed the model with previous experience
    def feed(self, state, action, next_state, reward):
        state = deepcopy(state)
        next_state = deepcopy(next_state)
        if tuple(state) not in self.model.keys():
            self.model[tuple(state)] = dict()
        self.model[tuple(state)][action] = [list(next_state), reward]

    # randomly sample from previous experience
    def sample(self):
        state_index = self.rand.choice(range(len(self.model.keys())))
        state = list(self.model)[state_index]
        action_index = self.rand.choice(range(len(self.model[state].keys())))
        action = list(self.model[state])[action_index]
        next_state, reward = self.model[state][action]
        state = deepcopy(state)
        next_state = deepcopy(next_state)
        return list(state), action, list(next_state), reward

# Time-based model for planning in Dyna-Q+
class TimeModel:
    # @maze: the maze instance. Indeed it's not very reasonable to give access to maze to the model.
    # @timeWeight: also called kappa, the weight for elapsed time in sampling reward, it need to be small
    # @rand: an instance of np.random.RandomState for sampling
    def __init__(self, maze, time_weight=1e-4, rand=np.random):
        self.rand = rand
        self.model = dict()

        # track the total time
        self.time = 0

        self.time_weight = time_weight
        self.maze = maze

    # feed the model with previous experience
    def feed(self, state, action, next_state, reward):
        state = deepcopy(state)
        next_state = deepcopy(next_state)
        self.time += 1
        if tuple(state) not in self.model.keys():
            self.model[tuple(state)] = dict()

            # Actions that had never been tried before from a state were allowed to be considered in the planning step
            for action_ in self.maze.actions:
                if action_ != action:
                    # Such actions would lead back to the same state with a reward of zero
                    # Notice that the minimum time stamp is 1 instead of 0
                    self.model[tuple(state)][action_] = [list(state), 0, 1]

        self.model[tuple(state)][action] = [
            list(next_state), reward, self.time]

    # randomly sample from previous experience
    def sample(self):
        state_index = self.rand.choice(range(len(self.model.keys())))
        state = list(self.model)[state_index]
        action_index = self.rand.choice(range(len(self.model[state].keys())))
        action = list(self.model[state])[action_index]
        next_state, reward, time = self.model[state][action]

        # adjust reward with elapsed time since last vist
        reward += self.time_weight * np.sqrt(self.time - time)

        state = deepcopy(state)
        next_state = deepcopy(next_state)

        return list(state), action, list(next_state), reward

# Model containing a priority queue for Prioritized Sweeping


class PriorityModel(TrivialModel):
    def __init__(self, rand=np.random):
        TrivialModel.__init__(self, rand)
        # maintain a priority queue
        self.priority_queue = PriorityQueue()
        # track predecessors for every state
        self.predecessors = dict()

    # add a @state-@action pair into the priority queue with priority @priority
    def insert(self, priority, state, action):
        # note the priority queue is a minimum heap, so we use -priority
        self.priority_queue.add_item((tuple(state), action), -priority)

    # @return: whether the priority queue is empty
    def empty(self):
        return self.priority_queue.empty()

    # get the first item in the priority queue
    def sample(self):
        (state, action), priority = self.priority_queue.pop_item()
        next_state, reward = self.model[state][action]
        state = deepcopy(state)
        next_state = deepcopy(next_state)
        return -priority, list(state), action, list(next_state), reward

    # feed the model with previous experience
    def feed(self, state, action, next_state, reward):
        state = deepcopy(state)
        next_state = deepcopy(next_state)
        TrivialModel.feed(self, state, action, next_state, reward)
        if tuple(next_state) not in self.predecessors.keys():
            self.predecessors[tuple(next_state)] = set()
        self.predecessors[tuple(next_state)].add((tuple(state), action))

    # get all seen predecessors of a state @state
    def predecessor(self, state):
        if tuple(state) not in self.predecessors.keys():
            return []
        predecessors = []
        for state_pre, action_pre in list(self.predecessors[tuple(state)]):
            predecessors.append(
                [list(state_pre), action_pre, self.model[state_pre][action_pre][1]])
        return predecessors
