import copy
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from lib.utility import (create_distribution_greedily,create_distribution_randomly)


class Q_Monte_Carlo_Method:
    def __init__(self, q_table, env, episodes=50000, discount=1.0):
        self.q_table = q_table
        self.env = env
        self.episodes = episodes
        self.discount = discount
        self.return_table = self._init_returns()
        self.create_distribution_greedily = create_distribution_greedily()

    def _init_returns(self):
        return_table = defaultdict(lambda: {})
        for state_index, action_values in self.q_table.items():
            for action_index, _ in action_values.items():
                return_table[state_index][action_index] = (0, 0.0)
        return return_table

    def evaluate(self, policy):
        for _ in tqdm(range(0, self.episodes)):
            trajectory = []
            current_state_index = self.env.reset()
            action_probability = policy.policy_table[current_state_index]
            action_index = np.random.choice(list(action_probability.keys()))

            while True:
                observation = self.env.step(action_index)
                
                reward = observation[1]
                trajectory.append((current_state_index, action_index, reward))
                
                current_state_index = observation[0]
                done = observation[2]
                if done:
                    break

                action_index = policy.select_action(current_state_index)

            R = 0.0
            for state_index, action_index, reward in trajectory[::-1]:
                R = reward+self.discount*R
                return_tuple = (self.return_table[state_index][action_index][0]+1, self.return_table[state_index][action_index][1]+R)
                self.return_table[state_index][action_index] = return_tuple
                self.q_table[state_index][action_index] = self.return_table[state_index][action_index][1]/self.return_table[state_index][action_index][0]

    def improve(self, policy):
        for state_index, _ in policy.policy_table.items():
            q_values= self.q_table[state_index]
            distribution = self.create_distribution_greedily(q_values)
            policy.policy_table[state_index]=distribution
