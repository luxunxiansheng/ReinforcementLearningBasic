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

from tqdm import tqdm

import matplotlib.pyplot as plt

from lib.utility import create_distribution_epsilon_greedily


class SARSA():
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.
    """

    def __init__(self, q_table, table_policy, epsilon, env, step_size=0.1, episodes=10000, discount=1.0):
        self.q_table = q_table
        self.policy = table_policy
        self.env = env
        self.episodes = episodes
        self.step_size = step_size
        self.discount = discount
        self.create_distribution_epsilon_greedily = create_distribution_epsilon_greedily(
            epsilon)

        self.episodesteps = {}
        

    def improve(self):
        for episode in tqdm(range(0, self.episodes)):
            self._run_one_episode(episode)

    def _run_one_episode(self,episode):
        steps = 0
   
        # S
        current_state_index = self.env.reset()

        # A
        current_action_index = self.policy.get_action(current_state_index)

        while True:

            observation = self.env.step(current_action_index)
            steps += 1

            # R
            reward = observation[1]
            done = observation[2]

            # S'
            next_state_index = observation[0]

            # A'
            next_action_index = self.policy.get_action(next_state_index)

            delta = reward + self.discount * self.q_table[next_state_index][next_action_index] - self.q_table[current_state_index][current_action_index]
            self.q_table[current_state_index][current_action_index] += self.step_size * delta

            # update policy softly 
            q_values = self.q_table[current_state_index]
            distribution = self.create_distribution_epsilon_greedily(q_values)
            self.policy.policy_table[current_state_index] = distribution

            if done:
                self.episodesteps[episode] = steps
                break

            current_state_index = next_state_index
            current_action_index = next_action_index


    def show_timesteps(self):
        x = []
        y = []
        for episode, steps in self.episodesteps.items():
            x.append(episode)
            y.append(steps)

        fig, ax = plt.subplots(1, figsize=(8, 6))
        fig.suptitle('Steps/Episode')

        # Plot the data
        ax.plot(x, y)

        # Show the grid lines as dark grey lines
        plt.grid(b=True, which='major', color='#666666', linestyle='-')

        plt.show()
