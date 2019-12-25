import sys
from abc import abstractmethod

import matplotlib.pyplot as plt
from gym.envs.toy_text import discrete
from mpl_toolkits.mplot3d import Axes3D


class Base_Discrete_Env(discrete.DiscreteEnv):

    @abstractmethod
    def build_Q_table(self):
        Q_table = {}
        for state_index in range(self.nS):
            Q_table[state_index] = {
                action_index: 0.0 for action_index in range(self.nA)}
        return Q_table

    @abstractmethod
    def build_V_table(self):
        V_table = {}
        for state_index in range(self.nS):
            V_table[state_index] = 0.0
        return V_table

    @abstractmethod
    def build_policy_table(self):
        policy_table = {}
        for state_index in range(self.nS):
            policy_table[state_index] = {
                action_index: 1.0/self.nA for action_index in range(self.nA)}
        return policy_table

    @abstractmethod
    def show_policy_on_console(self, policy):
        outfile = sys.stdout
        for state_index, probability_values in policy.policy_table.items():
            outfile.write("\n\nstate_index {:2d}\n".format(state_index))
            for action_index, probability_value in probability_values.items():
                outfile.write("        action_index {:2d} : value {}     ".format(
                    action_index, probability_value))
            outfile.write("\n")
        outfile.write(
            '--------------------------------------------------------------------------\n')

    @abstractmethod
    def show_policy(self, policy):

        x = []
        y = []
        for state_index, _ in policy.policy_table.items():
            x.append(state_index)
            y.append(policy.select_action(state_index))

        fig, ax = plt.subplots(1, figsize=(8, 6))
        fig.suptitle('Action taken at state')

        # Plot the data
        ax.scatter(x, y)

        # Show the grid lines as dark grey lines
        plt.grid(b=True, which='major', color='#666666', linestyle='-')

        plt.show()

    @abstractmethod
    def show_v_table(self, v_table):
        x = []
        y = []
        for state_index, value in v_table.items():
            x.append(state_index)
            y.append(value)

        fig, ax = plt.subplots(1, figsize=(8, 6))
        fig.suptitle('Value of State')

        # Plot the data
        ax.scatter(x, y)

        # Show the grid lines as dark grey lines
        plt.grid(b=True, which='major', color='#666666', linestyle='-')

        plt.show()

    @abstractmethod
    def show_v_table_on_console(self,v_table):
        outfile = sys.stdout
        for state_index, value in v_table.items():
            outfile.write(
                "\n\nstate_index {:2d}:\n {:2f}\n".format(state_index, value))
            outfile.write("\n")
        outfile.write(
            '--------------------------------------------------------------------------\n')

    @abstractmethod
    def show_q_table(self, q_table):

        x, y, z = [], [], []

        for state_index, actions in q_table.items():
            for action_index, value in actions.items():
                x.append(state_index)
                y.append(action_index)
                z.append(value)

        fig = plt.figure()
        ax = Axes3D(fig)
        #ax.plot(x, y, z, zdir='z')
        ax.scatter(x, y, z, c='r', marker='o')

        ax.set_xlabel('State Index')
        ax.set_ylabel('Action Index')
        ax.set_zlabel('Q_Value')

        plt.show()

    @abstractmethod
    def show_q_table_on_console(self, q_table):
        outfile = sys.stdout
        for state_index, action_values in q_table.items():
            outfile.write("\n\nstate_index {:2d}\n".format(state_index))
            for action_index, action_value in action_values.items():
                outfile.write("        action_index {:2d} : value {}     ".format(
                    action_index, action_value))
            outfile.write("\n")
        outfile.write(
            '--------------------------------------------------------------------------\n')
