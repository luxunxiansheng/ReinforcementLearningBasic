import copy
import sys

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from lib.utility import create_distribution_greedily


class Q_Value_Iteration_Method:
    def __init__(self,q_table,p,delta=1e-8):
        self.q_table = q_table
        self.transition_table = p
        self.delta=delta
        self.build_distribution_method = create_distribution_greedily()
        
    def improve(self,policy):
        while True:
            delta= self._bellman_optimize()
            if delta < self.delta:
                break 
        
        for state_index, action_values in self.q_table.items():
            distibution=self.build_distribution_method(action_values)
            for action_index, _ in action_values.items():
                policy[state_index][action_index]= distibution[action_index]
            
    
    def _bellman_optimize(self):
        delta= 1e-10
        new_q_table = copy.deepcopy(self.q_table)
        for state_index,action_values in self.q_table.items():
            for action_index,action_value in action_values.items():
                optimal_value_of_action = self._get_optimal_value_of_action(state_index,action_index)  
                delta = max(abs(action_value-optimal_value_of_action),delta)
                new_q_table[state_index][action_index] = optimal_value_of_action

        self.q_table = new_q_table

        self._show_q_table()

        return delta

    
    def _get_optimal_value_of_action(self,state_index,action_index,discount=1.0):
        current_env_transition = self.transition_table[state_index][action_index]
        optimal_value_of_action = 0
        for transition_prob, next_state_index, reward, done in current_env_transition:  # For each next state
            optimal_value_of_next_state = 0 if done else self._get_optimal_value_of_state(next_state_index)   
            # the reward is also related to the next state
            optimal_value_of_action += transition_prob*(discount*optimal_value_of_next_state+reward)
        return optimal_value_of_action


    def _get_optimal_value_of_state(self,state_index):
        action_values_of_state = self.q_table[state_index]
        return max(action_values_of_state.values())

    def _show_q_table_on_console(self):
       
        outfile = sys.stdout
        for state_index, action_values in self.q_table.items():
            outfile.write("\n\nstate_index {:2d}\n".format(state_index))
            for action_index,action_value in action_values.items():
                outfile.write("        action_index {:2d} : value {}     ".format(action_index,action_value))
            outfile.write("\n")
        outfile.write('--------------------------------------------------------------------------\n')

    def _show_q_table(self):


        x, y, z = [], [], []

        for state_index, actions in self.q_table.items():
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
        