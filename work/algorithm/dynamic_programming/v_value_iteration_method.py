import copy
import sys

from lib.utility import create_distribution_greedily


class V_Value_Iteration_Method:
    def __init__(self,v_table,p,delta=1e-3):
        self.v_table = v_table
        self.transition_table = p
        self.delta=delta
        self.build_distribution_method = create_distribution_greedily()
        
    def improve(self,policy):
        while True:
            delta= self._bellman_optimize()
            if delta < self.delta:
                break 


        for state_index,transitions in self.transition_table.items():
            q_values = self._get_value_of_actions(transitions)
            distibution=self.build_distribution_method(q_values)
            for action_index, _ in transitions.items():
                policy[state_index][action_index]= distibution[action_index]
              
              
    
    def _bellman_optimize(self):
        delta= 1e-10
        new_v_table = copy.deepcopy(self.v_table)
        for state_index,transitions in self.transition_table.items():
            q_values = self._get_value_of_actions(transitions)
            new_v_table[state_index]= self._get_optimal_value_of_action(q_values)
            delta = max(abs(self.v_table[state_index]-new_v_table[state_index]),delta)   
        self.v_table = new_v_table
        self._show_v_table()
        return delta

    def _get_value_of_actions(self,transitions,discount=0.9):
        q_values={}
        for action_index,transition in transitions.items():
            value_of_action = self._get_value_of_action(transition,discount)  
            q_values[action_index] = value_of_action
        return q_values  
    
    def _get_value_of_action(self,transition,discount):
        value_of_action = 0
        for transition_prob, next_state_index, reward, done in transition:  # For each next state
            # the reward is also related to the next state
            value_of_action += transition_prob*(discount*self.v_table[next_state_index]+reward)
        return value_of_action
   
    def _get_optimal_value_of_action(self,q_values):
        return max(q_values.values())

    def _show_v_table(self):
       
        outfile = sys.stdout
        for state_index, value in self.v_table .items():
            outfile.write("\n\nstate_index {:2d}:{:2f}\n".format(state_index,value))
            outfile.write("\n")
        outfile.write('--------------------------------------------------------------------------\n')

    