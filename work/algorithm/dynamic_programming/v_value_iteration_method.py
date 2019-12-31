import copy

from lib.utility import create_distribution_greedily


class V_Value_Iteration_Method:
    def __init__(self, v_table, p, delta=1e-8, discount=1.0):
        self.v_table = v_table
        self.transition_table = p
        self.delta = delta
        self.discount = discount
        self.create_distribution_greedily = create_distribution_greedily()

    def improve(self, policy):
        while True:
            delta = self._bellman_optimize()
            if delta < self.delta:
                break

        for state_index, actions in policy.policy_table.items():
            q_values={}
            for action_index, _ in actions.items():
                transition= self.transition_table[state_index][action_index]
                q_values[action_index]=self._get_value_of_action(transition)
            
            greedy_distibution = self.create_distribution_greedily(q_values)
            policy.policy_table[state_index]=greedy_distibution 

    def _bellman_optimize(self):
        delta = 1e-10
        new_v_table = copy.deepcopy(self.v_table)
        for state_index, transitions in self.transition_table.items():
            
            q_values = {}
            for action_index, transition in transitions.items():
                value_of_action = self._get_value_of_action(transition)
                q_values[action_index] = value_of_action
            
            new_v_table[state_index] = self._get_optimal_value_of_action(q_values)
            delta = max(abs(self.v_table[state_index]-new_v_table[state_index]), delta)
        self.v_table = new_v_table
        return delta



    def _get_value_of_action(self, transition):
        value_of_action = 0
        for transition_prob, next_state_index, reward, done in transition:  # For each next state
            # the reward is also related to the next state
            value_of_next_state = 0 if done else self.v_table[next_state_index]
            value_of_action += transition_prob * \
                (self.discount * value_of_next_state+reward)
        return value_of_action

    def _get_optimal_value_of_action(self, q_values):
        return max(q_values.values())
