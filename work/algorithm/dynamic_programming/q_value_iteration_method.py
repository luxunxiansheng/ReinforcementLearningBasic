import copy

from lib.utility import create_distribution_greedily


class Q_Value_Iteration_Method:
    def __init__(self, q_table, table_policy, p, delta=1e-8):
        self.q_table = q_table
        self.policy =  table_policy
        self.transition_table = p
        self.delta = delta
        self.create_distribution_greedily = create_distribution_greedily()

    def improve(self, policy):
        while True:
            delta = self._bellman_optimize()
            if delta < self.delta:
                break

        for state_index, action_values in self.q_table.items():
            distibution = self.create_distribution_greedily(action_values)
            policy.policy_table[state_index]= distibution

    def _bellman_optimize(self):
        delta = 1e-10
        new_q_table = copy.deepcopy(self.q_table)
        for state_index, action_values in self.q_table.items():
            for action_index, action_value in action_values.items():
                optimal_value_of_action = self._get_optimal_value_of_action(
                    state_index, action_index)
                delta = max(abs(action_value-optimal_value_of_action), delta)
                new_q_table[state_index][action_index] = optimal_value_of_action

        self.q_table = new_q_table

        return delta

    def _get_optimal_value_of_action(self, state_index, action_index, discount=1.0):
        current_env_transition = self.transition_table[state_index][action_index]
        optimal_value_of_action = 0
        for transition_prob, next_state_index, reward, done in current_env_transition:  # For each next state
            optimal_value_of_next_state = 0 if done else self._get_optimal_value_of_state(
                next_state_index)
            # the reward is also related to the next state
            optimal_value_of_action += transition_prob * \
                (discount*optimal_value_of_next_state+reward)
        return optimal_value_of_action

    def _get_optimal_value_of_state(self, state_index):
        action_values_of_state = self.q_table[state_index]
        return max(action_values_of_state.values())
