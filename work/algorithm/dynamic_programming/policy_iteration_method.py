import copy

from lib.utility import create_distribution_greedily


class Policy_Iteration_Method:
    def __init__(self, v_table, p, delta=1e-3, discount=1.0):
        self.v_table = v_table
        self.transition_table = p
        self.delta = delta
        self.discount = discount
        self.build_distribution_method = create_distribution_greedily()

    def evaluate(self, policy):
        while True:
            delta = self.evaluate_once(policy)
            if delta < self.delta:
                break

    def evaluate_once(self, policy):
        delta = 1e-10
        new_v_table = copy.deepcopy(self.v_table)

        for state_index, transitions in self.transition_table.items():
            value_of_state = 0.0
            for action_index, transition in transitions.items():
                value_of_action = self._get_value_of_action(transition)
                value_of_state += policy.policy_table[state_index][action_index] * \
                    value_of_action

            new_v_table[state_index] = value_of_state
            delta = max(
                abs(self.v_table[state_index]-new_v_table[state_index]), delta)

        self.v_table = new_v_table

        return delta

    def improve(self, policy):
        delta = 1e-10
        old_policy = copy.deepcopy(policy)
        for state_index, transitions in self.transition_table.items():
            q_values = self._get_value_of_actions(transitions)
            distibution = self.build_distribution_method(q_values)
            for action_index, _ in transitions.items():
                policy.policy_table[state_index][action_index] = distibution[action_index]
                delta = max(abs(
                    old_policy.policy_table[state_index][action_index]-policy.policy_table[state_index][action_index]), delta)

        return delta

    def _get_value_of_actions(self, transitions):
        q_values = {}
        for action_index, transition in transitions.items():
            value_of_action = self._get_value_of_action(transition)
            q_values[action_index] = value_of_action
        return q_values

    def _get_value_of_action(self, transition):
        value_of_action = 0.0
        for transition_prob, next_state_index, reward, done in transition:  # For each next state
            # the reward is also related to the next state
            value_of_next_state = 0 if done else self.v_table[next_state_index]
            value_of_action += transition_prob * \
                (self.discount*value_of_next_state+reward)
        return value_of_action
