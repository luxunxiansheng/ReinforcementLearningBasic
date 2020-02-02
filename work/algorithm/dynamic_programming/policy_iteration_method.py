import copy

from lib.utility import create_distribution_greedily


class Policy_Iteration_Method:
    def __init__(self, v_table, policy, transition_table, delta=1e-5, discount=1.0):
        self.v_table = v_table
        self.policy = policy
        self.transition_table = transition_table
        self.delta = delta
        self.discount = discount

    def evaluate(self):
        while True:
            delta = self.evaluate_once()
            if delta < self.delta:
                break

    """     
    def evaluate_once(self):
        delta = 1e-10
        new_v_table = copy.deepcopy(self.v_table)

        for state_index, transitions in self.transition_table.items():
            value_of_state = 0.0
            for action_index, transition in transitions.items():
                value_of_action = self._get_value_of_action(transition)
                value_of_state += self.policy.policy_table[state_index][action_index] * \
                    value_of_action

            new_v_table[state_index] = value_of_state
            delta = max(
                abs(self.v_table[state_index]-new_v_table[state_index]), delta)

        self.v_table = new_v_table
        return delta """

    def evaluate_once(self):
        delta = 1e-10
        new_v_table = copy.deepcopy(self.v_table)

        for state_index, _ in self.v_table.items():
            value_of_state = 0.0
            action_transitions = self.transition_table[state_index]
            for action_index, transitions in action_transitions.items():
                value_of_action = self._get_value_of_action(transitions)
                value_of_state += self.policy.policy_table[state_index][action_index] * value_of_action
            new_v_table[state_index] = value_of_state
            delta = max(abs(self.v_table[state_index]-new_v_table[state_index]), delta)

        self.v_table = new_v_table
        return delta

    def improve(self):
        delta = 1e-10
        old_policy = copy.deepcopy(self.policy)
        for state_index, actions in self.policy.policy_table.items():
            q_values = {}
            for action_index, _ in actions.items():
                transition = self.transition_table[state_index][action_index]
                q_values[action_index] = self._get_value_of_action(transition)

            greedy_distibution = create_distribution_greedily()(q_values)
            self.policy.policy_table[state_index] = greedy_distibution
            delta = max(abs(old_policy.policy_table[state_index]-self.policy.policy_table[state_index]), delta)
        return delta

    

    def _get_value_of_action(self, transitions):
        value_of_action = 0.0
        for transition_prob, next_state_index, reward, done in transitions:  # For each next state
            # the reward is also related to the next state
            value_of_next_state = 0 if done else self.v_table[next_state_index]
            value_of_action += transition_prob * (self.discount*value_of_next_state+reward)
        return value_of_action
