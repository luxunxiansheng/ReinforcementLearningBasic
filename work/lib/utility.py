import numpy as np


def create_distribution_greedily():
    def create_fn(dict_values):
        best_action_index = max(dict_values, key=dict_values.get)
        probs = {}
        for index, _ in dict_values.items():
            probs[index] = 0.0
        probs[best_action_index] = 1.0
        return probs
    return create_fn


def create_distribution_randomly():
    def create_fn(dict_values):
        probs = {}
        num_values = len(dict_values)
        for index, _ in dict_values.items():
            probs[index] = 1.0/num_values
        return probs
    return create_fn


def create_distribution_epsilon_greedily(epsilon):
    def create_fn(dict_values):
        probs = {}
        num_values = len(dict_values)
        for index, _ in dict_values.items():
            probs[index] = epsilon/num_values
        best_action_index = max(dict_values, key=dict_values.get)
        probs[best_action_index] += (1.0 - epsilon)
        return probs
    return create_fn
