import numpy as np
import math 


def create_distribution_greedily():
    def fn(dict_values):
        best_action_index = max(dict_values, key=dict_values.get)
        probs = {}
        for index, _ in dict_values.items():
            probs[index] = 0.0
        probs[best_action_index] = 1.0
        return probs
    return fn


def create_distribution_randomly():
    def fn(dict_values):
        probs = {}
        num_values = len(dict_values)
        for index, _ in dict_values.items():
            probs[index] = 1.0/num_values
        return probs
    return fn

def create_distribution_boltzmann():
    def fn(dict_values):
        probs = {}
        z = sum([math.exp(x) for x in dict_values.values()])
        for index, _ in dict_values.items():
            probs[index] = math.exp(dict_values[index])/z 
        return probs
    return fn




def create_distribution_epsilon_greedily(epsilon):
    def fn(dict_values):
        probs = {}
        num_values = len(dict_values)
        for index, _ in dict_values.items():
            probs[index] = epsilon/num_values
        best_action_index = max(dict_values, key=dict_values.get)
        probs[best_action_index] += (1.0 - epsilon)
        return probs
    return fn

