import numpy as np
import math

import torch
from torch import optim
from torch._C import Value 


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


class SharedAdam(optim.Adam):
    """Implements Adam algorithm with shared states.
    """
    def __init__(self,
                params,
                lr=1e-3,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=0):
        super(SharedAdam, self).__init__(params, lr, betas, eps, weight_decay)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1)
                state['exp_avg'] = p.data.new().resize_as_(p.data).zero_()
                state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_()

    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'].share_memory_()
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad,value=1 - beta2)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1**state['step'].item()
                bias_correction2 = 1 - beta2**state['step'].item()
                step_size = group['lr'] * math.sqrt(
                    bias_correction2) / bias_correction1

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

