
import os
import errno
import math 
from configparser import ConfigParser
from pathlib import Path

import numpy as np

import torch
from torch import optim
import torch.nn as nn

import logging
import sys

FORMATTER = logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s")

def get_console_handler():
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)
    return console_handler


def get_logger(logger_name):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG) # better to have too much log than not enough
    logger.addHandler(get_console_handler())

    # with this pattern, it's rarely necessary to propagate the error up to parent
    logger.propagate = False
    return logger


def gpu_id_with_max_memory():
        os.system('nvidia-smi -q -d Memory|grep -A4 GPU|grep Free > dump')
        memory_available = [int(x.split()[2]) for x in open('dump', 'r').readlines()]
        os.system('rm ./dump')
        return np.argmax(memory_available)


def config(config_file="config.ini"):
        # parser config
        config = ConfigParser()
        config.read(os.path.join(Path(__file__).parents[1], config_file))
        return config


def layer_init(layer, w_scale=1.0):
        nn.init.orthogonal_(layer.weight.data)
        layer.weight.data.mul_(w_scale)
        nn.init.constant_(layer.bias.data, 0)
        return layer
    
    # Taken from https://stackoverflow.com/a/600612/119527
def mkdir_p(path):
        try:
            os.makedirs(path)
        except OSError as exc: # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(path):
                pass
            else:
                raise

def save_checkpoint(checkpoint, agentname,checkpointpathname='checkpoint'):
        checkpoint_file_path=  os.path.join(Path(__file__).parents[1],checkpointpathname,agentname)    
        mkdir_p(checkpoint_file_path)
        checkfile=os.path.join(checkpoint_file_path,'checkpoint.pth.tar')
        torch.save(checkpoint, checkfile)
        
    

def load_checkpoint(agentname, checkpointpathname='checkpoint'):
        checkpoint_file=  os.path.join(Path(__file__).parents[1],checkpointpathname,agentname,'checkpoint.pth.tar')    
        if os.path.isfile(checkpoint_file):
            return torch.load(checkpoint_file)
        else:
            return None

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
            closure (callable, optional): A closure that reexploits the model
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
