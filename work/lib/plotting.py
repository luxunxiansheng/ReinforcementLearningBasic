from collections import namedtuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

QValue = namedtuple('QValue',['x_name','y_name','q_function'])

EpisodeStats = namedtuple("EpisodeStats", ["algo", "episode_lengths", "episode_rewards",'q_value'])

StateValues = namedtuple("StateValues", ['appr_method','state_value'])



def plot_episode_stats(stats, smoothing_window=10, noshow=False):
    # Plot the episode length over time
    fig1 = plt.figure(figsize=(10, 5))
   
    for index in range(len(stats)):
        plt.plot(stats[index].episode_lengths, label=stats[index].algo)
    
    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Time")
    if noshow:
        plt.close(fig1)
    else:
        plt.show(fig1)

    # Plot the episode reward over time
    fig2 = plt.figure(figsize=(10, 5))

    for index in range(len(stats)):
        plt.plot(pd.Series(stats[index].episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean(),label=stats[index].algo)

    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
    if noshow:
        plt.close(fig2)
    else:
        plt.show(fig2)

    # Plot time steps and episode number
    fig3 = plt.figure(figsize=(10, 5))

    for index in range(len(stats)):
        plt.plot(np.cumsum(stats[index].episode_lengths),np.arange(len(stats[0].episode_lengths)),label=stats[index].algo)

    plt.legend()
    plt.xlabel("Time Steps")
    plt.ylabel("Episode")
    plt.title("Episode per time step")
    if noshow:
        plt.close(fig3)
    else:
        plt.show(fig3)

    return fig1, fig2, fig3


def  plot_state_value(env,value_function_stats,noshow=False):
    # Plot the episode length over time
    fig1 = plt.figure(figsize=(10, 5))
    
    for index in range(len(value_function_stats)):
        state_value = np.asarray([value_function_stats[index].state_value.value(state_index/env.nS) for state_index in range(env.nS)])
        plt.plot(state_value,label=value_function_stats[index].appr_method)

    state_value = np.asarray([ state_index/env.nS for state_index in range(env.nS)])
    plt.plot(state_value,label='True Value')

    plt.legend()
    plt.xlabel("State_index")
    plt.ylabel("Value")
    plt.title("Approximation of state value ")
    if noshow:
        plt.close(fig1)
    else:
        plt.show(fig1) 


def plot_2d_q_value(env, stats, noshow = False):
    
    fig1 = plt.figure(figsize=(10, 5))
    ax = fig1.add_subplot(111, projection='3d')
    grid_size = 40
   
    
        
    xs = np.linspace(env.observation_space.low[0],env.observation_space.high[0], grid_size)
    ys = np.linspace(env.observation_space.low[1],env.observation_space.high[1], grid_size)

    for index in range(len(stats)):
        axis_x = []
        axis_y = []
        axis_z = []
        for x in xs:
            for y in ys:
                axis_x.append(x)
                axis_y.append(y)
                axis_z.append(cost_to_go(stats[index].q_value.q_function,[x,y],env.action_space))
        ax.scatter(axis_x, axis_y, axis_z)
            
    ax.set_xlabel(stats[0].q_value.x_name)
    ax.set_ylabel(stats[0].q_value.y_name)
    ax.set_zlabel('cost to go')
      
    if noshow:
        plt.close(fig1)
    else:
        plt.show(fig1) 



# get # of steps to reach the goal under current state value function
def cost_to_go(value_function,state,actions):
    costs = []
    for action in range(actions.n):
        costs.append(value_function.value(state, action))
    return -np.max(costs)   