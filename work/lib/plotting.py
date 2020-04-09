from collections import namedtuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

EpisodeStats = namedtuple("EpisodeStats", ["algo", "episode_lengths", "episode_rewards"])

def plot_episode_stats(stats, smoothing_window=100, noshow=False):
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
