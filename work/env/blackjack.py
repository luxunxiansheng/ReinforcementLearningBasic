# #### BEGIN LICENSE BLOCK #####
# Version: MPL 1.1/GPL 2.0/LGPL 2.1
#
# The contents of this file are subject to the Mozilla Public License Version
# 1.1 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
# http://www.mozilla.org/MPL/
#
# Software distributed under the License is distributed on an "AS IS" basis,
# WITHOUT WARRANTY OF ANY KIND, either express or implied. See the License
# for the specific language governing rights and limitations under the
# License.
#
#
# Contributor(s):
#
#    Bin.Li (ornot2008@yahoo.com)
#
#
# Alternatively, the contents of this file may be used under the terms of
# either the GNU General Public License Version 2 or later (the "GPL"), or
# the GNU Lesser General Public License Version 2.1 or later (the "LGPL"),
# in which case the provisions of the GPL or the LGPL are applicable instead
# of those above. If you wish to allow use of your version of this file only
# under the terms of either the GPL or the LGPL, and not to allow others to
# use your version of this file under the terms of the MPL, indicate your
# decision by deleting the provisions above and replace them with the notice
# and other provisions required by the GPL or the LGPL. If you do not delete
# the provisions above, a recipient may use your version of this file under
# the terms of any one of the MPL, the GPL or the LGPL.
#
# #### END LICENSE BLOCK #####
#
# /

from collections import defaultdict

from gym import spaces
from gym.utils import seeding

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


from env.base_discrete_env import PureDiscreteEnv


def cmp(a, b):
    return float(a > b) - float(a < b)


# 1 = Ace, 2-10 = Number cards, Jack/Queen/King = 10
deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]


def draw_card(np_random):
    return int(np_random.choice(deck))


def draw_hand(np_random):
    return [draw_card(np_random), draw_card(np_random)]


def usable_ace(hand):  # Does this hand have a usable ace?
    return 1 in hand and sum(hand) + 10 <= 21


def sum_hand(hand):  # Return current hand total
    if usable_ace(hand):
        return sum(hand) + 10
    return sum(hand)


def is_bust(hand):  # Is this hand a bust?
    return sum_hand(hand) > 21


def score(hand):  # What is the score of this hand (0 if bust)
    return 0 if is_bust(hand) else sum_hand(hand)


def is_natural(hand):  # Is this hand a natural blackjack?
    return sorted(hand) == [1, 10]


class BlackjackEnv(PureDiscreteEnv):
    """Simple blackjack environment

    Blackjack is a card game where the goal is to obtain cards that sum to as
    near as possible to 21 without going over.  They're playing against a fixed
    dealer.
    Face cards (Jack, Queen, King) have point value 10.
    Aces can either count as 11 or 1, and it's called 'usable' at 11.

    This game is placed with an infinite deck (or with replacement).
    The game starts with each (player and dealer) having one face up and one
    face down card.

    The player can request additional cards (hit=1) until they decide to stop
    (stick=0) or exceed 21 (bust).

    After the player sticks, the dealer reveals their facedown card, and draws
    until their sum is 17 or greater.  If the dealer goes bust the player wins.

    If neither player nor dealer busts, the outcome (win, lose, draw) is
    decided by whose sum is closer to 21.  The reward for winning is +1,
    drawing is 0, and losing is -1.

    The observation of a 3-tuple of: the players current sum, 
    the dealer's one showing card (1-10 where 1 is ace),
    and whether or not the player holds a usable ace (0 or 1).

    This environment corresponds to the version of the blackjack problem
    described in Example 5.1 in Reinforcement Learning: An Introduction
    by Sutton and Barto.

    """
    STICK = 0
    HIT = 1

    def __init__(self, random_start=True,natural=False):

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(33),
            spaces.Discrete(11),
            spaces.Discrete(2)))

        self.seed()

        # Flag to payout 1.5 on a "natural" blackjack win, like casino rules
        # Ref: http://www.bicyclecards.com/how-to-play/blackjack/
        self.natural = natural
        # Start the first game

        self.random_start = random_start

        self.reset()


    def build_V_table(self):
        v_table = {}
        for sum_index in range(self.observation_space.spaces[0].n):
            for showcard_index in range(self.observation_space.spaces[1].n):
                for usable_ace_index in range(self.observation_space.spaces[2].n):
                    v_table[(sum_index, showcard_index, usable_ace_index)] = 0.0

        return v_table

    def build_Q_table(self):
        q_table = defaultdict(lambda: {})
        for sum_index in range(self.observation_space.spaces[0].n):
            for showcard_index in range(self.observation_space.spaces[1].n):
                for usable_ace_index in range(self.observation_space.spaces[2].n):
                    for action_index in range(self.action_space.n):
                        q_table[((sum_index, showcard_index,usable_ace_index))][action_index] = 0.0
        return q_table

    def build_policy_table(self):
        policy_table = defaultdict(lambda: {})
        for sum_index in range(self.observation_space.spaces[0].n):
            for showcard_index in range(self.observation_space.spaces[1].n):
                for usable_ace_index in range(self.observation_space.spaces[2].n):
                    for action_index in range(self.action_space.n):
                        policy_table[((sum_index, showcard_index, usable_ace_index))][action_index] = 1.0/self.action_space.n

        return policy_table

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)
        if action:  # hit: add a card to players hand and return
            self.player.append(draw_card(self.np_random))
            if is_bust(self.player):
                done = True
                reward = -1
            else:
                done = False
                reward = 0
        else:  # stick: play out the dealers hand, and score
            done = True
            while sum_hand(self.dealer) < 17:
                self.dealer.append(draw_card(self.np_random))
            reward = cmp(score(self.player), score(self.dealer))
            if self.natural and is_natural(self.player) and reward == 1:
                reward = 1.5

        return self._get_obs(done), reward, done, {}

    def _get_obs(self,done):
        return (22 if done else sum_hand(self.player), self.dealer[0], usable_ace(self.player))

    def reset(self):
        if self.random_start:
            self.dealer = draw_hand(self.np_random)
            self.player = draw_hand(self.np_random)
        else:
            self.dealer = [2, draw_card(self.np_random)]
            self.player = [1, 2]
        return self._get_obs(False)

    def show_policy(self, table_policy):
        ace_usable = np.zeros([33, 11])
        ace_no_usable = np.zeros([33, 11])

        for state, action_values in table_policy.policy_table.items():
            summary = state[0]
            showcard = state[1]
            ace = state[2]
            if ace:
                ace_usable[summary][showcard] = min(
                    action_values, key=lambda k: action_values[k])
            else:
                ace_no_usable[summary][showcard] = min(
                    action_values, key=lambda k: action_values[k])

        fig, axes = plt.subplots(1, 2, figsize=(16, 12))
        plt.subplots_adjust(wspace=0.1, hspace=0.2)
        axes = axes.flatten()

        fig = sns.heatmap(np.flipud(ace_usable), cmap="YlGnBu", ax=axes[0], xticklabels=range(0, 11), yticklabels=list(reversed(range(1, 33))))
        fig.set_ylabel('player sum', fontsize=30)
        fig.set_xlabel('dealer showing', fontsize=30)
        fig.set_title('Usable Ace', fontsize=30)

        fig = sns.heatmap(np.flipud(ace_no_usable), cmap="YlGnBu", ax=axes[1], xticklabels=range(1, 11),yticklabels=list(reversed(range(1, 33))))
        fig.set_ylabel('player sum', fontsize=30)
        fig.set_xlabel('dealer showing', fontsize=30)
        fig.set_title('No Usable ace', fontsize=30)

        plt.show()
