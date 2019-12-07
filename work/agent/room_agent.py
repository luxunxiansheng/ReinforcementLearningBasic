import sys
import numpy as np

from algorithm.implicity_policy import dynamic_programming
from agent.base_agent import Base_Agent
from env.room import RoomEnv

class Room_Agent(Base_Agent):
    def __init__(self):
        self.env = RoomEnv()
    
    
    def make_decision(self):
        pass