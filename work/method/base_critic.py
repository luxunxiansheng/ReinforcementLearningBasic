from abc import ABC, abstractmethod


class Base_Critic(ABC):

    @abstractmethod
    def evaluate(self):
        pass
        
