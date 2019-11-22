from abc import ABC, abstractmethod

class Base_Policy(ABC):
    @abstractmethod
    def __call__(self, observation):
        pass
 



    

        

