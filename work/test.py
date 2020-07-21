from abc import abstractmethod

class Base:
    @abstractmethod
    def say(self):
        print("Say In Base")


class Derived(Base):
    def __init__(self):
        print("Init")



d = Derived()
d.say()