from abc import ABC, abstractmethod

class Environment(ABC):
    """
    Abstract class for an agent training environment.
    """

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def display(self):
        pass