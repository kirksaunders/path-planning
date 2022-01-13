from abc import ABC, abstractmethod

class ReplayBuffer(ABC):
    """
    Abstract replay buffer storage for past experiences of an agent.
    """

    @abstractmethod
    def add(self, experience):
        pass

    @abstractmethod
    def mini_batch(self):
        pass