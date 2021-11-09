from abc import ABC, abstractmethod

class ReplayBuffer(ABC):
    @abstractmethod
    def add(self, experience):
        pass

    @abstractmethod
    def mini_batch(self):
        pass