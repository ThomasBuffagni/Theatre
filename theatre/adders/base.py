import abc


class BaseAdder(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def reset(self):
        """
            Reset the state of the Adder.
        """

    @abc.abstractmethod
    def add_first(self, observation):
        """
            Add the first observation to the Adder.
        """

    @abc.abstractmethod
    def add(self, action, next_observation, reward, discount, done):
        """
            Add a new
        """