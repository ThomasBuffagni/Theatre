import ray
from collections import namedtuple

Transition = namedtuple(
    'Transition',
    [
        'observation', 'action', 'reward',
        'discount', 'next_observation', 'done'
    ]
)


class Actor:
    def __init__(self):
        pass


class Learner:
    def __init__(self):
        pass


class Agent:
    def __init__(self):
        pass


def get(shared_storage, key):
    return ray.get(shared_storage.get.remote(key))
