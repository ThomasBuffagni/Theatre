import ray
import torch
import numpy as np
from collections import defaultdict
from torch.nn.functional import softmax

from theatre.core import Transition
from theatre.logger import Logger

@ray.remote
class ReplayBuffer:
    """
    High-level interface to gather experience and store
    them into a Replay Buffer.

    Attributes
    ----------

    Private Attributes
    ----------
    capacity: Integer
        Maximum number of transitions stored in the buffer.
    transitions: Dictionary
        Stored transitions.
    priorities: Dictionary
        Priorities associated with the transitions.
    versions: Dictionary
        Versions associated with the transitions.
    next_index: Integer
        Index of the next transition to store.

    Methods
    -------
    reset
        Reinitialize Replay Buffer state.
    add
        Add a new transition to the buffer.
    get_transitions
        Sample random transitions according to their priorities.
    get_batch
        Sample random transitions according to their priorities
        and format them as a batch.
    update_priorities
        Update priorities of identified transitions.


    Private Methods
    ---------------

    """
    def __init__(self, capacity: int, shared_storage):
        self._capacity = capacity
        self._transitions = dict()
        self._priorities = dict()
        self._versions = defaultdict(int)
        self._next_index = 0

        self._shared_storage = shared_storage
        self._logger = Logger('replay_buffer.txt')

    def reset(self):
        """
        Reinitialize Replay Buffer state.

        Args: no args

        Returns:
            None
        """
        self._transitions = dict()
        self._priorities = dict()
        self._versions = defaultdict(int)
        self._next_index = 0

    def add(self, transition, priority: int = 1e6):
        """
        Add a new transition to the buffer.

        The buffer acts as a FIFO queue. If a new transition is
        added while the buffer is already at full capacity. The
        oldest transition is replaced. Priority and Version are
        updated accordingly.

        Args:
            transition: the transition to store in the buffer.
            priority: the priority associated with the transition.

        Returns:
            None
        """
        self._transitions[self._next_index] = transition
        self._priorities[self._next_index] = priority
        self._versions[self._next_index] += 1
        self._next_index += 1
        self._next_index %= self._capacity  # ids loop from 0 to capacity
        self._shared_storage.increment_num_observations.remote()

    def get_transitions(self, n: int):
        """
        Sample random transitions according to their priorities.

        Args:
            n (Integer): Number of transitions.

        Returns:
            selected_ids: IDs of the returned transitions.
                Might be used to update the priorities of these
                transitions.
            selected_transitions: The sampled of transitions
        """
        if n > self._capacity:
            raise ValueError(
                'Cannot sample more transitions than the buffer capacity.'
            )

        indices = range(len(self._transitions))
        weights = [self._priorities[i] for i in indices]
        probabilities = softmax(
            torch.tensor(weights, dtype=torch.float32, requires_grad=False),
            dim=0
        ).numpy()

        selected_indices = np.random.choice(
            indices, size=n, replace=False, p=probabilities
        )

        selected_ids = [
            (index, self._versions[index]) for index in selected_indices
        ]

        selected_transitions = [
            self._transitions[id_] for id_ in selected_indices
        ]

        selected_probabilities = [
            self._priorities[id_] for id_ in selected_indices
        ]

        self._logger.log(str(selected_ids))
        return selected_transitions, selected_ids, selected_probabilities

    def get_batch(self, batch_size: int):
        """
        Sample random transitions according to their priorities
        and format them as a batch.

        Args:
            batch_size (Integer): Number of transitions in
            the batch.

        Returns:
            transitions_ids: IDs of the returned transitions.
                Might be used to update the priorities of these
                transitions.
            transition_data: The batch of transitions
        """
        transitions, transition_ids, probabilities = self.get_transitions(
            batch_size
        )

        observations = [transition.observation for transition in transitions]
        actions = [transition.action for transition in transitions]
        next_observations = [
            transition.next_observation for transition in transitions
        ]
        rewards = [transition.reward for transition in transitions]
        discounts = [transition.discount for transition in transitions]

        transition_batch = Transition(
            observations, actions, rewards, discounts, next_observations, None
        )

        return transition_batch, transition_ids, probabilities

    def update_priorities(self, ids, new_priorities):
        """
        Update priorities defined by they ids if necessary.

        If the version of the currently stored transition is
        different from the version in the id, then the
        transition has been replaced. This means that there
        is no need to update this priority.

        Args:
            ids: IDs of the priorities to update.
            new_priorities: New priority values.

        Returns:
            None
        """
        for i, (index, version) in enumerate(ids):
            if self._versions[index] == version:
                self._priorities[index] = new_priorities[i]
