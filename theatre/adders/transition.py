from theatre.core import Transition

from .base import BaseAdder
from collections import deque
from itertools import islice


class NStepTransitionAdder(BaseAdder):
    def __init__(self, replay_buffer, n_steps):
        super().__init__()
        self.n = n_steps
        self._replay_buffer = replay_buffer
        self._local_buffer = deque(maxlen=n_steps)
        self._last_observation = None

    def reset(self):
        self._local_buffer.clear()
        self._last_observation = None

    def add_first(self, observation):
        if self._last_observation is not None:
            raise ValueError('You must reset the adder before calling add_first.')
        self._last_observation = observation

    def add(self, action, next_observation, reward, discount, done):
        if self._last_observation is None:
            raise ValueError('You must call add_first to record the first observation.')

        one_step_transition = Transition(
            observation=self._last_observation,
            action=action,
            next_observation=next_observation,
            reward=reward,
            discount=discount,
            done=done
        )
        self._local_buffer.append(one_step_transition)
        n_step_transition = self._build_n_step_transition()
        self._replay_buffer.add.remote(n_step_transition)

        if done:
            for start in range(1, self.n):
                n_step_transition = self._build_n_step_transition(start=start)
                self._replay_buffer.add.remote(n_step_transition)
            self.reset()
        else:
            self._last_observation = next_observation

    def _build_n_step_transition(self, start=0):
        observation = self._local_buffer[start].observation
        action = self._local_buffer[start].action
        next_observation = self._local_buffer[-1].next_observation
        discount = self._local_buffer[start].discount
        discounted_reward = self._local_buffer[start].reward * discount

        done = False
        for t in islice(self._local_buffer, start + 1, None):
            discount *= t.discount
            discounted_reward += t.reward * discount
            done = t.done

        return Transition(
            observation=observation,
            action=action,
            next_observation=next_observation,
            reward=discounted_reward,
            discount=1,  # reward is already discounted
            done=done
        )


class TransitionAdder(NStepTransitionAdder):
    def __init__(self, replay_buffer):
        super().__init__(replay_buffer, n_steps=1)
