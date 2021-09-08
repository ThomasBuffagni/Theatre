import random
import torch
from gym import spaces

from theatre.exploration import EpsilonPolicy
from theatre.shared_storage import set_weights_from_storage


class FeedForwardfActor:
    def __init__(
            self,
            network_class,
            network_args,
            action_space,
            exploration_config,
            adder_class,
            adder_args,
            shared_storage,
            n_timesteps_per_update,
            initial_timestep_count=0
    ):
        self._timestep_count = initial_timestep_count
        self._policy_network = network_class(**network_args)
        set_weights_from_storage(self._policy_network, shared_storage)
        self._policy_network.train()

        self._adder = adder_class(**adder_args)
        self._shared_storage = shared_storage
        self._n_timesteps_per_update = n_timesteps_per_update
        self._n_timesteps = 0

        self._epsilon = EpsilonPolicy(**exploration_config)

        if isinstance(action_space, spaces.Discrete):
            self._n_actions = action_space.n
        else:
            raise ValueError('Can only handle Discrete action spaces.')

    def select_action(self, observation):
        r = random.random()

        if r < self._epsilon.get(self._timestep_count):
            action = random.randint(0, self._n_actions-1)
        else:
            observation = torch.tensor(observation, dtype=torch.float32)
            batched_obs = torch.unsqueeze(observation, dim=0)
            action_values = self._policy_network(batched_obs).detach().numpy()[0]
            action = action_values.argmax().item()

        self._timestep_count += 1
        return action

    def observe_first(self, observation):
        self._adder.add_first(observation)

    def observe(self, action, observation, reward, done, info):
        self._n_timesteps += 1
        self._adder.add(
            action,
            observation,
            reward,
            info.get('discount', 0),
            done
        )

    def update(self):
        if self._n_timesteps % self._n_timesteps_per_update == 0:
            set_weights_from_storage(
                self._policy_network,
                self._shared_storage
            )
