import ray
import torch

from theatre.logger import Logger

"""
Deepmind's environment loop also implements a logger and a counter...
"""

@ray.remote
class TrainingEnvironmentLoop:
    def __init__(
        self,
        env_class,
        env_args,
        actor_class,
        actor_args,
        shared_storage,
    ):
        self._environment = env_class(**env_args)
        self._actor = actor_class(**actor_args)
        self._shared_storage = shared_storage

        self._action_logger = Logger('actions.txt')
        self._reward_logger = Logger('reward.txt')

    def run(self):
        episode_count = 0

        while not ray.get(self._shared_storage.get.remote('done')):
            self.run_episode()
            episode_count += 1

    def run_episode(self):
        actions = []
        total_reward = 0
        observation = self._environment.reset()
        self._actor.observe_first(observation)
        done = False

        while not done:
            action = self._actor.select_action(observation)

            observation, reward, done, info = self._environment.step(action)

            self._actor.observe(action, observation, reward, done, info)
            actions.append(action)
            total_reward += reward

            self._actor.update()

        self._environment.close()
        self._action_logger.log(actions)
        self._reward_logger.log(total_reward)
        return self._actor._timestep_count


class TestingEnvironmentLoop:
    def __init__(
        self,
        env_class,
        env_args,
        network_class,
        network_args,
        network_state_dict
    ):
        self._environment = env_class(**env_args)
        self._network = network_class(**network_args)
        self._network.load_state_dict(network_state_dict)
        self._network.eval()

    def run_episode(self):
        timestep_count = 0
        observation = self._environment.reset()
        done = False

        while not done:
            self._environment.render()
            action = self.select_action(observation)
            observation, reward, done, info = self._environment.step(action)
            timestep_count += 1

        self._environment.close()
        return timestep_count

    def select_action(self, observation):
        observation = torch.tensor(observation, dtype=torch.float32)
        batched_obs = torch.unsqueeze(observation, dim=0)
        action_values = self._network(batched_obs).detach().numpy()[0]
        action = action_values.argmax().item()
        return action
