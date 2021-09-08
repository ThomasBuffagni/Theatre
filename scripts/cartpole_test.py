import gym
import torch
from torch import nn

from theatre.environment_loop import TestingEnvironmentLoop

env_class = gym.make
env_args = {'id': 'CartPole-v0'}


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    test_loop = TestingEnvironmentLoop(
        env_class, env_args,
        Net, {},
        torch.load('weights')
    )

    timestep_count = test_loop.run_episode()
    print(timestep_count)
