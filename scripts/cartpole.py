import gym
import torch
from torch import nn

from theatre.agents.dqn.agent import DQNAgent

env_class = gym.make
env_args = {'id': 'CartPole-v1'}


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # def block(in_features, out_features):
        #     return [
        #         nn.Linear(
        #             in_features=in_features,
        #             out_features=out_features
        #         ),
        #         nn.ReLU(),
        #         nn.Dropout(0.5)
        #     ]

        self.model = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.model(x)


exploration_config = {
    'start_value': 0.5,
    'end_value': 0.001,
    'end_timestep': 300_000
}


if __name__ == '__main__':
    agent = DQNAgent(
        Net, {},
        env_class, env_args,
        1, 64, 100, 3, 100,
        exploration_config,
        observations_per_step=1000,
        min_num_observations=0
    )

    trained_state_dict = agent.train()

    torch.save(trained_state_dict, 'weights')
