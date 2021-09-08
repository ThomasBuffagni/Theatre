import ray

import theatre.core as core
from theatre.environment_loop import TrainingEnvironmentLoop
from theatre.actors import FeedForwardfActor
from theatre.adders.transition import NStepTransitionAdder
from theatre.agents.dqn.shared_storage import DQNSharedStorage
from theatre.agents.dqn.learner import DQNLearner
from theatre.replay_buffer import ReplayBuffer

DQN_DEFAULT_STATE = {
    'done': False,
    'num_observations': 0,
    'current_training_step': 0,
    'episode_count': 0,
    'weights': None,
    'target_weights': None
}


class DQNAgent(core.Agent):
    """
    A DQN Agent

    Contains all necessary code to train a DQN Agent
    in a distributed fashion. It is composed of a DQN
    Learner, many DQN Actors, a Replay Buffer, a Shared
    Storage which are all 'ray actors'.

    """
    def __init__(
        self,
        network_class,
        network_args,
        env_class,
        env_args,
        n_actors,
        batch_size,
        n_timesteps_per_update,
        n_transition_step,
        n_training_step,
        exploration_config,
        min_num_observations=100,
        observations_per_step=2,
        learning_rate=0.001,
        discount=0.99,
        importance_sampling_exponent=0.2,
        target_update_period=100,
        huber_loss_parameter=1.0,
        max_gradient_norm=1e10,
        replay_buffer_capacity=1_000_000
    ):
        """
        Creates a DQN Agent
        """
        super().__init__()

        self._n_actors = n_actors

        self._shared_storage_state = DQN_DEFAULT_STATE
        self._shared_storage_state.update({
            'total_training_step': n_training_step,
            'min_num_observations': min_num_observations,
            'observations_per_step': observations_per_step
        })

        self._replay_buffer_capacity = replay_buffer_capacity

        self._network_class = network_class
        self._network_args = network_args
        self._env_class = env_class
        self._env_args = env_args

        # Learner Parameters
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._importance_sampling_exponent = importance_sampling_exponent
        self._target_update_period = target_update_period
        self._huber_loss_parameter = huber_loss_parameter
        self._max_gradient_norm = max_gradient_norm
        self._discount = discount

        # Actor Parameters
        self._n_timesteps_per_update = n_timesteps_per_update
        self._n_transition_step = n_transition_step
        self._exploration_config = exploration_config

    def train(self):
        """
        Trains the Apex Agent

        """
        ray.init()

        # Initialize shared storage with network weights
        network = self._network_class(**self._network_args)
        self._shared_storage_state.update({
            'weights': network.state_dict(),
            'target_weights': network.state_dict()
        })
        shared_storage = DQNSharedStorage.remote(
            self._shared_storage_state
        )

        replay_buffer = ReplayBuffer.remote(
            self._replay_buffer_capacity,
            shared_storage,
        )

        actor_args = {
            'network_class': self._network_class,
            'network_args': self._network_args,
            'action_space': self._env_class(**self._env_args).action_space,
            'exploration_config': self._exploration_config,
            'adder_class': NStepTransitionAdder,
            'adder_args': {
                'replay_buffer': replay_buffer,
                'n_steps': self._n_transition_step
            },
            'shared_storage': shared_storage,
            'n_timesteps_per_update': self._n_timesteps_per_update
        }

        env_loops = [
            TrainingEnvironmentLoop.remote(
                self._env_class,
                self._env_args,
                FeedForwardfActor,
                actor_args,
                shared_storage,
            ) for _ in range(self._n_actors)
        ]

        learner = DQNLearner(
            shared_storage,
            replay_buffer,
            self._network_class,
            self._network_args,
            self._batch_size,
            self._learning_rate,
            self._discount,
            self._importance_sampling_exponent,
            self._shared_storage_state['min_num_observations'],
            self._shared_storage_state['total_training_step'],
            self._shared_storage_state['observations_per_step'],
            self._target_update_period,
            self._huber_loss_parameter,
            self._max_gradient_norm
        )

        for env_loop in env_loops:
            env_loop.run.remote()

        trained_state_dict = learner.train()

        shared_storage.set.remote('done', True)

        ray.shutdown()

        return trained_state_dict
