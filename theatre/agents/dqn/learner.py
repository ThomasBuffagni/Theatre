import ray
import time
import torch
from torch.nn import HuberLoss
from torch.nn.utils import clip_grad_norm_

from theatre.core import get
from theatre.shared_storage import set_weights_from_storage
from theatre.logger import Logger

class DQNLearner:
    def __init__(
        self,
        shared_storage,
        replay_buffer,
        network_class,
        network_args,
        batch_size,
        learning_rate,
        discount,
        importance_sampling_exponent,
        min_num_observations,
        total_training_steps,
        observations_per_step,
        target_update_period,
        huber_loss_parameter=1.0,
        max_gradient_norm=1e6
    ):
        self._shared_storage = shared_storage
        self._replay_buffer = replay_buffer

        self._min_num_observations = min_num_observations
        self._total_training_steps = total_training_steps
        self._observations_per_step = observations_per_step
        self._target_update_period = target_update_period
        self._current_training_step = 0

        self._q_network = network_class(**network_args)
        self._target_network = network_class(**network_args)

        set_weights_from_storage(self._q_network, shared_storage)
        set_weights_from_storage(self._target_network, shared_storage)

        self._batch_size = batch_size
        self._optimizer = torch.optim.Adam(
            self._q_network.parameters(),
            lr=learning_rate
        )

        if huber_loss_parameter < 0:
            raise ValueError("quadratic_linear_boundary must be >= 0.")
        self._loss_fn = HuberLoss(
            reduction='none',
            delta=huber_loss_parameter
        )
        self._max_gradient_norm = max_gradient_norm
        self._discount = discount
        self._importance_sampling_exponent = importance_sampling_exponent

        self._logger = Logger('loss.txt')

    def train(self):
        while self._should_continue_training():
            if self._should_do_next_training_step():
                print(f"\nTraining step: {self._current_training_step} / {self._total_training_steps}")
                batch_ray_id = self._replay_buffer.get_batch.remote(
                    self._batch_size)
                batch = ray.get(batch_ray_id)
                transitions, transition_ids, probabilities = batch

                td_error = self.training_step(transitions, probabilities)

                self._replay_buffer.update_priorities.remote(transition_ids, td_error)

                self._store_new_weights()

                self._current_training_step += 1

            if self._should_update_target_network():
                self._target_network.load_state_dict(
                    self._q_network.state_dict()
                )

        return self._q_network.state_dict()

    def training_step(self, transitions, probabilities):
        # https://github.com/deepmind/acme/blob/master/acme/agents/tf/dqn/learning.py#L127
        # https://github.com/deepmind/trfl/blob/master/trfl/action_value_ops.py#L87
        # https://github.com/deepmind/acme/blob/master/acme/tf/losses/huber.py#L21

        self._optimizer.zero_grad()

        # Evaluate our networks.
        observations = torch.tensor(
            transitions.observation,
            dtype=torch.float32,
            requires_grad=False
        )
        q_tm1 = self._q_network(observations)  # [batch_size, n_actions]

        next_observations = torch.tensor(transitions.next_observation, dtype=torch.float32)
        q_t_value = self._target_network(next_observations)  # [batch_size, n_actions]

        q_t_selector = self._q_network(next_observations)  # [batch_size, n_actions]

        reward_batch = torch.tensor(
            transitions.reward,
            dtype=torch.float32,
            requires_grad=False
        )
        reward_batch = torch.clip(reward_batch, -1., 1.)  # [batch_size, 1]

        discount_batch = torch.tensor(
            transitions.discount,
            dtype=torch.float32,
            requires_grad=False
        )  # [batch_size, 1]
        discount_batch *= torch.tensor(
            self._discount,
            dtype=torch.float32,
            requires_grad=False
        )

        best_action = torch.argmax(q_t_selector, dim=1).long()
        one_hot_indices = torch.nn.functional.one_hot(
            best_action, num_classes=q_t_value.size(-1)
        ).type_as(q_t_value)
        double_q_bootstrapped = torch.sum(
            q_t_value * one_hot_indices,
            dim=-1
        )

        with torch.no_grad():
            target = reward_batch + discount_batch * double_q_bootstrapped
            discount_batch = torch.unsqueeze(discount_batch, dim=-1)

        actions = torch.tensor(
            transitions.action,
            dtype=torch.int64,
            requires_grad=False
        )
        one_hot_indices = torch.nn.functional.one_hot(
            actions, num_classes=q_tm1.size(-1)
        ).float()
        qa_tm1 = torch.sum(
            q_tm1 * one_hot_indices,
            dim=-1
        )

        loss = self._loss_fn(qa_tm1, target)

        # Get the importance weights.
        importance_weights = 1. / torch.tensor(probabilities, dtype=torch.float32)  # [B]
        importance_weights = torch.pow(importance_weights, self._importance_sampling_exponent)
        importance_weights /= torch.max(importance_weights)

        # Reweight.
        loss *= importance_weights.type_as(loss)  # [B]
        loss = torch.mean(loss, dim=0)  # []
        self._logger.log(loss)
        loss.backward()

        clip_grad_norm_(self._q_network.parameters(), self._max_gradient_norm)

        self._optimizer.step()

        td_error = target - qa_tm1
        return td_error.detach().numpy()

    def _store_new_weights(self):
        self._shared_storage.set.remote(
            'weights',
            self._q_network.state_dict()
        )
        self._shared_storage.set.remote(
            'target_weights',
            self._target_network.state_dict()
        )

    def _should_wait_for_enough_observations(self):
        def enough():
            return get(
                self._shared_storage, 'num_observations'
            ) >= self._min_num_observations

        while not enough():
            time.sleep(0.1)

    def _should_continue_training(self):
        return self._current_training_step < self._total_training_steps

    def _should_do_next_training_step(self):
        n_observations = get(
            self._shared_storage, 'num_observations'
        ) - self._min_num_observations

        training_step = n_observations // self._observations_per_step

        return training_step > self._current_training_step

    def _should_update_target_network(self):
        return self._current_training_step % self._target_update_period == 0
