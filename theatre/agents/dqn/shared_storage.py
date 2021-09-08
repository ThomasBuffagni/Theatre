import ray
from theatre.shared_storage import SharedStorage


@ray.remote
class DQNSharedStorage(SharedStorage):
    def increment(self, key, n):
        self.set(key, self.get(key) + n)

    def increment_training_step(self, n=1):
        self.increment('current_training_step', n)

    def increment_num_observations(self, n=1):
        self.increment('num_observations', n)

    def increment_episode_count(self, n=1):
        self.increment('episode_count', n)

    def get_state(self):
        return self._data
