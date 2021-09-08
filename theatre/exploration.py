from theatre.logger import Logger


class EpsilonPolicy:
    def __init__(
        self,
        start_value,
        end_value,
        end_timestep,
        method='linear',
        decay=None
    ):
        method = method.lower()
        if method not in ['linear', 'exponential']:
            raise ValueError(
                f"Method {method} unknown."
                f" Please use 'linear' or 'exponential'."
            )

        self._start_value = start_value
        self._end_value = end_value
        self._end_timestep = end_timestep
        self._method = method
        self._decay = decay
        self._linear_coeff = (end_value - start_value) / end_timestep

        self._logger = Logger('exploration.txt')

    def get(self, timestep_number):
        if timestep_number < 0:
            raise ValueError('Timestep number cannot be lower than 0.')

        if timestep_number >= self._end_timestep:
            epsilon = self._end_value
        else:
            if self._method == 'linear':
                epsilon = self.linear(timestep_number)
            else:
                raise ValueError('Exploration: Can only handle linear decay.')

        self._logger.log(epsilon)
        return epsilon

    def linear(self, timestep_number):
        return self._linear_coeff * timestep_number + self._start_value
