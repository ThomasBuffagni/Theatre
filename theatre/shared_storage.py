import ray


def set_weights_from_storage(network, storage):
    network.load_state_dict(
        ray.get(storage.get.remote('weights'))
    )


class SharedStorage:
    def __init__(self, data):
        """

        Args:
            data (Dict[str, Any]): The data shared between agent components.
        """
        self._data = data

    def get(self, key):
        """
        Retrieve an element from the storage identified by its key.

        Args:
            key (str): An element identifier.

        Returns:
            The element in the storage identified by the key.
        """
        return self._data[key]

    def set(self, key, value):
        if self.exists(key) is False:
            raise KeyError(
                f'Shared Storage has not been initialized with the key \'{key}\'. Cannot set its value to {value}.'
            )
        self._data[key] = value

    def exists(self, key):
        return key in self._data
