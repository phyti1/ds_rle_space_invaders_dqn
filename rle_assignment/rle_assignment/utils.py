from typing import Tuple, Dict

import numpy as np


class LinearSchedule(object):
    def __init__(self, initial_value: float, final_value: float, schedule_steps: int) -> None:
        super().__init__()
        self._initial_value = initial_value
        self._final_value = final_value
        self._schedule_steps = schedule_steps

    def value(self, t: int):
        interpolation = min(t / self._schedule_steps, 1.0)
        return self._initial_value + interpolation * (self._final_value - self._initial_value)


class RingBuffer(object):
    def __init__(self, size, specs: Dict[str, Tuple[Tuple, np.dtype]]):
        self.size = size
        self.specs = specs
        self.buffers = {k: np.empty((size,) + tuple(shape), dtype) for k, (shape, dtype) in specs.items()}
        self.next_idx = 0
        self.num_in_buffer = 0

    def __len__(self):
        return self.num_in_buffer

    def put(self, samples: Dict[str, np.ndarray]) -> None:
        num_samples = next(iter(samples.values())).shape[0]
        for key, buffer in self.buffers.items():
            features = samples[key]
            assert features.shape[0] == num_samples
            if self.next_idx+num_samples > self.size:
                buffer[self.next_idx:] = features[:self.size-self.next_idx] # features[:self.next_idx+num_samples-self.size]
                buffer[:(self.next_idx + num_samples) % self.size] = features[self.size-self.next_idx:] # features[self.next_idx+num_samples-self.size:]
            else:
                buffer[self.next_idx:self.next_idx+num_samples] = features
        self.next_idx = (self.next_idx + num_samples) % self.size
        self.num_in_buffer = min(self.size, self.num_in_buffer + num_samples)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        idx = np.random.randint(0, self.num_in_buffer, batch_size)
        return {
            key: buffer[idx]
            for key, buffer in self.buffers.items()
        }