# coding=utf-8
# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A sum tree data structure that uses JAX for controlling randomness."""

import functools

import jax
import numpy as np


@functools.partial(jax.jit, backend='cpu', static_argnums=(1,))
def stratified_offsets(rng, batch_size):
    """Draws one reproducible unit offset per stratum."""
    return jax.random.uniform(rng, shape=(batch_size,))


class DeterministicSumTree(object):
    """A sum tree data structure for storing replay priorities.

    In contrast to the original implementation, this uses JAX for handling
    randomness, which allows us to reproduce the same results when using the
    same
    seed.
  """

    def __init__(self, capacity):
        """Creates the sum tree data structure for the given replay capacity.

    Args:
      capacity: int, the maximum number of elements that can be stored in
        this data structure.

    Raises:
      ValueError: If requested capacity is not positive.
    """
        assert isinstance(capacity, int)
        if capacity <= 0:
            raise ValueError(
                'Sum tree capacity should be positive. Got: {}'.format(
                    capacity))

        self.nodes = []
        self.depth = int(np.ceil(np.log2(capacity)))
        self.low_idx = (2**self.depth) - 1  # pri_idx + low_idx -> tree_idx
        self.high_idx = capacity + self.low_idx
        self.nodes = np.zeros(2**(self.depth + 1) - 1)  # Double precision.
        self.capacity = capacity

        # -1 distinguishes an empty tree from a tree whose leaf zero is set.
        self.highest_set = -1

        self.max_recorded_priority = 1.0

    def _total_priority(self):
        """Returns the sum of all priorities stored in this sum tree.

        Returns:
          float, sum of priorities stored in this sum tree.
    """
        return self.nodes[0]

    def _find_indices(self, query_values):
        """Traverses the host-resident tree for a vector of priority queries."""
        query_values = np.asarray(query_values, dtype=np.float64).reshape(-1)
        indices = np.zeros(query_values.shape, dtype=np.int64)
        for _ in range(self.depth):
            left_children = 2 * indices + 1
            left_sums = self.nodes[left_children]
            take_right = query_values >= left_sums
            query_values = query_values - take_right * left_sums
            indices = left_children + take_right.astype(np.int64)
        return np.minimum(indices - self.low_idx, self.highest_set)

    def sample(self, rng, query_value=None):
        """Samples an element from the sum tree."""
        total_priority = self._total_priority()
        if total_priority == 0.0:
            raise ValueError('Cannot sample from an empty sum tree.')
        fraction = (float(np.asarray(jax.random.uniform(rng)))
                    if query_value is None else float(query_value))
        if fraction < 0.0 or fraction > 1.0:
            raise ValueError('query_value must be in [0, 1].')
        return self._find_indices([fraction * total_priority])[0]

    def stratified_sample(self, batch_size, rng):
        """Performs stratified sampling using the sum tree."""
        batch_size = int(batch_size)
        if batch_size < 1:
            raise ValueError('batch_size must be positive.')
        total_priority = self._total_priority()
        if total_priority == 0.0:
            raise ValueError('Cannot sample from an empty sum tree.')
        offsets = np.asarray(
            stratified_offsets(rng, batch_size), dtype=np.float64)
        # Do the stratum arithmetic in float64 on the host. In float32 the
        # final value can round to exactly 1.0, which would bias a padded tree
        # toward its last populated leaf.
        fractions = (
            np.arange(batch_size, dtype=np.float64) + offsets) / batch_size
        return self._find_indices(fractions * total_priority)

    def get(self, node_index):
        """Returns the value of the leaf node corresponding to the index.

    Args:
        node_index: The index of the leaf node.

    Returns:
        The value of the leaf node.
    """
        return self.nodes[node_index + self.low_idx]

    def reset_priorities(self):
        """Makes every populated leaf equal and rebuilds the tree in O(N).

        Calling ``set`` for every leaf performed O(N log N) Python updates.
        A reset discards all relative priority information, so filling the
        contiguous populated leaf prefix and reducing each tree level is both
        equivalent and substantially cheaper.
        """
        if self.highest_set < 0:
            return
        self.nodes.fill(0.0)
        populated = self.highest_set + 1
        self.nodes[self.low_idx:self.low_idx + populated] = (
            self.max_recorded_priority)
        for level in range(self.depth - 1, -1, -1):
            start = (2**level) - 1
            end = (2**(level + 1)) - 1
            self.nodes[start:end] = (
                self.nodes[2 * start + 1:2 * end + 1:2] +
                self.nodes[2 * start + 2:2 * end + 2:2])

    def set(self, node_index, value):
        """Sets the value of a leaf node and updates internal nodes accordingly.

    This operation takes O(log(capacity)).
    Args:
        node_index: int, the index of the leaf node to be updated.
        value: float, the value which we assign to the node. This value must
          be nonnegative. Setting value = 0 will cause the element to never
          be sampled.

    Raises:
        ValueError: If the given value is negative.
    """
        if value < 0.0:
            raise ValueError(
                'Sum tree values should be nonnegative. Got {}'.format(value))
        self.highest_set = max(node_index, self.highest_set)
        node_index = node_index + self.low_idx
        self.max_recorded_priority = max(value, self.max_recorded_priority)

        delta_value = value - self.nodes[node_index]

        # Now traverse back the tree, adjusting all sums along the way.
        for _ in reversed(range(self.depth)):
            # Note: Adding a delta leads to some tolerable numerical inaccuracies.
            self.nodes[node_index] += delta_value
            node_index = (node_index - 1) // 2

        self.nodes[node_index] += delta_value
        assert node_index == 0, ('Sum tree traversal failed, final node index '
                                 'is not 0.')
