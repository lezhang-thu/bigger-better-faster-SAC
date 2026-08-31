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

from bbf.replay_memory import sum_tree
import jax
from jax import numpy as jnp
import numpy as np


@functools.partial(jax.jit, static_argnums=(1,), backend='cpu')
def stratified_queries(rng, n):
    """Stratum-uniform queries in [0, 1), matching the previous JAX sampler.

    CPU-pinned so converting the result to NumPy does not wait on the GPU
    train step that sampling is meant to overlap.
    """

    def one(i):
        r = jax.random.fold_in(rng, i)
        return jax.random.uniform(r, minval=i / n, maxval=(i + 1) / n)

    return jax.vmap(one)(jnp.arange(n))


class DeterministicSumTree(sum_tree.SumTree):
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

        self.highest_set = 0

        self.max_recorded_priority = 1.0

    def _total_priority(self):
        """Returns the sum of all priorities stored in this sum tree.

        Returns:
          float, sum of priorities stored in this sum tree.
    """
        return self.nodes[0]

    def _walk(self, queries):
        """Walk the numpy tree in float32; queries are in [0, 1).

        The previous JAX walk ran under default x64-off, so the stored
        float64 nodes were computed as float32. Matching that rounding
        keeps the same leaf choices.
        """
        total = np.float32(self.nodes[0])
        q = np.asarray(queries, dtype=np.float32) * total
        out = np.empty(q.shape[0], dtype=np.int64)
        nodes = self.nodes
        depth = self.depth
        for k in range(q.shape[0]):
            qq = np.float32(q[k])
            index = 0
            for _ in range(depth):
                left = index * 2 + 1
                left_sum = np.float32(nodes[left])
                if qq < left_sum:
                    index = left
                else:
                    index = left + 1
                    qq = np.float32(qq - left_sum)
            out[k] = index
        return np.minimum(out - self.low_idx, self.highest_set)

    def sample(self, rng, query_value=None):
        """Samples an element from the sum tree."""
        if query_value is None:
            query_value = stratified_queries(rng, 1)
        return int(self._walk(np.atleast_1d(query_value))[0])

    def stratified_sample(self, batch_size, rng):
        """Performs stratified sampling using the sum tree."""
        if self._total_priority() == 0.0:
            raise Exception('Cannot sample from an empty sum tree.')
        queries = stratified_queries(rng, int(batch_size))
        return self._walk(queries)

    def get(self, node_index):
        """Returns the value of the leaf node corresponding to the index.

    Args:
        node_index: The index of the leaf node.

    Returns:
        The value of the leaf node.
    """
        return self.nodes[node_index + self.low_idx]

    def reset_priorities(self):
        for i in range(self.highest_set):
            self.set(i, self.max_recorded_priority)

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
