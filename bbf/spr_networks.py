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
"""Various networks for Jax Dopamine SPR agents."""

import collections
import enum
import functools
import time
from typing import Any, Sequence, Tuple

from absl import logging
from flax import linen as nn
import gin
import jax
from jax import random
import jax.numpy as jnp
import numpy as onp

SPROutputType = collections.namedtuple(
    'RL_network',
    ['q_values', 'logits', 'probabilities', 'latent', 'representation'],
)
PRNGKey = Any
Array = Any
Shape = Tuple[int]
Dtype = Any


def _absolute_dims(rank, dims):
    return tuple([rank + dim if dim < 0 else dim for dim in dims])


# --------------------------- < Data Augmentation > -----------------------------


def _random_crop(key, img, cropped_shape):
    """Random crop an image."""
    _, width, height = cropped_shape[:-1]
    key_x, key_y = random.split(key, 2)
    x = random.randint(key_x, shape=(), minval=0, maxval=img.shape[1] - width)
    y = random.randint(key_y, shape=(), minval=0, maxval=img.shape[2] - height)
    return img[:, x:x + width, y:y + height]


def _crop_with_indices(img, x, y, cropped_shape):
    cropped_image = jax.lax.dynamic_slice(img, [x, y, 0], cropped_shape[1:])
    return cropped_image


def _per_image_random_crop(key, img, cropped_shape):
    """Random crop an image."""
    batch_size, width, height = cropped_shape[:-1]
    key_x, key_y = random.split(key, 2)
    x = random.randint(key_x,
                       shape=(batch_size,),
                       minval=0,
                       maxval=img.shape[1] - width)
    y = random.randint(key_y,
                       shape=(batch_size,),
                       minval=0,
                       maxval=img.shape[2] - height)
    return jax.vmap(_crop_with_indices, in_axes=(0, 0, 0, None))(img, x, y,
                                                                 cropped_shape)


def _intensity_aug(key, x, scale=0.05):
    """Follows the code in Schwarzer et al. (2020) for intensity augmentation."""
    r = random.normal(key, shape=(x.shape[0], 1, 1, 1))
    noise = 1.0 + (scale * jnp.clip(r, -2.0, 2.0))
    return x * noise


@jax.jit
def drq_image_aug(key, obs, img_pad=4):
    """Padding and cropping for DrQ."""
    flat_obs = obs.reshape(-1, *obs.shape[-3:])
    paddings = [(0, 0), (img_pad, img_pad), (img_pad, img_pad), (0, 0)]
    cropped_shape = flat_obs.shape
    # The reference uses ReplicationPad2d in pytorch, but it is not available
    # in Jax. Use 'edge' instead.
    flat_obs = jnp.pad(flat_obs, paddings, 'edge')
    key1, key2 = random.split(key, num=2)
    cropped_obs = _per_image_random_crop(key2, flat_obs, cropped_shape)
    # cropped_obs = _random_crop(key2, flat_obs, cropped_shape)
    aug_obs = _intensity_aug(key1, cropped_obs)
    return aug_obs.reshape(*obs.shape)


# --------------------------- < RainbowNetwork >--------------------------------
class FeatureLayer(nn.Module):
    """Layer encapsulating a standard linear layer.

  Attributes:
    net: The layer (nn.Module).
    features: Output size.
    dtype: Dtype (float32 | float16 | bfloat16)
    initializer: Jax initializer.
  """
    features: int
    dtype: Dtype = jnp.float32
    initializer: Any = nn.initializers.xavier_uniform()

    def setup(self):
        self.net = nn.Dense(
            self.features,
            kernel_init=self.initializer,
            dtype=self.dtype,
        )

    def __call__(self, x, eval_mode):
        return self.net(x)


class LinearHead(nn.Module):
    """A linear DQN head supporting dueling networks.

  Attributes:
    advantage: Advantage layer.
    value: Value layer.
    num_actions: int, size of action space.
    num_atoms: int, number of value prediction atoms per action.
    dtype: Jax dtype.
    initializer: Jax initializer.
  """
    num_actions: int
    num_atoms: int
    dtype: Dtype = jnp.float32
    initializer: Any = nn.initializers.xavier_uniform()

    def setup(self):
        self.advantage = FeatureLayer(
            self.num_actions * self.num_atoms,
            dtype=self.dtype,
            initializer=self.initializer,
        )
        self.value = FeatureLayer(
            self.num_atoms,
            dtype=self.dtype,
            initializer=self.initializer,
        )

    def __call__(self, x, eval_mode):
        adv = self.advantage(x, eval_mode)
        value = self.value(x, eval_mode)
        adv = adv.reshape((self.num_actions, self.num_atoms))
        value = value.reshape((1, self.num_atoms))
        logits = value + (adv - (jnp.mean(adv, -2, keepdims=True)))
        return logits


def process_inputs(x, data_augmentation=False, rng=None, dtype=jnp.float32):
    """Input normalization and if specified, data augmentation."""

    if dtype == 'float32':
        dtype = jnp.float32
    elif dtype == 'float16':
        dtype = jnp.float16
    elif dtype == 'bfloat16':
        dtype = jnp.bfloat16

    out = x.astype(dtype) / 255.0
    if data_augmentation:
        if rng is None:
            raise ValueError('Pass rng when using data augmentation')
        out = drq_image_aug(rng, out)
    return out


def renormalize(tensor, has_batch=False):
    shape = tensor.shape
    if not has_batch:
        tensor = jnp.expand_dims(tensor, 0)
    tensor = tensor.reshape(tensor.shape[0], -1)
    max_value = jnp.max(tensor, axis=-1, keepdims=True)
    min_value = jnp.min(tensor, axis=-1, keepdims=True)
    return ((tensor - min_value) /
            (max_value - min_value + 1e-5)).reshape(*shape)


class ConvTMCell(nn.Module):
    """MuZero-style transition model for SPR."""

    num_actions: int
    latent_dim: int
    renormalize: bool
    dtype: Dtype = jnp.float32
    initializer: Any = nn.initializers.xavier_uniform()

    @nn.compact
    def __call__(self, x, action, eval_mode=False, key=None):
        sizes = [self.latent_dim, self.latent_dim]
        kernel_sizes = [3, 3]
        stride_sizes = [1, 1]

        action_onehot = jax.nn.one_hot(action, self.num_actions)
        action_onehot = jax.lax.broadcast(action_onehot,
                                          (x.shape[-3], x.shape[-2]))
        x = jnp.concatenate([x, action_onehot], -1)
        for layer in range(1):
            x = nn.Conv(
                features=sizes[layer],
                kernel_size=(kernel_sizes[layer], kernel_sizes[layer]),
                strides=(stride_sizes[layer], stride_sizes[layer]),
                kernel_init=self.initializer,
                dtype=self.dtype,
            )(x)
            x = nn.relu(x)
        x = nn.Conv(
            features=sizes[-1],
            kernel_size=(kernel_sizes[-1], kernel_sizes[-1]),
            strides=(stride_sizes[-1], stride_sizes[-1]),
            kernel_init=self.initializer,
            dtype=self.dtype,
        )(x)
        x = nn.relu(x)

        if self.renormalize:
            x = renormalize(x)

        return x, x


@gin.configurable
class ImpalaCNN(nn.Module):
    """ResNet encoder based on Impala.

  Attributes:
    width_scale: Float, width scale relative to the default.
    dims: Dimensions for each stage.
    num_blocks: Number of resblocks per stage.
    dtype: Jax Dtype.
    fixup_init: Whether to do a fixup-style init (final layer of each resblock
      has weights set to 0).
    initializer: Jax initializer.
  """
    width_scale: int = 1
    dims: Tuple[int, Ellipsis] = (16, 32, 32)
    num_blocks: int = 2
    dtype: Dtype = jnp.float32
    fixup_init: bool = False
    initializer: Any = nn.initializers.xavier_uniform()

    @nn.compact
    def __call__(self, x, deterministic=None):
        for width in self.dims:
            x = ResidualStage(
                dims=int(width * self.width_scale),
                num_blocks=self.num_blocks,
                dtype=self.dtype,
                fixup_init=self.fixup_init,
                initializer=self.initializer,
            )(x, deterministic)
        x = nn.relu(x)
        return x


class ResidualStage(nn.Module):
    """A single residual stage for an Impala-style ResNet.

  Attributes:
    dims: Number of channels.
    num_blocks: Number of blocks in the stage.
    use_max_pooling: Whether to pool (downsample) before the blocks.
    dtype: Jax dtype.
    fixup_init: Whether to initialize the last weights in each block to 0.
    initializer: Jax initializer.
  """

    dims: int
    num_blocks: int
    use_max_pooling: bool = True
    dtype: Dtype = jnp.float32
    fixup_init: bool = False
    initializer: Any = nn.initializers.xavier_uniform()

    @nn.compact
    def __call__(self, x, deterministic=None):
        if self.fixup_init:
            final_initializer = nn.initializers.zeros
        else:
            final_initializer = self.initializer

        conv_out = nn.Conv(
            features=self.dims,
            kernel_size=(3, 3),
            strides=1,
            kernel_init=self.initializer,
            padding='SAME',
            dtype=self.dtype,
        )(x)
        if self.use_max_pooling:
            conv_out = nn.max_pool(conv_out,
                                   window_shape=(3, 3),
                                   padding='SAME',
                                   strides=(2, 2))

        for _ in range(self.num_blocks):
            block_input = conv_out
            conv_out = nn.relu(conv_out)
            conv_out = nn.Conv(
                features=self.dims,
                kernel_size=(3, 3),
                strides=1,
                kernel_init=self.initializer,
                padding='SAME',
                dtype=self.dtype,
            )(conv_out)
            conv_out = nn.relu(conv_out)
            conv_out = nn.Conv(
                features=self.dims,
                kernel_size=(3, 3),
                strides=1,
                kernel_init=final_initializer,
                padding='SAME',
                dtype=self.dtype,
            )(conv_out)
            conv_out += block_input
        return conv_out


class TransitionModel(nn.Module):
    """An SPR-style transition model.

  Attributes:
    num_actions: Size of action conditioning input.
    latent_dim: Number of channels.
    renormalize: Whether to renormalize outputs to [0, 1] as in MuZero.
    dtype: Jax dtype.
    initializer: Jax initializer.
  """
    num_actions: int
    latent_dim: int
    renormalize: bool
    dtype: Dtype = jnp.float32
    initializer: Any = nn.initializers.xavier_uniform()

    @nn.compact
    def __call__(self, x, action):
        scan = nn.scan(
            ConvTMCell,
            in_axes=0,
            out_axes=0,
            variable_broadcast=['params'],
            split_rngs={'params': False},
        )(
            latent_dim=self.latent_dim,
            num_actions=self.num_actions,
            renormalize=self.renormalize,
            dtype=self.dtype,
            initializer=self.initializer,
        )
        return scan(x, action)


@gin.configurable
class RainbowDQNNetwork(nn.Module):
    """Jax Rainbow network for Full Rainbow.

  Attributes:
      num_actions: int, number of actions the agent can take at any state.
      num_atoms: int, the number of buckets of the value function distribution.
      noisy: bool, Whether to use noisy networks.
      distributional: bool, whether to use distributional RL.
  """

    num_actions: int
    num_atoms: int
    noisy: bool
    distributional: bool
    renormalize: bool = False
    padding: Any = 'SAME'
    hidden_dim: int = 512
    width_scale: float = 1.0
    dtype: Dtype = jnp.float32

    def setup(self):
        initializer = nn.initializers.xavier_uniform()

        self.encoder = ImpalaCNN(
            width_scale=self.width_scale,
            dtype=self.dtype,
            initializer=initializer,
        )
        latent_dim = self.encoder.dims[-1] * self.width_scale

        # debug - start
        #print('*' * 20)
        #print(' latent_dim: {}'.format(latent_dim))
        #print(' self.num_actions: {}'.format(self.num_actions))
        #exit(0)
        # debug - end

        self.transition_model = TransitionModel(
            num_actions=self.num_actions,
            latent_dim=int(latent_dim),
            renormalize=self.renormalize,
            dtype=self.dtype,
            initializer=initializer,
        )

        self.projection = FeatureLayer(
            int(self.hidden_dim),
            dtype=jnp.float32,
            initializer=initializer,
        )
        self.predictor = nn.Dense(int(self.hidden_dim),
                                  dtype=jnp.float32,
                                  kernel_init=initializer)
        self.head = LinearHead(
            num_actions=self.num_actions,
            num_atoms=self.num_atoms,
            dtype=jnp.float32,
            initializer=initializer,
        )

        # ******** #
        self.policy_projection = FeatureLayer(
            int(self.hidden_dim),
            dtype=jnp.float32,
            initializer=initializer,
        )
        self.predict_policy = nn.Dense(int(self.hidden_dim),
                                       dtype=jnp.float32,
                                       kernel_init=initializer)
        self.policy = nn.Dense(self.num_actions,
                               dtype=jnp.float32,
                               kernel_init=initializer)
        self._log_alpha = self.param('_log_alpha', nn.initializers.zeros_init(),
                                     ())

    def entropy_scale(self):
        return jnp.exp(self._log_alpha)

    def encode(self, x, eval_mode=False):
        latent = self.encoder(x, deterministic=not eval_mode)
        if self.renormalize:
            latent = renormalize(latent)
        return latent

    def encode_project(self, x, eval_mode):
        latent = self.encode(x, eval_mode)
        representation = latent.reshape(-1)
        return jnp.concatenate([
            self.project(representation, eval_mode),
            self.policy_projection(representation, eval_mode)
        ],
                               axis=-1)
        #return self.project(representation, eval_mode)
        #return self.project(representation, eval_mode) + self.policy_projection(
        #    representation, eval_mode)

    def project(self, x, eval_mode):
        projected = self.projection(x, eval_mode=eval_mode)
        return projected

    def spr_predict(self, x, eval_mode):
        return jnp.concatenate([
            self.predictor(self.project(x, eval_mode)),
            self.predict_policy(self.policy_projection(x, eval_mode))
        ],
                               axis=-1)
        #projected = self.project(x, eval_mode)
        #projected = self.project(x, eval_mode) + self.policy_projection(
        #    x, eval_mode)
        #return self.predictor(projected)

    def spr_rollout(self, latent, actions):
        _, pred_latents = self.transition_model(latent, actions)

        representations = pred_latents.reshape(pred_latents.shape[0], -1)
        predictions = jax.vmap(self.spr_predict,
                               in_axes=(0, None))(representations, True)
        return predictions

    def init_fn(
        self,
        x,
        support,
        actions=None,
        do_rollout=False,
        eval_mode=False,
    ):
        y = self(x, support, actions, do_rollout, eval_mode)
        return (
            y,
            self.policy(
                nn.relu(
                    self.policy_projection(
                        #jax.lax.stop_gradient(y.representation),
                        y.representation,
                        eval_mode))))
        #return (y,
        #        self.policy(jax.lax.stop_gradient(nn.relu(self.project(y.representation, eval_mode)))))

    def get_policy(self, x):
        x = self.encode(x, False)
        x = x.reshape(-1)
        #x = jax.lax.stop_gradient(x)
        logits = self.policy(nn.relu(self.policy_projection(x, False)))
        #logits = self.policy(jax.lax.stop_gradient(nn.relu(self.encode_project(x, False))))
        return (logits,
                jax.random.categorical(self.make_rng('action_sample'), logits))

    def __call__(
        self,
        x,
        support,
        actions=None,
        do_rollout=False,
        eval_mode=False,
    ):
        spatial_latent = self.encode(x, eval_mode)
        representation = spatial_latent.reshape(-1)
        # Single hidden layer
        x = self.project(representation, eval_mode)
        x = nn.relu(x)

        logits = self.head(x, eval_mode)

        if do_rollout:
            spatial_latent = self.spr_rollout(spatial_latent, actions)

        probabilities = jnp.squeeze(nn.softmax(logits))
        q_values = jnp.squeeze(jnp.sum(support * probabilities, axis=-1))
        return SPROutputType(q_values, logits, probabilities, spatial_latent,
                             representation)
