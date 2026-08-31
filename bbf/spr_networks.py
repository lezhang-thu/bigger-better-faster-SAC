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
    [
        'q_values', 'logits', 'probabilities', 'latent', 'representation',
        'spatial_latent', 'rollout_reps'
    ],
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
    """MuZero-style transition model for SPR.

  hidden_layers=1 reproduces the original SPR cell (one hidden conv plus the
  output conv); larger values add extra hidden convs for deeper dynamics.
  """

    num_actions: int
    latent_dim: int
    renormalize: bool
    hidden_layers: int = 1
    dtype: Dtype = jnp.float32
    initializer: Any = nn.initializers.xavier_uniform()

    @nn.compact
    def __call__(self, x, action, eval_mode=False, key=None):
        action_onehot = jax.nn.one_hot(action, self.num_actions)
        action_onehot = jax.lax.broadcast(action_onehot,
                                          (x.shape[-3], x.shape[-2]))
        x = jnp.concatenate([x, action_onehot], -1)
        for _ in range(self.hidden_layers):
            x = nn.Conv(
                features=self.latent_dim,
                kernel_size=(3, 3),
                strides=(1, 1),
                kernel_init=self.initializer,
                dtype=self.dtype,
            )(x)
            x = nn.relu(x)
        x = nn.Conv(
            features=self.latent_dim,
            kernel_size=(3, 3),
            strides=(1, 1),
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


class PredictionHead(nn.Module):
    """Small MLP head for world-model reward/continue prediction.

  Attributes:
    hidden_dim: Width of the single hidden layer.
    zero_init: Whether to zero-initialize the output layer, so that
      predictions start neutral (as in DreamerV3's outscale=0).
    dtype: Jax dtype.
    initializer: Jax initializer.
  """
    hidden_dim: int
    zero_init: bool = False
    dtype: Dtype = jnp.float32
    initializer: Any = nn.initializers.xavier_uniform()

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim,
                     kernel_init=self.initializer,
                     dtype=self.dtype)(nn.relu(x))
        x = nn.relu(x)
        out_initializer = (nn.initializers.zeros
                           if self.zero_init else self.initializer)
        x = nn.Dense(1, kernel_init=out_initializer, dtype=self.dtype)(x)
        return jnp.squeeze(x, -1)


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
    hidden_layers: int = 1
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
            hidden_layers=self.hidden_layers,
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
    transition_hidden_layers: int = 1
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
            hidden_layers=self.transition_hidden_layers,
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

        # ******** world-model heads for imagination ******** #
        self.reward_head = PredictionHead(
            hidden_dim=512,
            zero_init=True,
            dtype=jnp.float32,
            initializer=initializer,
        )
        self.continue_head = PredictionHead(
            hidden_dim=512,
            zero_init=False,
            dtype=jnp.float32,
            initializer=initializer,
        )

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

    def encode_project_with_latent(self, x, eval_mode):
        latent = self.encode(x, eval_mode)
        return self.features_from_spatial(latent, eval_mode)

    def features_from_spatial(self, spatial, eval_mode):
        representation = spatial.reshape(-1)
        projection = jnp.concatenate([
            self.project(representation, eval_mode),
            self.policy_projection(representation, eval_mode)
        ],
                                     axis=-1)
        return projection, representation

    def probabilities_from_spatial(self, spatial, eval_mode):
        representation = spatial.reshape(-1)
        logits = self.head(
            nn.relu(self.project(representation, eval_mode)), eval_mode)
        return jnp.squeeze(nn.softmax(logits))

    def project(self, x, eval_mode):
        projected = self.projection(x, eval_mode=eval_mode)
        return projected

    def spr_predict(self, x, eval_mode):
        return jnp.concatenate([
            self.predictor(self.project(x, eval_mode)),
            self.predict_policy(self.policy_projection(x, eval_mode))
        ],
                               axis=-1)

    def reward_from_feature(self, x):
        return self.reward_head(x)

    def continue_from_feature(self, x):
        return self.continue_head(x)

    def policy_logits_from_feature(self, x, eval_mode):
        return self.policy(nn.relu(self.policy_projection(x, eval_mode)))

    def q_logits_from_feature(self, x):
        h = nn.relu(self.project(x, eval_mode=True))
        return self.head(h, eval_mode=True)

    def q_values_from_feature(self, x, support):
        logits = self.q_logits_from_feature(x)
        probabilities = nn.softmax(logits)
        return jnp.sum(support * probabilities, axis=-1)

    def spr_rollout(self, latent, actions):
        _, pred_latents = self.transition_model(latent, actions)

        representations = pred_latents.reshape(pred_latents.shape[0], -1)
        predictions = jax.vmap(self.spr_predict,
                               in_axes=(0, None))(representations, True)
        return predictions, representations

    def imagine_from_latent(self, latent, horizon):
        """Rolls out the transition model under the current policy.

    All features are stop-gradiented so imagination losses can only train
    the heads applied to them (policy and, optionally, Q), never the
    encoder or transition model.

    Args:
      latent: Spatial latent of the start state, from encode().
      horizon: Number of imagined transitions (returns horizon + 1 steps).

    Returns:
      Dict of arrays stacked along a leading time axis of size horizon + 1.
    """
        latent = jax.lax.stop_gradient(latent)
        key = self.make_rng('action_sample')
        keys = jax.random.split(key, horizon + 1)
        features, actions, log_probs, entropies = [], [], [], []
        probs, rewards, continues = [], [], []
        for i in range(horizon + 1):
            feature = jax.lax.stop_gradient(latent.reshape(-1))
            logits = self.policy_logits_from_feature(feature, False)
            log_prob = jax.nn.log_softmax(logits)
            prob = jax.nn.softmax(logits)
            action = jax.random.categorical(keys[i], logits)

            features.append(feature)
            actions.append(action)
            log_probs.append(log_prob[action])
            entropies.append(-jnp.sum(prob * log_prob))
            probs.append(prob)
            rewards.append(self.reward_from_feature(feature))
            continues.append(jax.nn.sigmoid(self.continue_from_feature(feature)))

            latent, _ = self.transition_model(latent, action[None])
            latent = jax.lax.stop_gradient(latent)
        return {
            'features': jnp.stack(features),
            'actions': jnp.stack(actions),
            'log_probs': jnp.stack(log_probs),
            'entropies': jnp.stack(entropies),
            'probs': jnp.stack(probs),
            'rewards': jnp.stack(rewards),
            'continues': jnp.stack(continues),
        }

    def init_fn(
        self,
        x,
        support,
        actions=None,
        do_rollout=False,
        eval_mode=False,
    ):
        y = self(x, support, actions, do_rollout, eval_mode)
        _ = self.reward_from_feature(y.representation)
        _ = self.continue_from_feature(y.representation)
        return (
            y,
            self.policy_logits_from_feature(
                #jax.lax.stop_gradient(y.representation),
                y.representation,
                eval_mode))
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

        latent = spatial_latent
        rollout_reps = None
        if do_rollout:
            latent, rollout_reps = self.spr_rollout(spatial_latent, actions)

        probabilities = jnp.squeeze(nn.softmax(logits))
        q_values = jnp.squeeze(jnp.sum(support * probabilities, axis=-1))
        return SPROutputType(q_values, logits, probabilities, latent,
                             representation, spatial_latent, rollout_reps)
