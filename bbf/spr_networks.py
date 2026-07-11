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

from bbf import rssm as rssm_lib

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
        # Keep arbitrary leading dimensions.  The original implementation only
        # accepted a single feature vector and therefore needed an outer vmap.
        # RSSM training naturally produces [batch, time, feature] tensors, so a
        # batch-safe head avoids shape-dependent call sites.
        adv = adv.reshape((*adv.shape[:-1], self.num_actions, self.num_atoms))
        value = value.reshape((*value.shape[:-1], 1, self.num_atoms))
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
    rssm_stoch_size: int = 16
    rssm_discrete_size: int = 16
    rssm_deter_size: int = 256
    rssm_hidden_size: int = 256
    rssm_embed_dim: int = 512
    rssm_unimix: float = 0.01
    rssm_free_nats: float = 1.0

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

        # Preserve the true pre-abf7870 two-branch SPR representation.
        self.projection = FeatureLayer(
            int(self.hidden_dim),
            dtype=jnp.float32,
            initializer=initializer,
        )
        self.predictor = nn.Dense(
            int(self.hidden_dim),
            dtype=jnp.float32,
            kernel_init=initializer,
        )
        self.policy_projection = FeatureLayer(
            int(self.hidden_dim),
            dtype=jnp.float32,
            initializer=initializer,
        )
        self.predict_policy = nn.Dense(
            int(self.hidden_dim),
            dtype=jnp.float32,
            kernel_init=initializer,
        )

        # RSSM control heads have separate input adapters. Sharing the SPR
        # projections here would require the RSSM feature and flattened image
        # encoder output to have the same dimension.
        self.q_projection = FeatureLayer(
            int(self.hidden_dim),
            dtype=jnp.float32,
            initializer=initializer,
        )
        self.actor_projection = FeatureLayer(
            int(self.hidden_dim),
            dtype=jnp.float32,
            initializer=initializer,
        )
        self.head = LinearHead(
            num_actions=self.num_actions,
            num_atoms=self.num_atoms,
            dtype=jnp.float32,
            initializer=initializer,
        )
        self.policy = nn.Dense(
            self.num_actions,
            dtype=jnp.float32,
            kernel_init=initializer,
        )
        self._log_alpha = self.param('_log_alpha', nn.initializers.zeros_init(),
                                     ())
        self.reward_head = nn.Dense(1,
                                    dtype=jnp.float32,
                                    kernel_init=initializer)
        self.continue_head = nn.Dense(1,
                                      dtype=jnp.float32,
                                      kernel_init=initializer)
        self.value_head = nn.Dense(1,
                                   dtype=jnp.float32,
                                   kernel_init=initializer)

        # The RSSM is the common state interface for the real C51 critic, the
        # shared actor, and imagined actor--critic training.  The old spatial
        # transition model remains above solely for the existing SPR loss.
        self.rssm_embed_projection = FeatureLayer(
            int(self.rssm_embed_dim),
            dtype=jnp.float32,
            initializer=initializer,
        )
        self.rssm = rssm_lib.RSSM(
            stoch_size=int(self.rssm_stoch_size),
            discrete_size=int(self.rssm_discrete_size),
            deter_size=int(self.rssm_deter_size),
            hidden_size=int(self.rssm_hidden_size),
            initial_mode='zeros',
            unimix=float(self.rssm_unimix),
            free_nats=float(self.rssm_free_nats),
        )
        # R2-Dreamer's Barlow objective compares this projection of an RSSM
        # posterior feature to a stopped encoder embedding.  It intentionally
        # has no extra predictor/EMA branch.
        self.rssm_projector = FeatureLayer(
            int(self.rssm_embed_dim),
            dtype=jnp.float32,
            initializer=initializer,
        )

    def entropy_scale(self):
        return jnp.exp(self._log_alpha)

    def encode(self, x, eval_mode=False):
        latent = self.encoder(x, deterministic=not eval_mode)
        if self.renormalize:
            # Normalize each encoded observation independently.  Flattening a
            # [batch, time, H, W, C] tensor here would otherwise leak scale
            # information across unrelated replay sequences/time steps.
            reduce_axes = tuple(range(latent.ndim - 3, latent.ndim))
            maximum = jnp.max(latent, axis=reduce_axes, keepdims=True)
            minimum = jnp.min(latent, axis=reduce_axes, keepdims=True)
            latent = (latent - minimum) / (maximum - minimum + 1e-5)
        return latent

    def _rssm_embed_from_spatial(self, spatial_latent, eval_mode=False):
        """Projects encoder output while preserving all leading dimensions."""
        flat = spatial_latent.reshape((*spatial_latent.shape[:-3], -1))
        embed = self.rssm_embed_projection(flat, eval_mode)
        return nn.silu(embed)

    def encode_rssm(self, x, eval_mode=False):
        """Encodes observations into the embedding consumed by the RSSM.

        `x` may be one observation or have arbitrary batch/time leading axes.
        The returned tensor has the same leading axes and a final
        `rssm_embed_dim` axis.
        """
        return self._rssm_embed_from_spatial(self.encode(x, eval_mode),
                                             eval_mode)

    def rssm_initial(self, batch_size):
        """Returns the zero/learned initial RSSM state without sampling."""
        return self.rssm.initial(batch_size, deterministic=True)

    def rssm_observe(self,
                     embeds,
                     prev_action_onehots,
                     is_first,
                     eval_mode=False):
        """Runs posterior inference over batch-major replay sequences.

        Args:
          embeds: `[batch, time, rssm_embed_dim]` encoder embeddings.
          prev_action_onehots: `[batch, time, num_actions]`; item `t` is the
            action that led into observation `t` (zero at an episode start).
          is_first: `[batch, time]` reset indicators.
          eval_mode: Use deterministic categorical states when true.
        """
        return self.rssm.observe(
            embeds,
            prev_action_onehots,
            is_first,
            deterministic=eval_mode,
        )

    def rssm_observe_step(self,
                          x,
                          prev_state,
                          prev_action,
                          is_first,
                          eval_mode=False):
        """Updates a recurrent RSSM state from one real observation."""
        embed = self.encode_rssm(x, eval_mode)
        post, prior = self.rssm.obs_step(
            prev_state,
            prev_action,
            embed,
            is_first,
            deterministic=eval_mode,
        )
        return post, prior

    def rssm_feature(self, state):
        """Returns `[deter, flatten(stoch)]`, the common control feature."""
        return self.rssm.get_feat(state)

    def rssm_barlow_prediction(self, feature, eval_mode=False):
        """Projects a posterior feature to the encoder-embedding space."""
        return self.rssm_projector(feature, eval_mode)

    def rssm_kl_loss(self, post, prior, free_nats=None, balance=None):
        """Exposes the RSSM's balanced categorical KL objective."""
        return self.rssm.kl_loss(post, prior, free_nats, balance)

    def encode_project(self, x, eval_mode):
        latent = self.encode(x, eval_mode)
        representation = latent.reshape(-1)
        return jnp.concatenate(
            [
                self.project(representation, eval_mode),
                self.policy_projection(representation, eval_mode),
            ],
            axis=-1,
        )

    def project(self, x, eval_mode):
        return self.projection(x, eval_mode=eval_mode)

    def spr_predict(self, x, eval_mode):
        return jnp.concatenate(
            [
                self.predictor(self.project(x, eval_mode)),
                self.predict_policy(
                    self.policy_projection(x, eval_mode)),
            ],
            axis=-1,
        )

    def policy_logits_from_feature(self, x, eval_mode):
        hidden = self.actor_projection(x, eval_mode)
        return self.policy(nn.relu(hidden))

    def actor_from_rssm_feature(self, feature, eval_mode=False):
        """Categorical logits for the one actor shared by real and imagined states."""
        return self.policy_logits_from_feature(feature, eval_mode)

    def q_from_rssm_feature(self, feature, support, eval_mode=False):
        """Evaluates the real-data C51 critic on a common RSSM feature."""
        hidden = nn.relu(self.q_projection(feature, eval_mode))
        logits = self.head(hidden, eval_mode)
        probabilities = nn.softmax(logits, axis=-1)
        q_values = jnp.sum(support * probabilities, axis=-1)
        return SPROutputType(q_values, logits, probabilities, feature, feature)

    def rssm_from_observation(self, x, eval_mode=False):
        """Infers a self-contained posterior when recurrent context is absent.

        This deterministic zero-context path keeps legacy C51/evaluation call
        sites usable.  Sequence training and online recurrent acting should use
        `rssm_observe` and `rssm_observe_step`, respectively.
        """
        embed = self.encode_rssm(x, eval_mode)
        leading_shape = embed.shape[:-1]
        prev_state = self.rssm.initial(leading_shape, deterministic=True)
        prev_action = jnp.zeros(
            (*leading_shape, self.num_actions), dtype=embed.dtype)
        is_first = jnp.ones(leading_shape, dtype=jnp.bool_)
        return self.rssm.obs_step(
            prev_state, prev_action, embed, is_first, deterministic=True)

    def reward_from_feature(self, x):
        return jnp.squeeze(self.reward_head(nn.relu(x)), axis=-1)

    def continue_from_feature(self, x):
        return jnp.squeeze(self.continue_head(nn.relu(x)), axis=-1)

    def value_from_feature(self, x):
        return jnp.squeeze(self.value_head(nn.relu(x)), axis=-1)

    def spr_rollout(self, latent, actions):
        _, pred_latents = self.transition_model(latent, actions)

        representations = pred_latents.reshape(pred_latents.shape[0], -1)
        predictions = jax.vmap(self.spr_predict,
                               in_axes=(0, None))(representations, True)
        return predictions

    def spr_from_observation(self, x, actions, eval_mode=False):
        """Runs only the true-baseline SPR path, without C51/RSSM heads."""
        spatial_latent = self.encode(x, eval_mode)
        return self.spr_rollout(spatial_latent, actions)

    def init_fn(
        self,
        x,
        support,
        actions=None,
        do_rollout=False,
        eval_mode=False,
    ):
        y = self(x, support, actions, do_rollout, eval_mode)
        post, _ = self.rssm_from_observation(x, eval_mode)
        feature = self.rssm_feature(post)
        # Touch every RSSM-side parameter tree during Flax initialization and
        # resets.  In particular, neither Q nor actor modules are ever invoked
        # on the old SPR representation, which could give them an incompatible
        # input kernel shape.
        _ = self.rssm_barlow_prediction(feature, eval_mode)
        _ = self.reward_from_feature(feature)
        # Initialize both original SPR branches even when rollouts are off.
        _ = self.spr_predict(y.representation, eval_mode)
        _ = self.continue_from_feature(feature)
        _ = self.value_from_feature(feature)
        logits = self.actor_from_rssm_feature(feature, eval_mode)
        return y, logits

    def get_policy(self, x):
        """Legacy stateless policy; recurrent acting uses `rssm_observe_step`."""
        post, _ = self.rssm_from_observation(x, eval_mode=True)
        feature = self.rssm_feature(post)
        logits = self.actor_from_rssm_feature(feature, eval_mode=True)
        return (logits,
                jax.random.categorical(self.make_rng('action_sample'), logits))

    def imagine_from_observation(self, x, horizon, eval_mode=False):
        """Legacy zero-context entry point for RSSM imagination."""
        post, _ = self.rssm_from_observation(x, eval_mode)
        return self.imagine_from_rssm(post, horizon, eval_mode)

    def imagine_from_rssm(self, start_state, horizon, eval_mode=False):
        """Rolls the shared actor forward through RSSM priors.

        Rewards and continuation probabilities are predicted from the *next*
        (post-action) feature, matching Dreamer's temporal convention.  Values
        and features include the start plus all `horizon` successor states.
        """
        key = self.make_rng('action_sample')
        keys = jax.random.split(key, horizon)
        log_probs = []
        entropies = []
        rewards = []
        continues = []
        values = []
        actions = []
        # Imagination supplies actor/value learning targets; it must not update
        # the world model.  Stop both the recurrent carry and every feature at
        # the model/head boundary while retaining gradients into the heads.
        start_state = jax.tree_util.tree_map(jax.lax.stop_gradient,
                                             start_state)
        states = [start_state]
        feature = jax.lax.stop_gradient(self.rssm_feature(start_state))
        features = [feature]

        for i in range(horizon):
            logits = self.actor_from_rssm_feature(feature, eval_mode)
            log_prob = jax.nn.log_softmax(logits)
            prob = jax.nn.softmax(logits)
            action = (jnp.argmax(logits, axis=-1) if eval_mode else
                      jax.random.categorical(keys[i], logits))
            selected_log_prob = jnp.take_along_axis(
                log_prob, action[..., None], axis=-1)[..., 0]

            actions.append(action)
            log_probs.append(selected_log_prob)
            entropies.append(-jnp.sum(prob * log_prob, axis=-1))
            values.append(self.value_from_feature(feature))

            action_onehot = jax.nn.one_hot(
                action, self.num_actions, dtype=feature.dtype)
            start_state = self.rssm.img_step(
                start_state, action_onehot, deterministic=eval_mode)
            start_state = jax.tree_util.tree_map(jax.lax.stop_gradient,
                                                 start_state)
            feature = jax.lax.stop_gradient(self.rssm_feature(start_state))
            states.append(start_state)
            features.append(feature)
            rewards.append(self.reward_from_feature(feature))
            continues.append(
                jax.nn.sigmoid(self.continue_from_feature(feature)))

        values.append(self.value_from_feature(feature))
        stacked_states = jax.tree_util.tree_map(
            lambda *xs: jnp.stack(xs, axis=0), *states)
        stacked_features = jnp.stack(features, axis=0)

        return {
            'actions': jnp.stack(actions, axis=0),
            'log_probs': jnp.stack(log_probs, axis=0),
            'entropies': jnp.stack(entropies, axis=0),
            'rewards': jnp.stack(rewards, axis=0),
            'continues': jnp.stack(continues, axis=0),
            'values': jnp.stack(values, axis=0),
            'features': stacked_features,
            'current_features': stacked_features[:-1],
            'next_features': stacked_features[1:],
            'states': stacked_states,
        }

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
        embed = self._rssm_embed_from_spatial(spatial_latent, eval_mode)
        leading_shape = embed.shape[:-1]
        prev_state = self.rssm.initial(leading_shape, deterministic=True)
        prev_action = jnp.zeros(
            (*leading_shape, self.num_actions), dtype=embed.dtype)
        is_first = jnp.ones(leading_shape, dtype=jnp.bool_)
        post, _ = self.rssm.obs_step(
            prev_state, prev_action, embed, is_first, deterministic=True)
        feature = self.rssm_feature(post)
        q_output = self.q_from_rssm_feature(feature, support, eval_mode)

        if do_rollout:
            spatial_latent = self.spr_rollout(spatial_latent, actions)

        return SPROutputType(q_output.q_values, q_output.logits,
                             q_output.probabilities, spatial_latent,
                             representation)
