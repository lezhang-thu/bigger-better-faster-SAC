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
"""Networks for the BBF/r2 world-model ablation."""

import collections
import enum
import functools
import math
import time
from typing import Any, Sequence, Tuple

from absl import logging
from flax import linen as nn
import gin
import jax
from jax import random
import jax.numpy as jnp
import numpy as onp
import optax

R2QOutputType = collections.namedtuple(
    'r2_q_network',
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

def _r2_initializer():
    """R2-Dreamer-style truncated normal initializer."""

    def init(key, shape, dtype=jnp.float32):
        if len(shape) < 2:
            fan_in = shape[0] if shape else 1
        else:
            fan_in = shape[0]
            if len(shape) == 2:
                fan_in = shape[0]
            elif len(shape) > 2:
                fan_in = shape[1] * math.prod(shape[2:])
        std = 1.1368 * math.sqrt(1.0 / max(1, fan_in))
        return nn.initializers.truncated_normal(stddev=std)(key, shape, dtype)

    return init


def _r2_symlog(x):
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))


def _r2_symexp(x):
    return jnp.sign(x) * jnp.expm1(jnp.abs(x))


def _r2_twohot_bins(bin_num):
    if bin_num % 2 == 1:
        half = jnp.linspace(-20.0, 0.0, (bin_num - 1) // 2 + 1,
                            dtype=jnp.float32)
        half = _r2_symexp(half)
        return jnp.concatenate([half, -jnp.flip(half[:-1])], axis=0)
    half = jnp.linspace(-20.0, 0.0, bin_num // 2, dtype=jnp.float32)
    half = _r2_symexp(half)
    return jnp.concatenate([half, -jnp.flip(half)], axis=0)


def _r2_kl_divergence(left_logits, right_logits):
    left_logprob = jax.nn.log_softmax(left_logits, axis=-1)
    right_logprob = jax.nn.log_softmax(right_logits, axis=-1)
    left_prob = jax.nn.softmax(left_logits, axis=-1)
    return jnp.sum(left_prob * (left_logprob - right_logprob), axis=-1)


def _r2_onehot_entropy(logits, unimix_ratio):
    probs = jax.nn.softmax(logits.astype(jnp.float32), axis=-1)
    uniform = unimix_ratio / probs.shape[-1]
    probs = probs * (1.0 - unimix_ratio) + uniform
    entropy = -jnp.sum(probs * jnp.log(probs + 1e-8), axis=-1)
    return jnp.sum(entropy, axis=-1)


def _r2_twohot_neg_log_prob(logits, target, bin_num):
    target = jnp.squeeze(target.astype(jnp.float32), axis=-1)
    bins = _r2_twohot_bins(bin_num)
    target_squashed = jax.lax.stop_gradient(_r2_symlog(target))
    below = jnp.sum((bins <= target_squashed[..., None]).astype(jnp.int32),
                    axis=-1) - 1
    above = bin_num - jnp.sum(
        (bins > target_squashed[..., None]).astype(jnp.int32), axis=-1)
    below = jnp.clip(below, 0, bin_num - 1)
    above = jnp.clip(above, 0, bin_num - 1)
    equal = below == above
    below_value = jnp.take(bins, below)
    above_value = jnp.take(bins, above)
    dist_to_below = jnp.where(equal, 1.0, jnp.abs(below_value - target_squashed))
    dist_to_above = jnp.where(equal, 1.0, jnp.abs(above_value - target_squashed))
    total = dist_to_below + dist_to_above
    weight_below = dist_to_above / total
    weight_above = dist_to_below / total
    mixed_target = (
        jax.nn.one_hot(below, bin_num) * weight_below[..., None] +
        jax.nn.one_hot(above, bin_num) * weight_above[..., None])
    log_pred = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
    return -jnp.sum(mixed_target * log_pred, axis=-1)


class R2BlockLinear(nn.Module):
    """Block-wise linear layer matching r2dreamer.networks.BlockLinear."""

    out_ch: int
    blocks: int
    initializer: Any = _r2_initializer()

    @nn.compact
    def __call__(self, x):
        in_ch = x.shape[-1]
        if in_ch % self.blocks != 0 or self.out_ch % self.blocks != 0:
            raise ValueError(
                f"BlockLinear requires divisible dims, got {in_ch}, "
                f"{self.out_ch}, blocks={self.blocks}")
        weight = self.param(
            "kernel",
            self.initializer,
            (self.out_ch // self.blocks, in_ch // self.blocks, self.blocks),
        )
        bias = self.param("bias", nn.initializers.zeros, (self.out_ch,))
        x_shape = x.shape[:-1]
        x = x.reshape(*x_shape, self.blocks, in_ch // self.blocks)
        x = jnp.einsum("...gi,oig->...go", x, weight)
        x = x.reshape(*x_shape, self.out_ch)
        return x + bias


class R2DreamerDeter(nn.Module):
    deter: int
    flat_stoch: int
    act_dim: int
    hidden: int
    blocks: int
    dyn_layers: int
    dtype: Dtype = jnp.float32
    initializer: Any = _r2_initializer()

    @nn.compact
    def __call__(self, stoch, deter, action):
        batch_size = action.shape[0]
        stoch = stoch.reshape(batch_size, -1)
        action = action / jax.lax.stop_gradient(jnp.maximum(jnp.abs(action),
                                                            1.0))

        def dense_norm_act(x, features, name):
            x = nn.Dense(features,
                         kernel_init=self.initializer,
                         dtype=self.dtype,
                         name=f"{name}_dense")(x)
            x = nn.RMSNorm(epsilon=1e-4,
                           dtype=jnp.float32,
                           name=f"{name}_norm")(x)
            return nn.silu(x)

        x0 = dense_norm_act(deter, self.hidden, "dyn_in0")
        x1 = dense_norm_act(stoch, self.hidden, "dyn_in1")
        x2 = dense_norm_act(action, self.hidden, "dyn_in2")

        x = jnp.concatenate([x0, x1, x2], axis=-1)
        x = jnp.expand_dims(x, axis=-2)
        x = jnp.broadcast_to(x, (batch_size, self.blocks, 3 * self.hidden))
        grouped_deter = deter.reshape(batch_size, self.blocks,
                                      self.deter // self.blocks)
        x = jnp.concatenate([grouped_deter, x], axis=-1)
        x = x.reshape(batch_size, -1)

        for i in range(self.dyn_layers):
            x = R2BlockLinear(self.deter,
                              self.blocks,
                              initializer=self.initializer,
                              name=f"dyn_hid_{i}")(x)
            x = nn.RMSNorm(epsilon=1e-4,
                           dtype=jnp.float32,
                           name=f"dyn_hid_norm_{i}")(x)
            x = nn.silu(x)

        x = R2BlockLinear(3 * self.deter,
                          self.blocks,
                          initializer=self.initializer,
                          name="dyn_gru")(x)
        gates = x.reshape(batch_size, self.blocks, 3 * self.deter //
                          self.blocks)
        reset, cand, update = jnp.split(gates, 3, axis=-1)
        reset = reset.reshape(batch_size, self.deter)
        cand = cand.reshape(batch_size, self.deter)
        update = update.reshape(batch_size, self.deter)
        reset = jax.nn.sigmoid(reset)
        cand = jnp.tanh(reset * cand)
        update = jax.nn.sigmoid(update - 1.0)
        return update * cand + (1.0 - update) * deter


class R2DreamerRSSM(nn.Module):
    embed_size: int
    act_dim: int
    stoch: int = 32
    deter: int = 6144
    hidden: int = 768
    discrete: int = 48
    img_layers: int = 2
    obs_layers: int = 1
    dyn_layers: int = 1
    blocks: int = 8
    unimix_ratio: float = 0.01
    dtype: Dtype = jnp.float32
    initializer: Any = _r2_initializer()

    @property
    def flat_stoch(self):
        return self.stoch * self.discrete

    @property
    def feat_size(self):
        return self.flat_stoch + self.deter

    def initial(self, batch_size):
        stoch = jnp.zeros((batch_size, self.stoch, self.discrete),
                          dtype=jnp.float32)
        deter = jnp.zeros((batch_size, self.deter), dtype=jnp.float32)
        return stoch, deter

    @nn.compact
    def _obs_logits(self, x):
        for i in range(self.obs_layers):
            x = nn.Dense(self.hidden,
                         kernel_init=self.initializer,
                         dtype=self.dtype,
                         name=f"obs_net_{i}")(x)
            x = nn.RMSNorm(epsilon=1e-4,
                           dtype=jnp.float32,
                           name=f"obs_net_norm_{i}")(x)
            x = nn.silu(x)
        x = nn.Dense(self.stoch * self.discrete,
                     kernel_init=self.initializer,
                     dtype=self.dtype,
                     name="obs_net_logit")(x)
        return x.reshape(*x.shape[:-1], self.stoch, self.discrete)

    @nn.compact
    def _prior_logits(self, deter):
        x = deter
        for i in range(self.img_layers):
            x = nn.Dense(self.hidden,
                         kernel_init=self.initializer,
                         dtype=self.dtype,
                         name=f"img_net_{i}")(x)
            x = nn.RMSNorm(epsilon=1e-4,
                           dtype=jnp.float32,
                           name=f"img_net_norm_{i}")(x)
            x = nn.silu(x)
        x = nn.Dense(self.stoch * self.discrete,
                     kernel_init=self.initializer,
                     dtype=self.dtype,
                     name="img_net_logit")(x)
        return x.reshape(*x.shape[:-1], self.stoch, self.discrete)

    def _sample_stoch(self, logits, rng):
        probs = jax.nn.softmax(logits.astype(jnp.float32), axis=-1)
        uniform = self.unimix_ratio / probs.shape[-1]
        probs = probs * (1.0 - self.unimix_ratio) + uniform
        mixed_logits = jnp.log(probs + 1e-8)
        gumbel = jax.random.gumbel(rng, mixed_logits.shape)
        y_soft = jax.nn.softmax(mixed_logits + gumbel, axis=-1)
        y_hard = jax.nn.one_hot(jnp.argmax(y_soft, axis=-1), y_soft.shape[-1])
        return y_hard + y_soft - jax.lax.stop_gradient(y_soft)

    @nn.compact
    def obs_step(self, stoch, deter, prev_action, embed, reset, rng):
        reset = reset.astype(bool).reshape((reset.shape[0],))
        stoch = jnp.where(reset[:, None, None], jnp.zeros_like(stoch), stoch)
        deter = jnp.where(reset[:, None], jnp.zeros_like(deter), deter)
        prev_action = jnp.where(reset[:, None], jnp.zeros_like(prev_action),
                                prev_action)
        deter = R2DreamerDeter(
            deter=self.deter,
            flat_stoch=self.flat_stoch,
            act_dim=self.act_dim,
            hidden=self.hidden,
            blocks=self.blocks,
            dyn_layers=self.dyn_layers,
            dtype=self.dtype,
            initializer=self.initializer,
            name="deter_net",
        )(stoch, deter, prev_action)
        logits = self._obs_logits(jnp.concatenate([deter, embed], axis=-1))
        stoch = self._sample_stoch(logits, rng)
        return stoch, deter, logits

    def observe(self, embed, action, initial, reset, rng):
        stoch, deter = initial
        keys = jax.random.split(rng, embed.shape[1])
        stochs, deters, logits = [], [], []
        reset = jnp.squeeze(reset, axis=-1)
        for i in range(embed.shape[1]):
            stoch, deter, logit = self.obs_step(stoch, deter, action[:, i],
                                                embed[:, i], reset[:, i],
                                                keys[i])
            stochs.append(stoch)
            deters.append(deter)
            logits.append(logit)
        return (jnp.stack(stochs, axis=1), jnp.stack(deters, axis=1),
                jnp.stack(logits, axis=1))

    def prior_logits(self, deter):
        return self._prior_logits(deter)

    def get_feat(self, stoch, deter):
        stoch = stoch.reshape(*stoch.shape[:-2], self.flat_stoch)
        return jnp.concatenate([stoch, deter], axis=-1)

    def kl_loss(self, post_logit, prior_logit, free):
        rep_loss = jnp.sum(
            _r2_kl_divergence(post_logit,
                              jax.lax.stop_gradient(prior_logit)),
            axis=-1)
        dyn_loss = jnp.sum(
            _r2_kl_divergence(jax.lax.stop_gradient(post_logit),
                              prior_logit),
            axis=-1)
        return jnp.maximum(dyn_loss, free), jnp.maximum(rep_loss, free)


class R2DreamerMLPHead(nn.Module):
    out_dim: int
    layers: int = 1
    units: int = 768
    outscale: float = 1.0
    dtype: Dtype = jnp.float32
    initializer: Any = _r2_initializer()

    @nn.compact
    def __call__(self, x):
        for i in range(self.layers):
            x = nn.Dense(self.units,
                         kernel_init=self.initializer,
                         dtype=self.dtype,
                         name=f"mlp_linear_{i}")(x)
            x = nn.RMSNorm(epsilon=1e-4,
                           dtype=jnp.float32,
                           name=f"mlp_norm_{i}")(x)
            x = nn.silu(x)
        if self.outscale == 0.0:
            kernel_init = nn.initializers.zeros
        else:
            base_init = self.initializer

            def kernel_init(key, shape, dtype=jnp.float32):
                return self.outscale * base_init(key, shape, dtype)

        return nn.Dense(self.out_dim,
                        kernel_init=kernel_init,
                        bias_init=nn.initializers.zeros,
                        dtype=self.dtype,
                        name="last")(x)


class R2DreamerProjector(nn.Module):
    out_dim: int
    dtype: Dtype = jnp.float32
    initializer: Any = _r2_initializer()

    @nn.compact
    def __call__(self, x):
        return nn.Dense(self.out_dim,
                        use_bias=False,
                        kernel_init=self.initializer,
                        dtype=self.dtype)(x)


class R2DreamerWorldModel(nn.Module):
    embed_size: int
    act_dim: int
    stoch: int = 32
    deter: int = 6144
    hidden: int = 768
    discrete: int = 48
    units: int = 768
    blocks: int = 8
    img_layers: int = 2
    obs_layers: int = 1
    dyn_layers: int = 1
    kl_free: float = 1.0
    unimix_ratio: float = 0.01
    barlow_lambd: float = 5e-4
    reward_bins: int = 255
    loss_scale_dyn: float = 1.0
    loss_scale_rep: float = 0.1
    loss_scale_barlow: float = 0.05
    loss_scale_rew: float = 1.0
    loss_scale_con: float = 1.0
    dtype: Dtype = jnp.float32
    initializer: Any = _r2_initializer()

    def setup(self):
        self.rssm = R2DreamerRSSM(
            embed_size=self.embed_size,
            act_dim=self.act_dim,
            stoch=self.stoch,
            deter=self.deter,
            hidden=self.hidden,
            discrete=self.discrete,
            img_layers=self.img_layers,
            obs_layers=self.obs_layers,
            dyn_layers=self.dyn_layers,
            blocks=self.blocks,
            unimix_ratio=self.unimix_ratio,
            dtype=self.dtype,
            initializer=self.initializer,
        )
        self.projector = R2DreamerProjector(self.embed_size,
                                            dtype=jnp.float32,
                                            initializer=self.initializer)
        self.reward = R2DreamerMLPHead(self.reward_bins,
                                       layers=1,
                                       units=self.units,
                                       outscale=0.0,
                                       dtype=jnp.float32,
                                       initializer=self.initializer)
        self.cont = R2DreamerMLPHead(1,
                                     layers=1,
                                     units=self.units,
                                     outscale=1.0,
                                     dtype=jnp.float32,
                                     initializer=self.initializer)

    @property
    def feat_size(self):
        return self.stoch * self.discrete + self.deter

    def initial(self, batch_size):
        return self.rssm.initial(batch_size)

    def observe_single(self, embed, prev_action, stoch, deter, is_first, rng):
        return self.rssm.obs_step(stoch, deter, prev_action, embed, is_first,
                                  rng)

    def loss(self, embed, action, reward, terminal, is_first, initial, rng):
        batch_size, batch_length = embed.shape[:2]
        post_stoch, post_deter, post_logit = self.rssm.observe(
            embed, action, initial, is_first, rng)
        prior_logit = self.rssm.prior_logits(post_deter)
        dyn_loss, rep_loss = self.rssm.kl_loss(post_logit, prior_logit,
                                               self.kl_free)
        dyn_loss = jnp.mean(dyn_loss)
        rep_loss = jnp.mean(rep_loss)

        feat = self.rssm.get_feat(post_stoch, post_deter)
        projected_feat = self.projector(feat.reshape(batch_size * batch_length,
                                                     -1))
        embed_target = jax.lax.stop_gradient(
            embed.reshape(batch_size * batch_length, -1))
        x1 = (projected_feat - jnp.mean(projected_feat, axis=0)) / (
            jnp.std(projected_feat, axis=0, ddof=1) + 1e-8)
        x2 = (embed_target - jnp.mean(embed_target, axis=0)) / (
            jnp.std(embed_target, axis=0, ddof=1) + 1e-8)
        corr = jnp.matmul(x1.T, x2) / (batch_size * batch_length)
        invariance = jnp.sum(jnp.square(jnp.diag(corr) - 1.0))
        off_diag = corr - jnp.diag(jnp.diag(corr))
        redundancy = jnp.sum(jnp.square(off_diag))
        barlow_loss = invariance + self.barlow_lambd * redundancy

        reward_logits = self.reward(feat)
        rew_loss = jnp.mean(
            _r2_twohot_neg_log_prob(reward_logits, reward, self.reward_bins))
        cont_target = 1.0 - terminal.astype(jnp.float32)
        con_loss = jnp.mean(
            optax.sigmoid_binary_cross_entropy(self.cont(feat), cont_target))

        total = (
            self.loss_scale_dyn * dyn_loss +
            self.loss_scale_rep * rep_loss +
            self.loss_scale_barlow * barlow_loss +
            self.loss_scale_rew * rew_loss +
            self.loss_scale_con * con_loss)
        metrics = {
            "R2WMLoss": total,
            "R2WMDynLoss": dyn_loss,
            "R2WMRepLoss": rep_loss,
            "R2WMBarlowLoss": barlow_loss,
            "R2WMRewardLoss": rew_loss,
            "R2WMContLoss": con_loss,
            "R2WMDynEntropy": jnp.mean(
                _r2_onehot_entropy(prior_logit, self.unimix_ratio)),
            "R2WMRepEntropy": jnp.mean(
                _r2_onehot_entropy(post_logit, self.unimix_ratio)),
        }
        return total, metrics, post_stoch, post_deter


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
    r2_world_model_stoch: int = 32
    r2_world_model_deter: int = 6144
    r2_world_model_hidden: int = 768
    r2_world_model_discrete: int = 48
    r2_world_model_units: int = 768
    r2_world_model_blocks: int = 8

    def setup(self):
        initializer = nn.initializers.xavier_uniform()

        self.encoder = ImpalaCNN(
            width_scale=self.width_scale,
            dtype=self.dtype,
            initializer=initializer,
        )
        self.representation_projection = FeatureLayer(
            int(self.hidden_dim),
            dtype=jnp.float32,
            initializer=initializer,
        )
        self.r2_feature_projection = FeatureLayer(
            int(self.hidden_dim),
            dtype=jnp.float32,
            initializer=initializer,
        )
        self.r2_head = LinearHead(
            num_actions=self.num_actions,
            num_atoms=self.num_atoms,
            dtype=jnp.float32,
            initializer=initializer,
        )
        self.r2_policy_projection = FeatureLayer(
            int(self.hidden_dim),
            dtype=jnp.float32,
            initializer=initializer,
        )
        self.r2_policy = nn.Dense(self.num_actions,
                                  dtype=jnp.float32,
                                  kernel_init=initializer)
        self.r2_world_model = R2DreamerWorldModel(
            embed_size=int(self.hidden_dim),
            act_dim=self.num_actions,
            stoch=self.r2_world_model_stoch,
            deter=self.r2_world_model_deter,
            hidden=self.r2_world_model_hidden,
            discrete=self.r2_world_model_discrete,
            units=self.r2_world_model_units,
            blocks=self.r2_world_model_blocks,
            dtype=jnp.float32,
            name="r2_world_model",
        )

    def encode(self, x, eval_mode=False):
        latent = self.encoder(x, deterministic=not eval_mode)
        if self.renormalize:
            latent = renormalize(latent)
        return latent

    def encode_project(self, x, eval_mode):
        latent = self.encode(x, eval_mode)
        return self.represent(latent.reshape(-1), eval_mode)

    def represent(self, x, eval_mode):
        return self.representation_projection(x, eval_mode=eval_mode)

    def init_fn(
        self,
        x,
        support,
        eval_mode=False,
    ):
        self.encode_project(x, eval_mode)
        dummy_states = jnp.zeros((1, 1) + x.shape, dtype=x.dtype)
        dummy_actions = jnp.zeros((1, 1, self.num_actions), dtype=jnp.float32)
        dummy_rewards = jnp.zeros((1, 1, 1), dtype=jnp.float32)
        dummy_terminals = jnp.zeros((1, 1, 1), dtype=jnp.float32)
        dummy_first = jnp.ones((1, 1, 1), dtype=jnp.float32)
        dummy_stoch, dummy_deter = self.r2_world_model.initial(1)
        self.r2_world_model_loss_from_states(
            dummy_states,
            dummy_actions,
            dummy_rewards,
            dummy_terminals,
            dummy_first,
            dummy_stoch,
            dummy_deter,
            jax.random.PRNGKey(0),
            eval_mode=True,
        )
        self.q_from_r2_features(dummy_stoch[0], dummy_deter[0], support,
                                eval_mode)
        self.r2_policy_logits(dummy_stoch[0], dummy_deter[0])
        return dummy_stoch, dummy_deter

    def r2_feature(self, stoch, deter):
        stoch = stoch.reshape(-1)
        deter = deter.reshape(-1)
        return jnp.concatenate([stoch, deter], axis=-1)

    def q_from_r2_features(self, stoch, deter, support, eval_mode=False):
        representation = self.r2_feature(stoch, deter)
        x = self.r2_feature_projection(representation, eval_mode)
        x = nn.relu(x)
        logits = self.r2_head(x, eval_mode)
        probabilities = jnp.squeeze(nn.softmax(logits))
        q_values = jnp.squeeze(jnp.sum(support * probabilities, axis=-1))
        return R2QOutputType(q_values, logits, probabilities, representation,
                             representation)

    def r2_policy_logits(self, stoch, deter):
        x = self.r2_feature(stoch, deter)
        x = nn.relu(self.r2_policy_projection(x, False))
        return self.r2_policy(x)

    def get_policy_from_r2(self, stoch, deter):
        logits = self.r2_policy_logits(stoch, deter)
        return (logits,
                jax.random.categorical(self.make_rng('action_sample'), logits))

    def r2_initial(self, batch_size):
        return self.r2_world_model.initial(batch_size)

    def r2_world_model_loss_from_states(
        self,
        states,
        actions,
        rewards,
        terminals,
        is_first,
        initial_stoch,
        initial_deter,
        rng,
        eval_mode=True,
    ):
        batch_size, batch_length = states.shape[:2]
        flat_states = states.reshape(batch_size * batch_length,
                                     *states.shape[2:])
        flat_embed = jax.vmap(
            lambda state: self.encode_project(state, eval_mode),
            in_axes=0,
            axis_name="r2_world_model_batch",
        )(flat_states)
        #flat_embed = jax.lax.stop_gradient(flat_embed)
        embed = flat_embed.reshape(batch_size, batch_length, -1)
        return self.r2_world_model.loss(embed, actions, rewards, terminals,
                                        is_first,
                                        (initial_stoch, initial_deter), rng)

    def r2_world_model_observe(
        self,
        state,
        prev_action,
        stoch,
        deter,
        is_first,
        rng,
        eval_mode=True,
    ):
        embed = jax.vmap(lambda x: self.encode_project(x, eval_mode),
                         in_axes=0)(state)
        return self.r2_world_model.observe_single(embed, prev_action, stoch,
                                                  deter, is_first, rng)
