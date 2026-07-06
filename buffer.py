import torch
from tensordict import TensorDict
from torchrl.data.replay_buffers import LazyTensorStorage, ReplayBuffer
from torchrl.data.replay_buffers.samplers import SliceSampler


class Buffer:
    def __init__(self, config):
        self.device = torch.device(config.device)
        self.storage_device = torch.device(config.storage_device)
        self.batch_size = int(config.batch_size)
        self.batch_length = int(config.batch_length)
        self.num_actions = int(getattr(config, "num_actions", 0))
        self.stoch = int(getattr(config, "stoch", 32))
        self.discrete = int(getattr(config, "discrete", 32))
        self.deter = int(getattr(config, "deter", 1024))
        self.num_eps = 0
        self._buffer = ReplayBuffer(
            storage=LazyTensorStorage(max_size=config.max_size, device=self.storage_device, ndim=2),
            sampler=SliceSampler(
                num_slices=self.batch_size, end_key=None, traj_key="episode", truncated_key=None, strict_length=True
            ),
            prefetch=0,
            batch_size=self.batch_size * (self.batch_length + 1),  # +1 for context
        )

    def add_transition(self, data):
        # This is batched data and lifted for storage.
        # (B, ...) -> (B, 1, ...)
        return self._buffer.extend(data.unsqueeze(1))

    def add_atari_transition(self, state, action, reward, is_terminal,
                             is_first, stoch, deter):
        """Add BBF Atari data while keeping Torch/TensorDict details local.

        Args:
            state: uint8 stacked frame state, (B, H, W, stack).
            action: one-hot action selected at this state, (B, A).
            reward: raw unclipped reward arriving with this state, (B, 1).
            is_terminal: terminal flag, (B, 1).
            is_first: RSSM reset flag, (B, 1).
            stoch: posterior stochastic state for this observation, (B, S, K).
            deter: posterior deterministic state for this observation, (B, D).
        """
        def to_tensor(value, dtype=None):
            if torch.is_tensor(value):
                return value.to(device=self.storage_device, dtype=dtype)
            if hasattr(value, "copy"):
                value = value.copy()
            return torch.as_tensor(value,
                                   dtype=dtype,
                                   device=self.storage_device)

        state = to_tensor(state)
        action = to_tensor(action, dtype=torch.float32)
        reward = to_tensor(reward, dtype=torch.float32)
        is_terminal = to_tensor(is_terminal, dtype=torch.bool)
        is_first = to_tensor(is_first, dtype=torch.bool)
        stoch = to_tensor(stoch, dtype=torch.float32)
        deter = to_tensor(deter, dtype=torch.float32)
        batch_size = state.shape[0]
        episode = torch.zeros(batch_size,
                              dtype=torch.int32,
                              device=self.storage_device)
        data = TensorDict(
            {
                "state": state,
                "action": action,
                "reward": reward.reshape(batch_size, 1),
                "is_first": is_first.reshape(batch_size, 1),
                "is_terminal": is_terminal.reshape(batch_size, 1),
                "stoch": stoch.reshape(batch_size, self.stoch,
                                       self.discrete),
                "deter": deter.reshape(batch_size, self.deter),
                "episode": episode,
            },
            batch_size=(batch_size,),
            device=self.storage_device,
        )
        return self.add_transition(data)

    def _index_tensor(self, index):
        if torch.is_tensor(index):
            index = index.to(device=self.storage_device, dtype=torch.long)
        else:
            if hasattr(index, "copy"):
                index = index.copy()
            index = torch.as_tensor(index,
                                    dtype=torch.long,
                                    device=self.storage_device)
        if index.shape[-1] != 2:
            raise ValueError(
                f"Expected r2 replay indices with last dim 2, got {index.shape}"
            )
        return index

    def get_latents(self, index):
        """Fetch cached RSSM posterior states for TorchRL indices.

        The stored index convention is [time, env]. TorchRL's TensorDict
        indexing expects [env, time] for this two-dimensional storage.
        """
        index = self._index_tensor(index)
        leading_shape = tuple(index.shape[:-1])
        flat_index = index.reshape(-1, 2)
        data = self._buffer[flat_index[:, 1], flat_index[:, 0]]
        stoch = data["stoch"].reshape(*leading_shape, self.stoch,
                                      self.discrete)
        deter = data["deter"].reshape(*leading_shape, self.deter)
        return stoch, deter

    def get_refresh_batch(self, current_index, next_index, batch_length=None):
        """Build a fixed-history no-grad RSSM refresh batch for BBF samples.

        current_index and next_index are [time, env] rows for s_t and the
        n-step bootstrap state. The refresh window ends at next_index and has
        batch_length observed states, so s_t is refreshed with RSSM context
        when batch_length is larger than the BBF update horizon. The row before
        the window provides the initial latent state and actions are shifted
        back by one row, matching sample().
        """
        current_index = self._index_tensor(current_index)
        next_index = self._index_tensor(next_index)
        if current_index.ndim != 2 or next_index.ndim != 2:
            raise ValueError("Expected current_index and next_index to be (B, 2)")
        if not torch.equal(current_index[:, 1], next_index[:, 1]):
            raise ValueError("Cannot refresh segments spanning env indices")
        if self._buffer.storage.shape is None:
            raise RuntimeError("Cannot refresh an empty r2 replay buffer")

        batch_length = self.batch_length if batch_length is None else int(batch_length)
        if batch_length <= 0:
            raise ValueError(f"Expected positive batch_length, got {batch_length}")

        storage_len = int(self._buffer.storage.shape[0])
        diff = next_index[:, 0] - current_index[:, 0]
        if torch.any(diff < 0):
            diff = torch.remainder(diff, storage_len)
        if torch.any(diff >= batch_length):
            raise ValueError(
                "BBF current_index is outside the r2 refresh history window; "
                f"max offset {int(diff.max().item())}, batch_length {batch_length}"
            )

        offsets = torch.arange(-batch_length,
                               1,
                               dtype=torch.long,
                               device=self.storage_device)
        # Atari100k does not wrap the configured r2 replay buffer. Before the
        # buffer has batch_length rows, clamp the left edge instead of wrapping
        # into uninitialized future storage.
        times = next_index[:, 0:1] + offsets[None]
        times = torch.clamp(times, min=0, max=storage_len - 1)
        envs = current_index[:, 1:2].expand_as(times)
        context = self._buffer[envs, times]

        initial = (context["stoch"][:, 0], context["deter"][:, 0])
        data = TensorDict(
            {
                "state": context["state"][:, 1:],
                "action": context["action"][:, :-1],
                "is_first": context["is_first"][:, 1:],
            },
            batch_size=context.batch_size[:-1] + (batch_length,),
            device=self.storage_device,
        )
        index = [times[:, 1:], envs[:, 1:]]
        return data, index, initial

    def sample(self):
        sample_td, info = self._buffer.sample(return_info=True)
        # The sampler returns a flattened batch of length B*(T+1).
        # (B*(T+1), ...) -> (B, T+1, ...)
        sample_td = sample_td.view(-1, self.batch_length + 1)
        src_dev = sample_td.device
        if src_dev.type == "cpu" and self.device.type == "cuda":
            sample_td = sample_td.pin_memory().to(self.device, non_blocking=True)
        elif src_dev != self.device:
            sample_td = sample_td.to(self.device, non_blocking=True)
        # The initial ones are used only to extract the latent vector
        initial = (sample_td["stoch"][:, 0], sample_td["deter"][:, 0])
        data = sample_td[:, 1:]
        data.set_("action", sample_td["action"][:, :-1])  # action is 1 step back
        index = [ind.view(-1, self.batch_length + 1)[:, 1:] for ind in info["index"]]
        return data, index, initial

    def update(self, index, stoch, deter):
        # Flatten the data
        index = [ind.reshape(-1) for ind in index]
        if not torch.is_tensor(stoch):
            if hasattr(stoch, "copy"):
                stoch = stoch.copy()
            stoch = torch.as_tensor(stoch,
                                    dtype=torch.float32,
                                    device=self.storage_device)
        if not torch.is_tensor(deter):
            if hasattr(deter, "copy"):
                deter = deter.copy()
            deter = torch.as_tensor(deter,
                                    dtype=torch.float32,
                                    device=self.storage_device)
        # (B, T, S, K) -> (B*T, S, K)
        stoch = stoch.reshape(-1, *stoch.shape[2:])
        # (B, T, D) -> (B*T, D)
        deter = deter.reshape(-1, *deter.shape[2:])
        # In storage, the length is the first dimension, and the batch (number of environments) is the second dimension.
        n = index[0].shape[0]
        self._buffer[index[1], index[0]] = TensorDict({"stoch": stoch, "deter": deter}, batch_size=(n,))

    def count(self):
        if self._buffer.storage.shape is None:
            return 0
        return self._buffer.storage.shape.numel()
