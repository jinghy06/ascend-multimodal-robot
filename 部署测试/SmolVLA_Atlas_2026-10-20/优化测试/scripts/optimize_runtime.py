"""Inference-only SmolVLA changes. No cross-observation feature cache."""
import types
import numpy as np
import torch


def install_static_loop(policy, module):
    """Keep the Euler integration grid, avoid device scalar reads per step."""
    model = policy.model
    old_sample = model.sample_actions
    old_position = module.create_sinusoidal_pos_embedding
    active = {"index": None, "embeddings": None}
    time_cache = {}

    def position(time, dimension, min_period, max_period, device="cpu"):
        if active["index"] is None:
            return old_position(time, dimension, min_period, max_period, device=device)
        return active["embeddings"][active["index"]]

    module.create_sinusoidal_pos_embedding = position

    def sample(self, images, img_masks, lang_tokens, lang_masks, state, noise=None):
        steps = self.config.num_steps
        bsize, device = state.shape[0], state.device
        key = (steps, bsize, str(device))
        if key not in time_cache:
            # Match the original float32 repeated-addition Euler grid.
            t, dt_host = np.float32(1), np.float32(-1.0 / steps)
            grid = []
            while t >= -dt_host / 2:
                grid.append(float(t))
                t = np.float32(t + dt_host)
            embeddings = []
            for t in grid:
                host_time = torch.full((bsize,), t, dtype=torch.float32)
                emb = old_position(host_time, self.vlm_with_expert.expert_hidden_size,
                                   self.config.min_period, self.config.max_period,
                                   device=torch.device("cpu"))
                embeddings.append(emb.float().to(device))
            time_cache[key] = (grid, embeddings,
                               torch.tensor(-1.0 / steps, dtype=torch.float32, device=device))
        grid, embeddings, dt = time_cache[key]
        if noise is None:
            noise = self.sample_noise((bsize, self.config.chunk_size,
                                       self.config.max_action_dim), device)
        prefix, padding, ar_mask = self.embed_prefix(images, img_masks, lang_tokens,
                                                    lang_masks, state=state)
        attention = module.make_att_2d_masks(padding, ar_mask)
        position_ids = torch.cumsum(padding, dim=1) - 1
        _, kv = self.vlm_with_expert.forward(attention_mask=attention,
                    position_ids=position_ids, past_key_values=None,
                    inputs_embeds=[prefix, None], use_cache=self.config.use_cache,
                    fill_kv_cache=True)
        x = noise
        active["embeddings"] = embeddings
        try:
            for i, t in enumerate(grid):
                active["index"] = i
                # This argument supplies shape/device only: position() uses cached
                # embeddings of the actual Euler time above, never old observations.
                timestep = torch.full((bsize,), t, dtype=torch.float32, device=device)
                v = self.denoise_step(padding, kv, x, timestep)
                x += dt * v
        finally:
            active["index"] = None
        return x

    model.sample_actions = types.MethodType(sample, model)

    def restore():
        model.sample_actions = old_sample
        module.create_sinusoidal_pos_embedding = old_position

    return restore


def install_batched_vision(policy):
    """Encode all current camera frames in one vision batch; discard every call."""
    vlm = policy.model.vlm_with_expert
    old_prefix = policy.model.embed_prefix
    old_image = vlm.embed_image
    queue = []

    def image(img):
        return queue.pop(0) if queue else old_image(img)

    def prefix(images, masks, lang_tokens, lang_masks, state=None):
        sizes = [x.shape[0] for x in images]
        combined = old_image(torch.cat(images, dim=0))
        queue[:] = list(combined.split(sizes, dim=0))
        try:
            return old_prefix(images, masks, lang_tokens, lang_masks, state=state)
        finally:
            queue.clear()

    policy.model.embed_prefix = prefix
    vlm.embed_image = image

    def restore():
        policy.model.embed_prefix = old_prefix
        vlm.embed_image = old_image

    return restore
