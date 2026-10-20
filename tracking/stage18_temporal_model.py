"""Trainable target memory and independent candidate adapters for Stage-18."""


def build_temporal_modules(torch, nn, embed_dim, lens_z, search_side):
    class TemporalModules(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = dict(embed_dim=embed_dim, lens_z=lens_z,
                               search_side=search_side, memory_slots=4, memory_dim=64)
            self.template = nn.Linear(embed_dim, 64)
            self.slots = nn.Parameter(torch.randn(4, 64) * 0.02)
            self.query = nn.Linear(embed_dim, 64)
            self.key = nn.Linear(64, 64)
            self.value = nn.Linear(64, 64)
            self.read_out = nn.Linear(64, embed_dim, bias=False)
            nn.init.zeros_(self.read_out.weight)
            self.adapters = nn.ModuleList([
                nn.Sequential(nn.Linear(embed_dim, 16), nn.GELU(),
                              nn.Linear(16, embed_dim)) for _ in range(6)
            ])
            for adapter in self.adapters:
                nn.init.zeros_(adapter[-1].weight)
                nn.init.zeros_(adapter[-1].bias)
            self.target = nn.Linear(embed_dim, 64)
            self.action_embedding = nn.Embedding(7, 16)
            self.update = nn.GRUCell(64 + 64 + 16 + 1, 64)
            self.write_gate = nn.Linear(64 + 16 + 1, 1)
            self.spatial = nn.Conv2d(embed_dim, 16, 1)
            self.observation = nn.Sequential(
                nn.Linear(8 * 8 * 16 + 64 + 4 * 64 + 14, 128),
                nn.GELU(), nn.LayerNorm(128))
            self.actor = nn.Linear(128, 7)
            self.critic = nn.Linear(128, 1)
            nn.init.zeros_(self.actor.weight)
            with torch.no_grad():
                self.actor.bias.copy_(torch.tensor([0.7] + [0.05] * 6).log())

        def initial_memory(self, tokens):
            anchor = self.template(tokens[:, :lens_z].mean(1))
            return anchor[:, None, :] + self.slots[None, :, :]

        def read_memory(self, tokens, memory):
            attention = torch.softmax(
                self.query(tokens) @ self.key(memory).transpose(1, 2) / 8.0, dim=-1)
            return tokens + self.read_out(attention @ self.value(memory))

        def adapt(self, tokens, action):
            return tokens if action == 0 else tokens + self.adapters[action - 1](tokens)

        def observe(self, tokens, memory, history):
            search = tokens[:, lens_z:].transpose(1, 2).reshape(
                len(tokens), embed_dim, search_side, search_side)
            spatial = torch.nn.functional.adaptive_avg_pool2d(
                self.spatial(search), (8, 8)).flatten(1)
            anchor = self.template(tokens[:, :lens_z].mean(1))
            features = self.observation(torch.cat(
                (spatial, anchor, memory.flatten(1), history), dim=1))
            return self.actor(features), self.critic(features).squeeze(-1)

        def pool_target(self, tokens, roi):
            # ROI is normalized xywh in the search crop, from the executed box.
            search = tokens[:, lens_z:].transpose(1, 2).reshape(
                len(tokens), embed_dim, search_side, search_side)
            offsets = (torch.arange(3, device=tokens.device, dtype=tokens.dtype) + 0.5) / 3
            yy, xx = torch.meshgrid(offsets, offsets, indexing="ij")
            grid = torch.stack((xx, yy), dim=-1)[None]
            grid = roi[:, None, None, :2] + grid * roi[:, None, None, 2:]
            sampled = torch.nn.functional.grid_sample(
                search, grid * 2 - 1, align_corners=False, padding_mode="border")
            return sampled.mean((2, 3))

        def write_memory(self, memory, tokens, roi, action, confidence):
            target = self.target(self.pool_target(tokens, roi))
            anchor = self.template(tokens[:, :lens_z].mean(1))
            action_features = self.action_embedding(action)
            context = torch.cat((target, action_features, confidence[:, None]), dim=1)
            inputs = torch.cat((context, anchor), dim=1)
            inputs = inputs[:, None].expand(-1, 4, -1).reshape(-1, inputs.shape[-1])
            proposed = self.update(inputs, memory.reshape(-1, 64)).reshape_as(memory)
            gate = torch.sigmoid(self.write_gate(context))[:, None, :]
            return memory + gate * (proposed - memory)

    return TemporalModules()
