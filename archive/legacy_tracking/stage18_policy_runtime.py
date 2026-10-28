"""On-policy actor observations over a frozen, live target-memory tracker."""

import numpy as np

from stage18_environment import TemporalEnvironment


POLICY_PREFIXES = ("spatial.", "observation.", "actor.", "critic.")


def configure_policy(modules):
    for name, parameter in modules.named_parameters():
        parameter.requires_grad_(name.startswith(POLICY_PREFIXES))
    return [p for p in modules.parameters() if p.requires_grad]


def load_tracking_representation(modules, state_dict):
    """Load Phase-A tracking weights, retaining the new seed's untrained policy."""
    policy = {name: value.detach().clone() for name, value in modules.state_dict().items()
              if name.startswith(POLICY_PREFIXES)}
    modules.load_state_dict(state_dict)
    modules.load_state_dict(policy, strict=False)


def compact_observation(torch, modules, tokens, memory, history):
    config = modules.config
    search = tokens[:, config["lens_z"]:].transpose(1, 2).reshape(
        len(tokens), config["embed_dim"], config["search_side"], config["search_side"])
    return dict(grid=torch.nn.functional.adaptive_avg_pool2d(search, (8, 8)),
                template=tokens[:, :config["lens_z"]].mean(1),
                memory=memory, history=history)


def policy_forward(torch, modules, observation):
    # Pool before the pointwise convolution for compact, lossless spatial
    # observations. Sampling and PPO recomputation both use this exact path.
    grid, memory, history = (observation[k] for k in ("grid", "memory", "history"))
    spatial = modules.spatial(grid).flatten(1)
    anchor = modules.template(observation["template"])
    features = modules.observation(torch.cat((spatial, anchor, memory.flatten(1), history), dim=1))
    logits, values = modules.actor(features), modules.critic(features).squeeze(-1)
    legal = torch.ones_like(logits, dtype=torch.bool)
    legal[:, 1:] = history[:, 5:6] >= 0.5
    return logits.masked_fill(~legal, -torch.inf), values


class _ActorView:
    def __init__(self, torch, modules):
        self.torch, self.modules = torch, modules
        self.stochastic = False
        self.last = None

    def __getattr__(self, name):
        return getattr(self.modules, name)

    def observe(self, tokens, memory, history):
        torch = self.torch
        obs = compact_observation(torch, self.modules, tokens, memory, history)
        logits, value = policy_forward(torch, self.modules, obs)
        distribution = torch.distributions.Categorical(logits=logits)
        action = distribution.sample() if self.stochastic else logits.argmax(1)
        self.last = dict(observation={k: v.detach() for k, v in obs.items()},
                         logits=logits, value=value, log_probability=distribution.log_prob(action))
        forced = torch.full_like(logits, -torch.inf).scatter(1, action[:, None], 0.)
        return forced, value


class PolicyEnvironment(TemporalEnvironment):
    def __init__(self, runtime, modules):
        self.actor_view = _ActorView(runtime.torch, modules)
        self.policy_modules = modules
        super().__init__(runtime, self.actor_view)

    def step(self, image, stochastic=False):
        self.actor_view.stochastic = bool(stochastic)
        with self.rt.torch.no_grad():
            result = super().step(image)
        result.update(self.actor_view.last)
        return result

    def peek_value(self, image):
        """Bootstrap the next observation without actions, heads, or RNG draws."""
        rt, torch = self.rt, self.rt.torch
        with torch.no_grad():
            search, _ = rt.prepare(image)
            tokens = rt.prefix(search)
            memory = self.memory
            if memory is None:
                memory = self.policy_modules.initial_memory(tokens)
            history = np.r_[self.motion, self.previous_confidence,
                            min(20, self.budget.credit + 3) / 20,
                            np.eye(8, dtype=np.float32)[self.previous_action + 1]]
            obs = compact_observation(torch, self.policy_modules, tokens, memory,
                                       tokens.new_tensor(history)[None])
            _, value = policy_forward(torch, self.policy_modules, obs)
        return float(value.item())
