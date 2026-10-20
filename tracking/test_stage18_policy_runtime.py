"""Actor-only optimization, on-policy likelihoods, and causal bootstrap tests."""

import importlib.util
import unittest
import numpy as np

from stage18_policy_runtime import (PolicyEnvironment, configure_policy,
                                    compact_observation, policy_forward,
                                    load_tracking_representation)


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch unavailable")
class PolicyRuntimeTests(unittest.TestCase):
    def system(self):
        from test_stage18_temporal import ModelEnvironmentTests
        return ModelEnvironmentTests().make_system()

    def test_actor_parameter_scope_and_observation_match(self):
        torch, modules, rt = self.system()
        configure_policy(modules)
        for name, p in modules.named_parameters():
            expected = name.startswith(("spatial.", "observation.", "actor.", "critic."))
            self.assertEqual(p.requires_grad, expected)
        tokens = torch.randn(1, 8, 8)
        memory = modules.initial_memory(tokens)
        history = torch.zeros(1, 14)
        history[:, 5] = 1
        with torch.no_grad():
            modules.actor.weight.normal_(0, .1)
            reference, ref_value = modules.observe(tokens, memory, history)
            obs = compact_observation(torch, modules, tokens, memory, history)
            logits, value = policy_forward(torch, modules, obs)
        torch.testing.assert_close(reference, logits, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(ref_value, value, rtol=1e-5, atol=1e-6)

    def test_representation_load_retains_seed_specific_policy(self):
        torch, modules, rt = self.system()
        policy = {name: p.detach().clone() for name, p in modules.named_parameters()
                  if name.startswith(("spatial.", "observation.", "actor.", "critic."))}
        changed = {name: value.detach().clone() + 1 for name, value in modules.state_dict().items()}
        load_tracking_representation(modules, changed)
        for name, p in modules.named_parameters():
            torch.testing.assert_close(p, policy.get(name, changed[name]), rtol=0, atol=0)

    def test_exact_likelihood_replay_and_budget(self):
        torch, modules, rt = self.system()
        configure_policy(modules)
        env = PolicyEnvironment(rt, modules)
        image = np.zeros((32, 32, 3))
        env.reset(image, [0, 0, 20, 20])
        for index in range(20):
            result = env.step(image, stochastic=True)
            logits, value = policy_forward(torch, modules, result["observation"])
            logp = torch.distributions.Categorical(logits=logits).log_prob(
                torch.tensor([result["action"]]))
            torch.testing.assert_close(logp, result["log_probability"], rtol=0, atol=0)
            self.assertLessEqual(env.budget.calls, (index + 1) * 3 // 10)
        loss = value.square().mean() - logp.mean()
        loss.backward()
        self.assertIsNotNone(modules.actor.weight.grad)
        self.assertTrue(all(p.grad is None for p in modules.update.parameters()))

    def test_peek_does_not_execute_head_change_state_or_consume_rng(self):
        torch, modules, rt = self.system()
        env = PolicyEnvironment(rt, modules)
        image = np.zeros((32, 32, 3))
        env.reset(image, [0, 0, 20, 20])
        env.step(image)
        state, rng, calls = env.snapshot(), torch.get_rng_state().clone(), rt.tracker.head_calls
        peek = env.peek_value(image)
        torch.testing.assert_close(torch.get_rng_state(), rng)
        torch.testing.assert_close(env.memory, state["memory"])
        self.assertEqual(env.budget.steps, state["budget"].steps)
        self.assertEqual(rt.tracker.head_calls, calls)
        result = env.step(image)
        self.assertAlmostEqual(peek, float(result["value"].item()), places=6)


if __name__ == "__main__":
    unittest.main()
