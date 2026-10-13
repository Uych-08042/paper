"""Pure helpers for deterministic periodic closed-loop policies."""

import math


PERIODIC_L9_BUDGETS = {
    "periodic_l9_10": 10,
    "periodic_l9_20": 20,
    "periodic_l9_30": 30,
    "periodic_l9_50": 50,
}


def periodic_l9_active(policy, decision_index):
    """Return whether a one-based decision spends the next L9 token."""
    if policy not in PERIODIC_L9_BUDGETS:
        raise ValueError("Not a periodic L9 policy: {}".format(policy))
    if decision_index is None:
        raise ValueError("Periodic L9 policies require a decision index")
    if decision_index < 1:
        return False
    rate = PERIODIC_L9_BUDGETS[policy] / 100.0
    completed = int(math.floor(decision_index * rate + 1e-12))
    previous = int(math.floor((decision_index - 1) * rate + 1e-12))
    return completed > previous

