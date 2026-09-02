"""Modular Consolidation — generic infrastructure for studying module lifecycle
(spawn / specialise / merge / compress / retire / reuse) under hard resource budgets.

Nothing in this package asserts that dynamic modularity works. It exists to make the
question measurable, including the outcome where it does not.
"""

__version__ = "0.1.0"

from .budget import BudgetLedger, BudgetCeiling, BudgetBreach
from .modules import Module, ModuleBank
from . import metrics, policies, toy

__all__ = [
    "BudgetLedger",
    "BudgetCeiling",
    "BudgetBreach",
    "Module",
    "ModuleBank",
    "metrics",
    "policies",
    "toy",
]
