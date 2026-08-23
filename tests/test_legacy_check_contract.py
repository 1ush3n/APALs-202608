from __future__ import annotations

import importlib
from types import ModuleType

import pytest


LEGACY_CHECK_MODULES = (
    "tests.test_reward_and_grad",
    "tests.test_domain_randomization",
    "tests.test_engine_and_mask",
)


@pytest.mark.parametrize("module_name", LEGACY_CHECK_MODULES)
def test_legacy_check_propagates_failure_to_pytest(module_name: str) -> None:
    module: ModuleType = importlib.import_module(module_name)
    original_state = (
        module.TOTAL_TESTS,
        module.PASSED_TESTS,
        list(module.FAILED_TESTS),
    )
    failure_name = "pytest 必须收到这个失败"
    try:
        with pytest.raises(AssertionError, match=failure_name):
            module.check(False, failure_name)
    finally:
        module.TOTAL_TESTS = original_state[0]
        module.PASSED_TESTS = original_state[1]
        module.FAILED_TESTS[:] = original_state[2]
