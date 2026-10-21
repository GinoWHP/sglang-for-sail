"""Run the four CPU-data pool cases and hook assertions with the real PPU SDK."""

import sys
from pathlib import Path

import pytest

from sglang.test.ci.ci_register import register_ppu_ci

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from ppu.module.test_ppu_deepep_hooks import TestPPUDeepEPHooks  # noqa: E402,F401
from ppu.test_kda_replayssm_pool import (  # noqa: E402,F401
    test_replayssm_pool_and_budget,
)

# Pool imports reach PPU quantization; retain these assertions on the SDK worker.
register_ppu_ci(est_time=10, suite="stage-b-test-1-gpu-ppu")

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
