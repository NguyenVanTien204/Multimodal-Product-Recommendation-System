from __future__ import annotations

from dataclasses import dataclass

import psutil

GIB = 1024**3


@dataclass(frozen=True)
class MemoryPlan:
    total_bytes: int
    available_bytes: int
    budget_bytes: int
    reserve_bytes: int
    batch_rows: int

    @property
    def budget_gb(self) -> float:
        return self.budget_bytes / GIB


def make_memory_plan(config: dict) -> MemoryPlan:
    """Create a conservative plan from live memory, never from installed RAM alone."""
    vm = psutil.virtual_memory()
    reserve = int(float(config.get("reserve_gb", 2.0)) * GIB)
    usable_after_reserve = max(0, int(vm.available) - reserve)
    minimum_budget = int(float(config.get("min_budget_mb", 64)) * 1024**2)
    if usable_after_reserve < minimum_budget:
        raise MemoryError(
            f"Only {vm.available / GIB:.2f} GiB RAM is available; cannot preserve the "
            f"configured {reserve / GIB:.2f} GiB reserve and a "
            f"{minimum_budget / 1024**2:.0f} MiB working budget. Close other programs or lower "
            "reserve_gb deliberately."
        )
    candidates = [
        usable_after_reserve,
        int(vm.available * float(config.get("available_fraction", 0.5))),
        int(vm.total * float(config.get("total_fraction", 0.25))),
    ]
    if config.get("max_memory_gb") is not None:
        candidates.append(int(float(config["max_memory_gb"]) * GIB))
    budget = max(minimum_budget, min(candidates))

    initial = int(config.get("initial_batch_rows", 10_000))
    minimum = int(config.get("min_batch_rows", 1_000))
    maximum = int(config.get("max_batch_rows", 50_000))
    # Assume a deliberately high 16 KiB/raw row until observed sizes can tune it.
    safe_rows = max(minimum, budget // (16 * 1024 * 4))
    batch_rows = min(maximum, initial, safe_rows)
    return MemoryPlan(vm.total, vm.available, budget, reserve, int(batch_rows))


def memory_pressure(plan: MemoryPlan) -> bool:
    return psutil.virtual_memory().available < plan.reserve_bytes + plan.budget_bytes // 4
