from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Dict, List, Optional, Sequence

Matrix = List[List[float]]


@dataclass
class PartitionMetrics:
    objective: float
    within: float
    leakage: float
    total: float
    leak_ratio: float
    normalized_leak: float
    locality_penalty: float
    cost_proxy: float
    group_sizes: List[int]

    def to_dict(self) -> Dict[str, object]:
        return {
            "objective": float(self.objective),
            "within": float(self.within),
            "leakage": float(self.leakage),
            "total": float(self.total),
            "leak_ratio": float(self.leak_ratio),
            "normalized_leak": float(self.normalized_leak),
            "locality_penalty": float(self.locality_penalty),
            "cost_proxy": float(self.cost_proxy),
            "group_sizes": list(map(int, self.group_sizes)),
        }


@dataclass
class PartitionResult:
    m: int
    groups: List[List[int]]
    pair_to_group: List[int]
    metrics: PartitionMetrics
    seed: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "M": int(self.m),
            "groups": [list(map(int, group)) for group in self.groups],
            "pair_to_group": list(map(int, self.pair_to_group)),
            "seed": int(self.seed),
            "metrics": self.metrics.to_dict(),
        }


def _as_square(dependency) -> Matrix:
    if hasattr(dependency, "detach"):
        dependency = dependency.detach().cpu().tolist()
    arr = [[float(value) for value in row] for row in dependency]
    if not arr or any(len(row) != len(arr) for row in arr):
        raise ValueError("dependency must be a square matrix")
    n = len(arr)
    out = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                out[i][j] = 0.5 * (arr[i][j] + arr[j][i])
    return out


def _balanced_sizes(n: int, m: int, min_size: int, max_size: int) -> List[int]:
    if m * min_size > n or m * max_size < n:
        raise ValueError(f"cannot split {n} items into {m} groups with size [{min_size}, {max_size}]")
    sizes = [min_size] * m
    remaining = n - m * min_size
    idx = 0
    while remaining > 0:
        add = min(max_size - sizes[idx], remaining)
        sizes[idx] += add
        remaining -= add
        idx = (idx + 1) % m
    sizes.sort(reverse=True)
    return sizes


def _contiguous_init(n: int, m: int, min_size: int, max_size: int) -> List[List[int]]:
    sizes = _balanced_sizes(n, m, min_size, max_size)
    groups: List[List[int]] = []
    cursor = 0
    for size in sizes:
        groups.append(list(range(cursor, cursor + size)))
        cursor += size
    return groups


def _random_init(n: int, m: int, min_size: int, max_size: int, rng: random.Random) -> List[List[int]]:
    sizes = _balanced_sizes(n, m, min_size, max_size)
    order = list(range(n))
    rng.shuffle(order)
    groups: List[List[int]] = []
    cursor = 0
    for size in sizes:
        groups.append(sorted(order[cursor : cursor + size]))
        cursor += size
    return groups


def pair_to_group(groups: Sequence[Sequence[int]], n: Optional[int] = None) -> List[int]:
    if n is None:
        n = max(max(group) for group in groups if group) + 1
    out = [-1] * n
    for group_idx, group in enumerate(groups):
        for item in group:
            out[int(item)] = int(group_idx)
    if any(value < 0 for value in out):
        raise ValueError("groups do not cover all items")
    return out


def locality_penalty(groups: Sequence[Sequence[int]]) -> float:
    total = 0.0
    for group in groups:
        if len(group) <= 1:
            continue
        mean = sum(float(item) for item in group) / float(len(group))
        total += sum((float(item) - mean) ** 2 for item in group) / float(len(group))
    return total


def within_score(dependency: Matrix, groups: Sequence[Sequence[int]]) -> float:
    score = 0.0
    for group in groups:
        if len(group) <= 1:
            continue
        for pos, u in enumerate(group):
            for v in group[pos + 1 :]:
                score += dependency[int(u)][int(v)]
    return score


def compute_metrics(
    dependency,
    groups: Sequence[Sequence[int]],
    *,
    gamma: float = 0.0,
) -> PartitionMetrics:
    dep = _as_square(dependency)
    n = len(dep)
    within = within_score(dep, groups)
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            total += dep[i][j]
    leakage = max(total - within, 0.0)
    leak_ratio = leakage / total if total > 0 else 0.0
    within_pairs = sum(len(group) * (len(group) - 1) / 2.0 for group in groups)
    all_pairs = n * (n - 1) / 2.0
    cross_pair_fraction = 1.0 - within_pairs / all_pairs if all_pairs > 0 else 0.0
    random_leak = total * cross_pair_fraction
    normalized_leak = leakage / random_leak if random_leak > 0 else 0.0
    loc = locality_penalty(groups)
    objective = within - float(gamma) * loc

    # KMeans proxy for equal total virtual-cell budget. The constant is not
    # important; it lets M values be compared monotonically in reports.
    cost_proxy = sum((2.0 * len(group)) for group in groups)

    return PartitionMetrics(
        objective=objective,
        within=within,
        leakage=leakage,
        total=total,
        leak_ratio=leak_ratio,
        normalized_leak=normalized_leak,
        locality_penalty=loc,
        cost_proxy=cost_proxy,
        group_sizes=[len(group) for group in groups],
    )


def _group_objective(dep: Matrix, group: Sequence[int], gamma: float) -> float:
    score = within_score(dep, [group])
    loc = locality_penalty([group])
    return score - gamma * loc


def _move_gain(dep: Matrix, groups: List[List[int]], src: int, dst: int, item: int, gamma: float) -> float:
    before = _group_objective(dep, groups[src], gamma) + _group_objective(dep, groups[dst], gamma)
    src_new = [x for x in groups[src] if x != item]
    dst_new = sorted(groups[dst] + [item])
    after = _group_objective(dep, src_new, gamma) + _group_objective(dep, dst_new, gamma)
    return after - before


def _swap_gain(dep: Matrix, groups: List[List[int]], a: int, b: int, item_a: int, item_b: int, gamma: float) -> float:
    before = _group_objective(dep, groups[a], gamma) + _group_objective(dep, groups[b], gamma)
    group_a = sorted([x for x in groups[a] if x != item_a] + [item_b])
    group_b = sorted([x for x in groups[b] if x != item_b] + [item_a])
    after = _group_objective(dep, group_a, gamma) + _group_objective(dep, group_b, gamma)
    return after - before


def optimize_partition(
    dependency,
    m: int,
    *,
    min_size: int = 2,
    max_size: int = 32,
    gamma: float = 0.01,
    seeds: int = 16,
    max_iter: int = 100,
    swap_passes: int = 4,
    random_seed: int = 0,
) -> PartitionResult:
    dep = _as_square(dependency)
    n = len(dep)
    if m <= 0:
        raise ValueError("m must be positive")
    if m * min_size > n or m * max_size < n:
        raise ValueError(f"M={m} cannot satisfy size constraints for {n} pairs")

    initializations: List[tuple[int, List[List[int]]]] = [(-1, _contiguous_init(n, m, min_size, max_size))]
    rng_master = random.Random(random_seed)
    for seed_idx in range(seeds):
        seed = int(rng_master.randrange(0, 2**31 - 1))
        initializations.append((seed, _random_init(n, m, min_size, max_size, random.Random(seed))))

    best_groups: Optional[List[List[int]]] = None
    best_seed = -1
    best_metrics: Optional[PartitionMetrics] = None

    for seed, init_groups in initializations:
        groups = [sorted(group) for group in init_groups]
        rng = random.Random(seed if seed >= 0 else random_seed)

        for _ in range(max_iter):
            improved = False
            items = list(range(n))
            rng.shuffle(items)
            assignment = pair_to_group(groups, n)
            for item in items:
                src = assignment[item]
                if len(groups[src]) <= min_size:
                    continue
                best_dst = None
                best_gain = 1e-12
                for dst in range(m):
                    if dst == src or len(groups[dst]) >= max_size:
                        continue
                    gain = _move_gain(dep, groups, src, dst, item, gamma)
                    if gain > best_gain:
                        best_gain = gain
                        best_dst = dst
                if best_dst is not None:
                    groups[src].remove(item)
                    groups[best_dst].append(item)
                    groups[best_dst].sort()
                    assignment[item] = best_dst
                    improved = True

            for _swap_pass in range(swap_passes):
                swap_improved = False
                for a in range(m):
                    for b in range(a + 1, m):
                        best_pair = None
                        best_gain = 1e-12
                        for item_a in groups[a]:
                            for item_b in groups[b]:
                                gain = _swap_gain(dep, groups, a, b, item_a, item_b, gamma)
                                if gain > best_gain:
                                    best_gain = gain
                                    best_pair = (item_a, item_b)
                        if best_pair is not None:
                            item_a, item_b = best_pair
                            groups[a].remove(item_a)
                            groups[b].remove(item_b)
                            groups[a].append(item_b)
                            groups[b].append(item_a)
                            groups[a].sort()
                            groups[b].sort()
                            swap_improved = True
                            improved = True
                if not swap_improved:
                    break

            if not improved:
                break

        metrics = compute_metrics(dep, groups, gamma=gamma)
        if best_metrics is None or metrics.objective > best_metrics.objective:
            best_groups = [list(group) for group in groups]
            best_metrics = metrics
            best_seed = seed

    if best_groups is None or best_metrics is None:
        raise RuntimeError("partition search failed")

    return PartitionResult(
        m=int(m),
        groups=best_groups,
        pair_to_group=pair_to_group(best_groups, n),
        metrics=best_metrics,
        seed=int(best_seed),
    )


def choose_by_constraint(
    results: Sequence[PartitionResult],
    *,
    max_normalized_leak: float,
) -> Optional[PartitionResult]:
    feasible = [
        result for result in results
        if result.metrics.normalized_leak <= max_normalized_leak
    ]
    if not feasible:
        return None
    feasible.sort(key=lambda item: (item.m, -item.metrics.normalized_leak), reverse=True)
    return feasible[0]
