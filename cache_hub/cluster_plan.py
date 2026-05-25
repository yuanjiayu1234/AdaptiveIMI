import math
from typing import Dict, List, Optional


def product_k_values(k_values: List[int]) -> int:
    product = 1
    for value in k_values:
        product *= max(int(value), 1)
    return product


def choose_learned_k_values(
    spec: Dict[str, object],
    token_count: int,
    target_cluster_size: int,
    *,
    default_m: Optional[int] = None,
) -> List[int]:
    raw = spec.get("k_values") or spec.get("K")
    token_count = max(int(token_count), 1)
    if raw:
        return [max(1, min(int(value), token_count)) for value in raw]

    m = int(spec.get("subspace_parts", spec.get("M", default_m or 1)) or default_m or 1)
    m = max(m, 1)
    target = max(float(token_count) / float(max(int(target_cluster_size), 1)), 1.0)
    log_target = math.log(target)

    best_num_threes = 0
    best_error = float("inf")
    best_product = 0
    for num_threes in range(m + 1):
        product = (3 ** num_threes) * (2 ** (m - num_threes))
        error = abs(math.log(float(product)) - log_target)
        if error < best_error or (error == best_error and product < best_product):
            best_error = error
            best_num_threes = num_threes
            best_product = product

    values = [3] * best_num_threes + [2] * (m - best_num_threes)
    return [min(value, token_count) for value in values]
