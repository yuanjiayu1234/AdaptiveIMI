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
    """Choose balanced adjacent k values from current context length.

    Partition specs define only subspace layout. Runtime k is derived from the
    current token count so product(k) covers token_count / target_cluster_size.
    """
    token_count = max(int(token_count), 1)
    m = int(spec.get("subspace_parts", spec.get("M", default_m or 1)) or default_m or 1)
    m = max(m, 1)
    target = float(token_count) / float(max(int(target_cluster_size), 1))

    base = min(3, token_count)
    while True:
        product = base ** m
        if float(product) > target:
            return [base] * m

        high = min(base + 1, token_count)
        if high == base:
            return [base] * m

        for high_count in range(1, m + 1):
            product = (high ** high_count) * (base ** (m - high_count))
            if float(product) > target:
                return [high] * high_count + [base] * (m - high_count)

        base = high
