from common import access_stats
from AFD_OFU import ofu


def dma(seq: list, variables: list, num_dbcs: int, num_domains: int):
    """Distribute variables using paper Algorithm 1; return (DBCs, K).

    The first K DBCs hold disjoint variables in first-use order. Remaining
    variables are distributed by decreasing frequency, breaking ties by first
    use (unaccessed variables last, in variable order). Infeasible partitions
    raise ValueError rather than changing the heuristic.
    """
    for name, value in (("num_dbcs", num_dbcs), ("num_domains", num_domains)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    num_variables = len(variables)
    for time, var_idx in enumerate(seq):
        if type(var_idx) is not int or not 0 <= var_idx < num_variables:
            raise ValueError(f"Invalid variable index {var_idx!r} at access {time}")
    if num_variables > num_dbcs * num_domains:
        raise ValueError(f"{num_variables} variables exceed capacity of "
                         f"{num_dbcs} DBCs x {num_domains} domains")

    freq, first, last = access_stats(seq, num_variables)
    order = sorted(range(num_variables),
                   key=lambda v: first[v] if first[v] != -1 else len(seq))
    active = [v for v in order if freq[v] > 0]
    disjoint = []
    t_min = -1
    for v in active:
        if first[v] > t_min:
            contained_freq = sum(freq[u] for u in active
                                 if first[u] > first[v] and last[u] < last[v])
            if freq[v] > contained_freq:
                disjoint.append(v)
                t_min = last[v]

    selected = set(disjoint)
    remaining = [v for v in order if v not in selected]
    remaining.sort(key=lambda v: freq[v], reverse=True)
    k = (len(disjoint) + num_domains - 1) // num_domains
    if k > num_dbcs or len(remaining) > (num_dbcs - k) * num_domains:
        raise ValueError(f"DMA partition infeasible: {k} disjoint DBCs leave "
                         f"{num_dbcs - k} DBCs for {len(remaining)} variables")

    dbcs = [[] for _ in range(num_dbcs)]
    for i, v in enumerate(disjoint):
        dbcs[i % k].append(v)
    for i, v in enumerate(remaining):
        dbcs[k + i % (num_dbcs - k)].append(v)
    return dbcs, k


def dma_ofu(seq: list, variables: list, num_dbcs: int, num_domains: int):
    """DMA distribution followed by OFU only on non-disjoint DBCs."""
    dbcs, k = dma(seq, variables, num_dbcs, num_domains)
    return dbcs[:k] + ofu(dbcs[k:], seq, len(variables))
