from common import access_stats


def afd(seq: list, variables: list, num_dbcs: int, num_domains: int):
    """Access Frequency based Distribution (paper Sec. III-A).

    Sort variables by access frequency in descending order (ties keep the
    original variable order), then distribute them round-robin over all DBCs.
    """
    num_variables = len(variables)
    if num_variables > num_dbcs * num_domains:
        raise ValueError(f"{num_variables} variables exceed capacity of "
                         f"{num_dbcs} DBCs x {num_domains} domains")

    freq, _, _ = access_stats(seq, num_variables)
    order = sorted(range(num_variables), key=lambda x: freq[x], reverse=True)

    dbcs = [[] for _ in range(num_dbcs)]
    for i, var_idx in enumerate(order):
        dbcs[i % num_dbcs].append(var_idx)

    return dbcs


def ofu(dbcs: list[list[int]], seq: list, num_variables: int):
    """Order of First Use: place variables of each DBC by first occurrence.

    Variables never accessed in seq are put at the end of their DBC.
    """
    _, first, _ = access_stats(seq, num_variables)
    never = len(seq)
    return [sorted(dbc, key=lambda x: first[x] if first[x] != -1 else never)
            for dbc in dbcs]


def afd_ofu(seq: list, variables: list, num_dbcs: int, num_domains: int):
    dbcs = afd(seq, variables, num_dbcs, num_domains)
    return ofu(dbcs, seq, len(variables))
