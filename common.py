import sys
import itertools


def parse_file(filename="./seq/example.seq") -> dict[str, dict[str, list[str]]]:
    parsed = {}

    try:
        with open(filename, "r") as f:
            lines = f.readlines()
            lines = [x.split() for x in lines if x != "\n"]
            toks = list(itertools.chain.from_iterable(lines))
            i = 0
            while i < len(toks):
                if toks[i] == "instance":
                    instance_name = toks[i + 1]
                    num_vars = int(toks[i + 2])
                    var_names = toks[i + 4:i + 4 + num_vars]
                    parsed[instance_name] = {}
                    parsed[instance_name]["vars"] = var_names
                    parsed[instance_name]["seqs"] = []
                    i = i + 4 + num_vars
                    num_seqs = int(toks[i])
                    i += 2
                    while num_seqs > 0:
                        i += 2
                        seq_len = int(toks[i])
                        seq = [int(x) for x in toks[i + 1:i + 1 + seq_len]]
                        parsed[instance_name]["seqs"].append(seq)
                        i += seq_len + 1
                        num_seqs -= 1
    except IOError as err:
        print(f"{err}")
        sys.exit(1)

    return parsed


def access_stats(seq: list, num_variables: int):
    freq = [0] * num_variables
    first = [-1] * num_variables
    last = [0] * num_variables

    for time, var_idx in enumerate(seq):
        freq[var_idx] += 1
        if first[var_idx] == -1:
            first[var_idx] = time
        if last[var_idx] < time:
            last[var_idx] = time

    return freq, first, last


def shift_cost(dbcs: list[list[int]], seq: list) -> list[int]:
    """Number of shifts per DBC, single access port per DBC.

    The port is assumed to be aligned to the first accessed variable of each
    DBC, so the first access to a DBC is free (as in Fig. 3 of the paper).
    """
    loc = {}
    for dbc_idx, dbc in enumerate(dbcs):
        for offset, var_idx in enumerate(dbc):
            loc[var_idx] = (dbc_idx, offset)

    shifts = [0] * len(dbcs)
    port = [None] * len(dbcs)
    for var_idx in seq:
        dbc_idx, offset = loc[var_idx]
        if port[dbc_idx] is not None:
            shifts[dbc_idx] += abs(offset - port[dbc_idx])
        port[dbc_idx] = offset

    return shifts
