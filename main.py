import sys
import itertools
import math


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


def algo1(seq: list, variables: list, num_dbcs: int, num_domains: int):
    num_variables = len(variables)
    freq = [0] * num_variables
    first = [-1] * num_variables
    last = [0] * num_variables

    for time, var_idx in enumerate(seq):
        freq[var_idx] += 1
        if first[var_idx] == -1:
            first[var_idx] = time
        if last[var_idx] < time:
            last[var_idx] = time

    print(f"vars: {variables}")
    print(f"freq: {freq}")
    print(f"first: {first}")
    print(f"last: {last}")

    # sort by first appearence time in ascending order
    ndisjoint = [item for _, item in sorted(zip(first, range(num_variables)))]
    print(f"sorted(ndj): {ndisjoint}")

    disjoint = []
    t_min = -1
    for var_idx in ndisjoint:
        if var_idx in disjoint:
            continue
        var_freq = freq[var_idx]
        t_begin = first[var_idx]
        t_end = last[var_idx]

        if t_begin > t_min:
            freq_sum = sum(freq[v] for v in ndisjoint if first[v]
                           > t_begin and last[v] < t_end)
            if var_freq > freq_sum:
                disjoint.append(var_idx)
                t_min = t_end

    # remove disjoint variables from ndj, set for speed up
    ds_set = set(disjoint)
    ndisjoint = [x for x in ndisjoint if x not in ds_set]

    # sort by frequency in descending order
    ndisjoint.sort(key=lambda x: freq[x], reverse=True)

    print(f"ndisjoint: {[variables[x] for x in ndisjoint]}")
    print(f"disjoint: {[variables[x] for x in disjoint]}")

    # k is number of DBCs required for ndj variables
    k = math.ceil(len(disjoint) / num_domains)

    print(f"k: {k}")

    # init num_dbcs empty lists
    dbcs = [[] for _ in range(num_dbcs)]

    # disjoint is already sorted by first occurence order
    while disjoint:
        for i in range(k):  # round-robin distribution among k dbcs
            if not disjoint:  # no more disjoint vars
                break
            if len(dbcs[i]) < num_domains:
                dbcs[i].append(disjoint.pop(0))

    # ndj is sorted by frequency in descending order
    while ndisjoint:
        for i in range(k, num_dbcs):
            if not ndisjoint:
                break
            if len(dbcs[i]) < num_domains:
                dbcs[i].append(ndisjoint.pop(0))

    print(f"DBCs: {dbcs}")
    print(f"DBCs: {[[variables[v] for v in dbc] for dbc in dbcs]}")

    # TODO: apply Chen(S-DBC-P) or ShiftReduce on dbcs[k] to dbcs[-1]


def main():
    parsed = parse_file("./seq/example.seq")

    for instance in parsed:
        print(f"instance: {instance}")
        variables = parsed[instance]["vars"]
        seqs = parsed[instance]["seqs"]
        print(f"    {len(variables)} vars: {variables}")
        print(f"    {len(seqs)} seqs:")
        for seq in seqs:
            print(f"{8 * ' '}{seq}")

        algo1(list(itertools.chain.from_iterable(
            seqs)), parsed[instance]["vars"], 2, 512)


if __name__ == "__main__":
    main()
