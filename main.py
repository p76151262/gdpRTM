import itertools

from common import parse_file, shift_cost
from AFD_OFU import afd_ofu
from DMA_OFU import dma, dma_ofu


def algo1(seq: list, variables: list, num_dbcs: int, num_domains: int):
    """Compatibility entry point for DMA distribution, without OFU ordering."""
    dbcs, _ = dma(seq, variables, num_dbcs, num_domains)
    return dbcs


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

        seq = list(itertools.chain.from_iterable(seqs))

        for name, method in (("AFD-OFU", afd_ofu), ("DMA-OFU", dma_ofu)):
            dbcs = method(seq, variables, 2, 512)
            shifts = shift_cost(dbcs, seq)
            print(f"{name} DBCs: {[[variables[v] for v in dbc] for dbc in dbcs]}")
            print(f"{name} shift cost: {shifts} (total {sum(shifts)})")




if __name__ == "__main__":
    main()
