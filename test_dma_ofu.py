import copy
from pathlib import Path
import subprocess
import sys
import unittest

from AFD_OFU import afd_ofu
from common import parse_file, shift_cost
from DMA_OFU import dma, dma_ofu
from main import algo1


class DMAOFUTests(unittest.TestCase):
    def setUp(self):
        instance = parse_file(Path(__file__).parent / 'seq' / 'example.seq')['example']
        self.variables = instance['vars']
        self.seq = instance['seqs'][0]

    def assert_valid(self, dbcs, nvars, q, capacity):
        self.assertEqual(len(dbcs), q)
        self.assertEqual(sorted(v for dbc in dbcs for v in dbc), list(range(nvars)))
        self.assertTrue(all(len(dbc) <= capacity for dbc in dbcs))

    def test_example_distribution_and_ordering(self):
        raw, k = dma(self.seq, self.variables, 2, 512)
        self.assertEqual(k, 1)
        self.assertEqual(raw, [[1, 2, 3, 4, 7], [0, 8, 6, 5]])
        ordered = dma_ofu(self.seq, self.variables, 2, 512)
        self.assertEqual(ordered, [[1, 2, 3, 4, 7], [0, 8, 5, 6]])
        self.assertEqual(ordered[:k], raw[:k])
        self.assertEqual(shift_cost(ordered, self.seq), [4, 5])
        self.assert_valid(ordered, 9, 2, 512)
        self.assertEqual(algo1(self.seq, self.variables, 2, 512), raw)

    def test_afd_baseline_unchanged(self):
        dbcs = afd_ofu(self.seq, self.variables, 2, 512)
        self.assertEqual(dbcs, [[0, 1, 3, 6, 7], [2, 8, 4, 5]])
        self.assertEqual(shift_cost(dbcs, self.seq), [14, 7])

    def test_multiple_disjoint_dbcs_round_robin(self):
        seq = [4, 4, 2, 2, 0, 0, 3, 3, 1, 1]
        dbcs, k = dma(seq, list('abcde'), 3, 2)
        self.assertEqual(k, 3)
        self.assertEqual(dbcs, [[4, 3], [2, 1], [0]])
        self.assertEqual(dma_ofu(seq, list('abcde'), 3, 2), dbcs)
        self.assert_valid(dbcs, 5, 3, 2)

    def test_frequency_ties_follow_first_use(self):
        # The outer variable ties the contained frequency, so strict > rejects it.
        dbcs, k = dma([2, 0, 1, 2], list('abc'), 2, 3)
        self.assertEqual((dbcs, k), ([[0, 1], [2]], 1))
        # Variables 0 and 1 tie in frequency, but 1 is first used earlier.
        dbcs, _ = dma([1, 0, 2, 3, 0, 1], list('abcd'), 2, 4)
        self.assertEqual(dbcs[1], [1, 0])

    def test_empty_sequence_and_zero_variables(self):
        self.assertEqual(dma([], [], 2, 2), ([[], []], 0))
        self.assertEqual(dma_ofu([], [], 2, 2), [[], []])
        self.assertEqual(dma([], list('abcd'), 2, 2), ([[0, 2], [1, 3]], 0))
        self.assertEqual(dma_ofu([], list('abcd'), 2, 2), [[0, 2], [1, 3]])

    def test_unaccessed_variables_are_remaining_and_last(self):
        self.assertEqual(dma([1, 1], list('abcd'), 2, 3),
                         ([[1], [0, 2, 3]], 1))
        seq = [0, 1, 0, 1]
        self.assertEqual(dma_ofu(seq, list('abcd'), 2, 4), [[0], [1, 2, 3]])

    def test_single_variable_and_all_disjoint(self):
        self.assertEqual(dma_ofu([0, 0], ['a'], 1, 1), [[0]])
        self.assertEqual(dma([2, 0, 1], list('abc'), 1, 3), ([[2, 0, 1]], 1))

    def test_inputs_are_not_modified(self):
        before = copy.deepcopy((self.seq, self.variables))
        dma(self.seq, self.variables, 2, 512)
        dma_ofu(self.seq, self.variables, 2, 512)
        self.assertEqual((self.seq, self.variables), before)

    def test_invalid_dimensions(self):
        for q, capacity in ((0, 2), (-1, 2), (2, 0), (2, -1),
                            (1.5, 2), (2, 1.5), (True, 2), (2, False)):
            with self.subTest(q=q, capacity=capacity):
                with self.assertRaisesRegex(ValueError, 'positive integer'):
                    dma([], [], q, capacity)

    def test_invalid_indices(self):
        for seq in ([-1], [1], [0.0], [True], ['0'], [None]):
            with self.subTest(seq=seq):
                with self.assertRaisesRegex(ValueError, 'Invalid variable index'):
                    dma(seq, ['a'], 2, 2)
        with self.assertRaises(ValueError):
            dma([0], [], 2, 2)

    def test_insufficient_total_capacity(self):
        with self.assertRaisesRegex(ValueError, 'exceed capacity'):
            dma([], list('abc'), 1, 2)

    def test_infeasible_partition_despite_sufficient_total_capacity(self):
        # a has frequency 7 > the contained frequency 6, reserving DBC0.
        # The three remaining variables cannot fit in DBC1.
        with self.assertRaisesRegex(ValueError, 'DMA partition infeasible'):
            dma([0, 1, 0, 2, 0, 3, 0, 1, 0, 2, 0, 3, 0], list('abcd'), 2, 2)
        # K equals q but an overlapping variable remains.
        with self.assertRaisesRegex(ValueError, 'DMA partition infeasible'):
            dma([0, 1, 0, 1], list('ab'), 1, 2)

    def test_main_reports_both_methods(self):
        result = subprocess.run([sys.executable, '-B', 'main.py'],
                                cwd=Path(__file__).parent, capture_output=True,
                                text=True, timeout=10, check=True)
        self.assertIn('AFD-OFU shift cost: [14, 7] (total 21)', result.stdout)
        self.assertIn('DMA-OFU shift cost: [4, 5] (total 9)', result.stdout)


if __name__ == '__main__':
    unittest.main()
