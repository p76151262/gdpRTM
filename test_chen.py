
import unittest
from copy import deepcopy

from Chen import (
    build_access_graph,
    chen,
    group_weight,
    grouping,
    inter_group_weight,
    order_groups,
    place_groups,
    placement,
    total_edge_weights,
    vertex_group_weight,
)


PAPER_ACCESSES = tuple("b h b h e d c a f c e h f d g g h g d a d g b c".split())
# Explicit project test inputs, not a complete schedule specified by the paper.
PAPER_THRESHOLDS = ((0.35, 0.5), (0, 0))


def make_graph(vertices, edges):
    """Build an undirected graph from explicit weighted edges."""
    graph = {vertex: {} for vertex in vertices}
    for left, right, weight in edges:
        graph[left][right] = weight
        graph[right][left] = weight
    return graph


def complete_graph(size):
    """Build a complete graph with unit edge weights for threshold tests."""
    return make_graph(range(size), ((i, j, 1) for i in range(size)
                                   for j in range(i + 1, size)))


def paper_example(variable_order="dhgcbaef"):
    """Return fresh variables, sequence and access graph for Figs. 4-5."""
    variables = list(variable_order)
    seq = [variables.index(vertex) for vertex in PAPER_ACCESSES]
    return variables, seq, build_access_graph(seq, variables)


class TestChenAccessGraph(unittest.TestCase):

    def test_access_graph(self):
        variables = ["A", "B", "C"]
        seq = [0, 1, 0, 2, 1, 0]

        graph = build_access_graph(seq, variables)

        self.assertEqual(graph[0][1], 3)
        self.assertEqual(graph[0][2], 1)
        self.assertEqual(graph[1][2], 1)

        self.assertEqual(graph[1][0], 3)

    def test_total_edge_weights(self):
        variables = ["A", "B", "C"]
        seq = [0, 1, 0, 2, 1, 0]

        graph = build_access_graph(seq, variables)
        weights = total_edge_weights(graph)

        self.assertEqual(weights[0], 4)
        self.assertEqual(weights[1], 4)
        self.assertEqual(weights[2], 2)

    def test_repeated_accesses(self):
        graph = build_access_graph(
            [0, 0, 1, 1], ["A", "B"]
        )
        self.assertEqual(graph[0][1], 1)

    def test_invalid_variable(self):
        with self.assertRaises(ValueError):
            build_access_graph([0, 3], ["A", "B"])


class TestChenGrouping(unittest.TestCase):

    def assert_partition(self, groups, graph, num_blocks, num_ports):
        flattened = [vertex for group in groups for vertex in group]
        self.assertEqual(len(groups), num_blocks // num_ports)
        self.assertTrue(all(len(group) <= num_ports for group in groups))
        self.assertCountEqual(flattened, graph)

    def test_group_weight_includes_external_edges_and_both_internal_endpoints(self):
        graph = build_access_graph([0, 1, 0, 2, 1, 0], ["A", "B", "C"])
        weights = total_edge_weights(graph)
        self.assertEqual(group_weight([], weights), 0)
        self.assertEqual(group_weight([0], weights), 4)
        # Internal edge 0--1 contributes 2 * 3; external edges contribute 1 + 1.
        self.assertEqual(group_weight([0, 1], weights), 8)

    def test_vertex_group_weight_sums_edges_and_treats_missing_edges_as_zero(self):
        graph = make_graph(range(4), [(0, 1, 3), (0, 2, 2)])
        self.assertEqual(vertex_group_weight(graph, 0, []), 0)
        self.assertEqual(vertex_group_weight(graph, 0, [1, 2, 3]), 5)

    def test_default_ports_and_thresholds_reuse_access_graph(self):
        graph = build_access_graph([0, 1, 0, 2, 1, 0], ["A", "B", "C"])
        groups = grouping(graph, num_blocks=4)
        self.assertEqual(groups, [[0, 1], [2]])
        self.assert_partition(groups, graph, 4, 2)

    def test_alpha_condition_is_inclusive(self):
        graph = make_graph(range(3), [(0, 1, 2), (0, 2, 2)])
        # Phi(1, {0}) = 2 = 0.5 * W_0; beta is deliberately nonbinding.
        self.assertEqual(grouping(graph, 4, alpha=0.5, beta=0), [[0, 1], [2]])

    def test_beta_condition_is_inclusive(self):
        graph = make_graph(range(3), [(0, 1, 2), (1, 2, 2)])
        # Vertex 1 is the seed; Phi(0, {1}) = 2 = 1.0 * W_0.
        self.assertEqual(grouping(graph, 4, alpha=0, beta=1), [[1, 0], [2]])

    def test_alpha_failure_rejects_even_when_beta_passes(self):
        graph = make_graph(range(3), [(0, 1, 2), (0, 2, 2)])
        with self.assertRaisesRegex(ValueError, "Grouping stalled"):
            grouping(graph, 4, alpha=0.51, beta=0)

    def test_beta_failure_rejects_even_when_alpha_passes(self):
        graph = complete_graph(4)
        with self.assertRaisesRegex(ValueError, "Grouping stalled"):
            grouping(graph, 4, alpha=0, beta=0.34)

    def test_group_weight_updates_after_each_addition(self):
        graph = make_graph(range(3), [(0, 1, 4), (0, 2, 3), (1, 2, 2)])
        # W_0=7, W_1=6. Vertex 1 passes: 4 >= 0.5*7. Vertex 2 then has
        # Phi=5 < 0.5*(7+6); using a stale seed weight would wrongly admit it.
        with self.assertRaisesRegex(ValueError, "Grouping stalled"):
            grouping(graph, 3, num_ports=3, alpha=0.5, beta=0)
        self.assertEqual(grouping(graph, 3, num_ports=3, alpha=0.5, beta=0,
                                  threshold_schedule=[(0.38, 0)]), [[0, 1, 2]])

    def test_refinement_maximizes_affinity_instead_of_queue_priority(self):
        graph = make_graph(range(4), [(0, 1, 8), (0, 2, 2),
                                    (0, 3, 4), (1, 2, 6), (2, 3, 1)])
        # Initial Q is 0,1,2,3. Thresholds 1/1 leave singleton seeds 0,1.
        # After explicit relaxation, 3 has higher Phi to {0} than 2 does,
        # even though 2 has higher W_i and occurs earlier in Q.
        groups = grouping(graph, 4, alpha=1, beta=1,
                          threshold_schedule=[(0.1, 0.1)])
        self.assertEqual(groups, [[0, 3], [1, 2]])

    def test_initialization_scans_queue_in_order(self):
        graph = make_graph(range(4), [(0, 1, 8), (0, 2, 2),
                                    (0, 3, 4), (1, 2, 6), (2, 3, 1)])
        self.assertEqual(grouping(graph, 4, alpha=0, beta=0), [[0, 1], [2, 3]])

    def test_stall_does_not_silently_relax_thresholds(self):
        graph = complete_graph(4)
        with self.assertRaisesRegex(ValueError, "provide an explicit threshold_schedule"):
            grouping(graph, 4)
        self.assertEqual(grouping(graph, 4, threshold_schedule=[(0.3, 0.3)]),
                         [[0, 2], [1, 3]])

    def test_refinement_skips_a_larger_affinity_that_fails_beta(self):
        graph = make_graph(range(5), [(0, 1, 60), (0, 2, 20), (0, 3, 15),
                                    (1, 2, 30), (1, 3, 5), (1, 4, 10),
                                    (2, 4, 20)])
        # Seeds are 1 and 0. At 0/0.5, no candidate qualifies for {1}.
        # For {0}, candidate 2 has Phi=20 but fails beta (W_2=70), whereas
        # candidate 3 has Phi=15 and passes (W_3=20), so choose 3 now.
        # At 0/0, the next pass assigns 2 to group 1 and 4 to group 0.
        groups = grouping(graph, 6, num_ports=3, alpha=1, beta=1,
                          threshold_schedule=[(0, 0.5), (0, 0)])
        self.assertEqual(groups, [[1, 2], [0, 3, 4]])
        self.assert_partition(groups, graph, 6, 3)

    def test_exhausted_schedule_keeps_last_thresholds_for_further_passes(self):
        graph = complete_graph(6)
        # All seeds remain alone at 0.5/0.5. One 0/0 relaxation admits one
        # vertex per group per pass; a second pass must reuse the last pair.
        groups = grouping(graph, 6, num_ports=3, threshold_schedule=[(0, 0)])
        self.assertEqual(groups, [[0, 2, 4], [1, 3, 5]])

    def test_exhausted_nonzero_schedule_can_still_stall(self):
        graph = complete_graph(4)
        with self.assertRaisesRegex(ValueError, "alpha=0.4, beta=0.4"):
            grouping(graph, 4, threshold_schedule=[(0.4, 0.4)])

    def test_paper_figure_5_with_explicit_schedule_and_tie_order(self):
        # Fig. 4(b)'s access sequence, p. 3096. The insertion order explicitly
        # reproduces Fig. 5(a)'s tie choices, NOT an inferred paper tie rule.
        variables, _, graph = paper_example()
        # PAPER_THRESHOLDS includes the paper's 0.35/0.5 example; its final
        # 0/0 pair is an explicit project test input, not an author's parameter.
        groups = grouping(graph, 8, threshold_schedule=PAPER_THRESHOLDS)
        named = [[variables[v] for v in group] for group in groups]
        self.assertEqual(named, [["d", "a"], ["h", "b"], ["g", "e"], ["c", "f"]])
        self.assert_partition(groups, graph, 8, 2)

    def test_configurable_port_count_and_unique_partition(self):
        graph = build_access_graph([0, 1, 2, 3, 4, 0], list("abcde"))
        for ports, blocks in ((1, 5), (2, 6), (3, 6), (4, 8), (8, 8)):
            with self.subTest(ports=ports):
                groups = grouping(graph, blocks, num_ports=ports,
                                  threshold_schedule=[(0, 0)])
                self.assert_partition(groups, graph, blocks, ports)

    def test_isolated_vertices_are_not_lost(self):
        graph = build_access_graph([], list("abcde"))
        groups = grouping(graph, 6)
        self.assertEqual(groups, [[0, 1], [2, 3], [4]])
        self.assert_partition(groups, graph, 6, 2)

    def test_empty_graph_and_unused_capacity(self):
        self.assertEqual(grouping({}, 4), [[], []])
        self.assertEqual(grouping({7: {}}, 6), [[7], [], []])

    def test_ties_preserve_graph_insertion_order(self):
        graph = make_graph([9, 3, 7, 1], [])
        self.assertEqual(grouping(graph, 4), [[9, 3], [7, 1]])

    def test_input_is_unchanged_on_success_and_stall(self):
        graph = complete_graph(4)
        original = deepcopy(graph)
        grouping(graph, 4, threshold_schedule=[(0, 0)])
        self.assertEqual(graph, original)
        with self.assertRaises(ValueError):
            grouping(graph, 4)
        self.assertEqual(graph, original)

    def test_invalid_capacity_or_port_count(self):
        graph = build_access_graph([], list("abc"))
        for blocks, ports in ((0, 2), (-1, 2), (4.0, 2), (True, 1),
                              (4, 0), (4, -1), (4, 1.5), (4, True),
                              (2, 2), (3, 2)):
            with self.subTest(blocks=blocks, ports=ports):
                with self.assertRaises(ValueError):
                    grouping(graph, blocks, num_ports=ports)

    def test_invalid_thresholds(self):
        for value in (-0.1, float("nan"), float("inf"), "0.5", True):
            for name in ("alpha", "beta"):
                with self.subTest(name=name, value=value):
                    with self.assertRaises(ValueError):
                        grouping({}, 2, **{name: value})

    def test_schedule_must_be_valid_and_non_increasing(self):
        for schedule in ([(0.6, 0.5)], [(0.5, 0.6)], [(0.3, 0.3), (0.4, 0.2)],
                         [(-1, 0)], [(0, float("nan"))]):
            with self.subTest(schedule=schedule):
                with self.assertRaises(ValueError):
                    grouping({}, 2, threshold_schedule=schedule)


class TestChenPlacement(unittest.TestCase):

    def test_inter_group_weight_counts_cross_edges_once(self):
        graph = make_graph(
            range(4), [(0, 1, 20), (2, 3, 30), (0, 2, 3),
                       (0, 3, 1), (1, 2, 2), (1, 3, 4)]
        )
        self.assertEqual(inter_group_weight(graph, [0, 1], [2, 3]), 10)
        self.assertEqual(inter_group_weight(graph, [2, 3], [0, 1]), 10)
        self.assertEqual(inter_group_weight(graph, [], [2, 3]), 0)

    def test_paper_figure_5_ordered_groups_and_shift_cost(self):
        variables, seq, graph = paper_example("abcdefgh")
        groups = [[variables.index(v) for v in pair]
                  for pair in ("da", "hb", "ge", "cf")]
        # Fig. 5(b) explicitly supplies order 3,0,2,1. This verifies the
        # address stage, not an inferred Algorithm 2 selection-cost formula.
        layout = place_groups([groups[i] for i in [3, 0, 2, 1]], 8)
        self.assertEqual([variables[v] for v in layout], list("cdghfaeb"))
        offsets = {vertex: address % 4 for address, vertex in enumerate(layout)}
        shifts = sum(abs(offsets[u] - offsets[v]) for u, v in zip(seq, seq[1:]))
        self.assertEqual(shifts, 21)  # Paper p. 3100, Fig. 5(c).
        self.assertEqual([inter_group_weight(graph, groups[i], groups[1])
                          for i in [0, 2, 3]], [0, 5, 2])

    def test_three_ports_put_group_members_in_distinct_segments(self):
        layout = place_groups([[0, 1, 2], [3, 4, 5]], 6, num_ports=3)
        self.assertEqual(layout, [0, 3, 1, 4, 2, 5])

    def test_one_port_preserves_order(self):
        self.assertEqual(place_groups([[2], [0], [1]], 3, num_ports=1), [2, 0, 1])

    def test_unused_blocks_preserve_offsets_and_segment_boundaries(self):
        groups = [[7], [], [3, 5]]
        self.assertEqual(place_groups(groups, 6), [7, None, 3, None, None, 5])
        self.assertEqual(place_groups([[], []], 4), [None] * 4)

    def test_member_order_selects_ports(self):
        self.assertEqual(place_groups([[4, 2]], 2), [4, 2])
        self.assertEqual(place_groups([[2, 4]], 2), [2, 4])

    def test_inputs_are_unchanged_on_success_and_error(self):
        groups = [[0, 2], [1]]
        original = deepcopy(groups)
        place_groups(groups, 4)
        self.assertEqual(groups, original)
        invalid = [[0, 2], [2]]
        original_invalid = deepcopy(invalid)
        with self.assertRaises(ValueError):
            place_groups(invalid, 4)
        self.assertEqual(invalid, original_invalid)

    def test_invalid_capacity_or_ports(self):
        for blocks, ports in ((0, 2), (-2, 2), (4.0, 2), (True, 1),
                              (4, 0), (4, -1), (4, 1.5), (4, True), (3, 2)):
            with self.subTest(blocks=blocks, ports=ports):
                with self.assertRaises(ValueError):
                    place_groups([], blocks, num_ports=ports)

    def test_invalid_group_count_size_or_members(self):
        for groups in ([[0]], [[0], [1], []], [[0, 1, 2], []],
                       [[0, 0], []], [[0], [0]], [[-1], []],
                       [[True], []], [[1.5], []], [["0"], []]):
            with self.subTest(groups=groups):
                with self.assertRaises(ValueError):
                    place_groups(groups, 4)

    def test_grouping_output_can_be_mapped_without_flattening_groups(self):
        graph = build_access_graph([0, 1, 0, 2, 1, 0], list("abc"))
        groups = grouping(graph, 4)
        self.assertEqual(groups, [[0, 1], [2]])
        self.assertEqual(place_groups(groups, 4), [0, 2, 1, None])

    def test_automatic_ordering_and_placement_reproduce_figure_5(self):
        # The tie order and explicit relaxation schedule are the same project
        # choices as the existing Fig. 5 Grouping test, not author defaults.
        variables, seq, graph = paper_example()
        groups = grouping(graph, 8, threshold_schedule=PAPER_THRESHOLDS)
        ordered = order_groups(graph, groups)
        self.assertEqual(ordered, [groups[i] for i in [3, 0, 2, 1]])
        layout = placement(graph, groups, 8)
        self.assertEqual([variables[v] for v in layout], list("cdghfaeb"))
        offsets = {vertex: address % 4 for address, vertex in enumerate(layout)}
        self.assertEqual(sum(abs(offsets[u] - offsets[v])
                             for u, v in zip(seq, seq[1:])), 21)

    def test_ordering_scores_all_placed_groups_not_just_the_front(self):
        graph = make_graph(
            range(4), [(0, 1, 10), (0, 2, 9), (0, 3, 1), (1, 2, 1), (1, 3, 8)]
        )
        # Seed 0, then choose 1. For L=[1,0], vertex 2 scores 1+9=10,
        # beating vertex 3's 8+1=9 despite its smaller edge to front group 1.
        groups = [[0], [1], [2], [3]]
        self.assertEqual(order_groups(graph, groups), [[3], [2], [1], [0]])
        self.assertEqual(placement(graph, groups, 4, num_ports=1), [3, 2, 1, 0])

    def test_initial_group_weight_includes_internal_edges(self):
        graph = make_graph(
            range(6), [(0, 1, 20), (2, 4, 10), (3, 5, 10)]
        )
        groups = [[0, 1], [2, 3], [4, 5]]
        # phi is 40,20,20; seed group 0 must end up last after prepending.
        self.assertEqual(order_groups(graph, groups), [groups[2], groups[1], groups[0]])

    def test_ordering_ties_use_group_order_not_vertex_ids(self):
        graph = {9: {}, 1: {}, 7: {}}
        self.assertEqual(order_groups(graph, [[7], [1], [9]]), [[9], [1], [7]])

    def test_ordering_and_placement_handle_empty_and_single_groups(self):
        self.assertEqual(order_groups({}, []), [])
        self.assertEqual(order_groups({}, [[], []]), [[], []])
        self.assertEqual(placement({}, [[], []], 4), [None] * 4)
        graph = make_graph([3, 7], [(3, 7, 2)])
        self.assertEqual(order_groups(graph, [[7, 3]]), [[7, 3]])
        self.assertEqual(placement(graph, [[7, 3]], 2), [7, 3])

    def test_automatic_placement_retains_empty_groups_and_unused_blocks(self):
        # All affinities are zero, so seed group 0 followed by group 1 then 2.
        graph = {7: {}, 3: {}, 5: {}}
        groups = [[7], [], [3, 5]]
        self.assertEqual(order_groups(graph, groups), [[3, 5], [], [7]])
        self.assertEqual(placement(graph, groups, 6), [3, None, 7, 5, None, None])

    def test_ordering_and_placement_do_not_mutate_inputs_or_alias_groups(self):
        graph = make_graph(range(3), [(0, 1, 2), (1, 2, 1)])
        groups = [[0, 1], [2]]
        original_graph, original_groups = deepcopy(graph), deepcopy(groups)
        ordered = order_groups(graph, groups)
        placement(graph, groups, 4)
        self.assertEqual(graph, original_graph)
        self.assertEqual(groups, original_groups)
        ordered[0].append(99)
        self.assertEqual(groups, original_groups)
        with self.assertRaises(ValueError):
            placement(graph, groups, 2)  # Wrong number of capacity groups.
        self.assertEqual(graph, original_graph)
        self.assertEqual(groups, original_groups)

    def test_automatic_ordering_requires_a_complete_valid_partition(self):
        graph = {0: {}, 1: {}}
        for groups in ([], [[0]], [[0, 1, 2]], [[0, 0], [1]],
                       [[0], [1, 0]], [[True], [0]], [[-1], [0, 1]],
                       [[0.0], [1]], [["0"], [1]]):
            with self.subTest(groups=groups):
                with self.assertRaises(ValueError):
                    order_groups(graph, groups)
                with self.assertRaises(ValueError):
                    placement(graph, groups, 4)

    def test_automatic_placement_checks_port_capacity(self):
        graph = {0: {}, 1: {}, 2: {}}
        with self.assertRaisesRegex(ValueError, "more vertices than num_ports"):
            placement(graph, [[0, 1, 2], []], 4)
        self.assertEqual(placement(graph, [[0, 1, 2]], 3, num_ports=3), [0, 1, 2])


class TestChenIntegration(unittest.TestCase):

    def test_ofu_style_arguments_run_both_phases_with_default_two_ports(self):
        self.assertEqual(chen([[0, 1, 2]], [0, 1, 0, 2, 1, 0], 3),
                         [[2, 0, None, 1]])

    def test_subsequences_bridge_other_dbcs_and_restore_global_ids(self):
        dbcs = [[0, 2, 4, 6], [1, 3]]
        seq = [0, 1, 2, 3, 0, 1, 4, 3, 2, 1, 0, 3, 6]
        self.assertEqual(chen(dbcs, seq, 7, threshold_schedule=[(0, 0)]),
                         [[4, 0, 6, 2], [1, 3]])

    def test_empty_dbcs_and_unassigned_variables(self):
        self.assertEqual(chen([[], [4], []], [4, 1, 4], 5), [[], [4, None], []])
        self.assertEqual(chen([], [], 0), [])
        self.assertEqual(chen([[], []], [], 0), [[], []])

    def test_never_accessed_variables_are_preserved(self):
        result = chen([[4, 0, 2]], [0, 0], 5)
        self.assertEqual(result, [[2, 4, None, 0]])
        self.assertCountEqual([v for v in result[0] if v is not None], [4, 0, 2])

    def test_configurable_ports(self):
        seq = [0, 1, 0, 2, 1, 0]
        self.assertEqual(chen([[0, 1, 2]], seq, 3, num_ports=1), [[2, 1, 0]])
        self.assertEqual(chen([[0, 1, 2]], [], 3, num_ports=3), [[0, 1, 2]])

    def test_threshold_parameters_and_stall_policy_are_forwarded(self):
        # Each K4 edge is traversed twice, so 0.5/0.5 cannot form pairs.
        seq = [0, 1, 2, 0, 3, 1, 0, 2, 3, 2, 1, 3, 0]
        with self.assertRaisesRegex(ValueError, "Grouping stalled"):
            chen([[0, 1, 2, 3]], seq, 4)
        self.assertEqual(chen([[0, 1, 2, 3]], seq, 4, alpha=0, beta=0),
                         [[2, 0, 3, 1]])

    def test_generator_schedule_is_reused_for_all_dbcs(self):
        trace = [0, 1, 2, 0, 3, 1, 0, 2, 3, 2, 1, 3, 0]
        seq = trace + [v + 4 for v in trace]
        schedule = (pair for pair in [(0, 0)])
        self.assertEqual(chen([[0, 1, 2, 3], [4, 5, 6, 7]], seq, 8,
                              threshold_schedule=schedule),
                         [[1, 0, 3, 2], [5, 4, 7, 6]])

    def test_paper_example_via_ofu_style_interface(self):
        variables, seq, _ = paper_example()
        layouts = chen([list(range(8))], seq, len(variables),
                       threshold_schedule=PAPER_THRESHOLDS)
        self.assertEqual([[variables[v] for v in dbc] for dbc in layouts],
                         [list("cdghfaeb")])

    def test_inputs_are_unchanged_and_outputs_do_not_alias(self):
        dbcs, seq = [[0, 1], [2]], [0, 2, 1, 2, 0]
        original_dbcs, original_seq = deepcopy(dbcs), seq[:]
        result = chen(dbcs, seq, 3)
        result[0].append(99)
        self.assertEqual(dbcs, original_dbcs)
        self.assertEqual(seq, original_seq)

    def test_invalid_global_ids_and_duplicate_members(self):
        for dbcs, seq in (([[0], [0]], []), ([[3]], []), ([[-1]], []),
                          ([[True]], []), ([[0]], [3]), ([[0]], [-1]),
                          ([[0]], [True]), ([[0]], [1.5])):
            with self.subTest(dbcs=dbcs, seq=seq):
                with self.assertRaises(ValueError):
                    chen(dbcs, seq, 3)

    def test_invalid_configuration(self):
        for count in (-1, 1.5, True):
            with self.subTest(num_variables=count):
                with self.assertRaises(ValueError):
                    chen([], [], count)
        for ports in (0, -1, 1.5, True):
            with self.subTest(num_ports=ports):
                with self.assertRaises(ValueError):
                    chen([], [], 0, num_ports=ports)
        with self.assertRaises(ValueError):
            chen([], [], 0, alpha=-1)


if __name__ == "__main__":
    unittest.main()
