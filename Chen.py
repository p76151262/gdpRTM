from collections.abc import Iterable
from math import isfinite
from numbers import Real


def build_access_graph(seq: list[int], variables: list[str]):
    """
    Build an undirected weighted access graph.
    Each vertex represents a variable.
    The weight w(u, v) counts consecutive accesses
    between u and v, regardless of direction.
    """
    n = len(variables)

    graph = {i: {} for i in range(n)}

    for var in seq:
        if not isinstance(var, int) or not 0 <= var < n:
            raise ValueError(f"Invalid variable index: {var}")

    for u, v in zip(seq, seq[1:]):
        if u == v:
            continue

        graph[u][v] = graph[u].get(v, 0) + 1
        graph[v][u] = graph[v].get(u, 0) + 1

    return graph


def total_edge_weights(graph):
    
    #Calculate W_i = sum_j w(v_i, v_j).
    return {
        vertex: sum(neighbors.values())
        for vertex, neighbors in graph.items()
    }


def group_weight(group: list[int], vertex_weights: dict[int, int]) -> int:

    #calculate total group weight
    return sum(vertex_weights[vertex] for vertex in group)


def vertex_group_weight(graph, vertex: int, group: list[int]) -> int:
    """
    Return Phi(v_i, g) = sum(w(v_i, v_j) for v_j in g).
    A missing edge in the existing sparse access graph has weight zero.
    """
    return sum(graph[vertex].get(member, 0) for member in group)


def inter_group_weight(graph, left: list[int], right: list[int]) -> int:
    """
    Return Psi(g_x, g_y), the sum of edges between disjoint groups.
    Each undirected edge contributes once, unlike the sum of vertex weights.
    """
    return sum(vertex_group_weight(graph, vertex, right) for vertex in left)


def _validate_thresholds(alpha, beta):
    for name, value in (("alpha", alpha), ("beta", beta)):
        if (isinstance(value, bool) or not isinstance(value, Real)
                or not isfinite(value) or value < 0):
            raise ValueError(f"{name} must be a finite nonnegative number")


def _validate_capacity(num_blocks, num_ports, num_variables=0):

    #validate a DBC's dimensions and return its segment size.
    for name, value in (("num_blocks", num_blocks), ("num_ports", num_ports)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if num_blocks < num_variables:
        raise ValueError("num_blocks must be at least the number of vertices")
    if num_blocks % num_ports:
        raise ValueError("num_blocks must be divisible by num_ports")
    return num_blocks // num_ports


def _validate_placement_capacity(groups, num_blocks, num_ports):

    #Validate the number and size of groups against the DBC capacity.
    segment_size = _validate_capacity(num_blocks, num_ports)
    if len(groups) != segment_size:
        raise ValueError("ordered_groups must contain num_blocks / num_ports groups")
    if any(len(group) > num_ports for group in groups):
        raise ValueError("a group cannot contain more vertices than num_ports")


def _validate_group_members(groups, graph=None):
    
    #Check unique vertex IDs and, when supplied, an exact graph partition.
    seen = set()
    for group in groups:
        for vertex in group:
            if isinstance(vertex, bool) or not isinstance(vertex, int) or vertex < 0:
                raise ValueError("group members must be nonnegative integer vertex IDs")
            if graph is not None and vertex not in graph:
                raise ValueError(f"unknown vertex {vertex} in groups")
            if vertex in seen:
                raise ValueError(f"vertex {vertex} occurs in more than one group position")
            seen.add(vertex)
    if graph is not None and seen != set(graph):
        raise ValueError("groups must contain every graph vertex exactly once")


def _qualifies(affinity, group_total, vertex_weight, alpha, beta):
    # Both comparisons are inclusive, exactly as in (14) and (15).
    # W_i stays fixed; phi(g) and Phi(v_i, g) change as the group grows.
    return (affinity >= alpha * group_total
            and affinity >= beta * vertex_weight)


def grouping(
    graph: dict[int, dict[int, int]],
    num_blocks: int,
    num_ports: int = 2,
    alpha: float = 0.5,
    beta: float = 0.5,
    *,
    threshold_schedule: Iterable[tuple[float, float]] | None = None,
) -> list[list[int]]:

    segment_size = _validate_capacity(num_blocks, num_ports, len(graph))
    _validate_thresholds(alpha, beta)

    # Validate the finite, explicit schedule before doing any grouping work.
    schedule = []
    previous_alpha, previous_beta = alpha, beta
    if threshold_schedule is not None:
        for next_alpha, next_beta in threshold_schedule:
            _validate_thresholds(next_alpha, next_beta)
            if next_alpha > previous_alpha or next_beta > previous_beta:
                raise ValueError("threshold_schedule must not increase alpha or beta")
            schedule.append((next_alpha, next_beta))
            previous_alpha, previous_beta = next_alpha, next_beta
    thresholds = iter(schedule)

    # Algorithm 1, lines 1-2: original W_i and stable descending queue Q.
    weights = total_edge_weights(graph)
    queue = sorted(graph, key=lambda vertex: weights[vertex], reverse=True)
    groups = [[] for _ in range(segment_size)]

    # Lines 3-8: seed each group, then scan Q in order, not by maximum Phi.
    for group in groups:
        if not queue:
            break
        group.append(queue.pop(0))
        group_total = weights[group[0]]
        # Snapshot Q so removing accepted vertices cannot skip a candidate.
        for vertex in queue[:]:
            if len(group) == num_ports:
                break
            affinity = vertex_group_weight(graph, vertex, group)
            if _qualifies(affinity, group_total, weights[vertex], alpha, beta):
                group.append(vertex)
                queue.remove(vertex)
                group_total += weights[vertex]

    # Lines 9-16: refill non-full groups using maximum qualifying affinity.
    while queue:
        progressed = False
        for group in groups:
            if len(group) == num_ports or not queue:
                continue
            group_total = group_weight(group, weights)
            best_vertex = None
            best_affinity = -1
            for vertex in queue:
                affinity = vertex_group_weight(graph, vertex, group)
                if (_qualifies(affinity, group_total, weights[vertex], alpha, beta)
                        and affinity > best_affinity):
                    # Strict > preserves the first qualifying candidate on ties.
                    best_vertex, best_affinity = vertex, affinity
            if best_vertex is not None:
                group.append(best_vertex)
                queue.remove(best_vertex)
                progressed = True
        if not queue:
            break
        next_pair = next(thresholds, None)
        if next_pair is not None:
            alpha, beta = next_pair
        elif not progressed:
            raise ValueError(
                f"Grouping stalled with unassigned vertices {queue} at "
                f"alpha={alpha}, beta={beta}; provide an explicit "
                "threshold_schedule with further relaxation"
            )
    return groups


def place_groups(
    ordered_groups: list[list[int]],
    num_blocks: int,
    num_ports: int = 2,
) -> list[int | None]:
    """
    Algorithm 2's automatic ordering is not performed by this function.
    Capacity n must be divisible by m ports; provide exactly t=n/m groups.
    """
    _validate_placement_capacity(ordered_groups, num_blocks, num_ports)
    _validate_group_members(ordered_groups)
    return _map_groups(ordered_groups, num_blocks)


def _map_groups(ordered_groups, num_blocks):
    
    #Map validated groups to physical addresses without repeating validation.
    segment_size = len(ordered_groups)
    layout = [None] * num_blocks
    for offset, group in enumerate(ordered_groups):
        for port, vertex in enumerate(group):
            # Paper offsets are 1-based; Python layout addresses are 0-based.
            # Retain empty blocks so neither offsets nor port boundaries move.
            layout[offset + port * segment_size] = vertex
    return layout


def order_groups(graph, groups: list[list[int]]) -> list[list[int]]:
    """Order groups using Algorithm 2 with the approved project selection rule.
    Start with maximal phi(g), then maximize sum(Psi(g, h) for h in L).
    Prepend each selected group to L, preserving member order within groups.

    This approved rule replaces the paper's unspecified 'minimum total cost';
    it is not confirmed as the authors' rule or a minimization of Eq. (3).
    Ties keep the input group order; keep empty groups as unused capacity.
    Require an exact partition of graph vertices; return independent copies.
    """
    _validate_group_members(groups, graph)
    return _order_groups(graph, groups)


def _order_groups(graph, groups):
    
    if not groups:
        return []

    # Algorithm 2, lines 1-3: use the same phi(g) as the grouping phase.
    weights = total_edge_weights(graph)
    group_weights = [group_weight(group, weights) for group in groups]
    remaining = list(range(len(groups)))
    first = max(remaining, key=lambda index: group_weights[index])
    order = [first]
    remaining.remove(first)
    """
    Approved project interpretation of line 5: greatest total affinity to L.
    Scores include ALL groups in L, not only the most recently placed one.
    Updating with each new group's Psi avoids repeatedly summing the full L.
    """
    scores = {index: inter_group_weight(graph, groups[index], groups[first])
              for index in remaining}
    while remaining:
        selected = max(remaining, key=lambda index: scores[index])
        order.insert(0, selected)  # Line 6: L <- g_p, L (prepend, not append).
        remaining.remove(selected)
        for index in remaining:
            scores[index] += inter_group_weight(graph, groups[index], groups[selected])
    return [list(groups[index]) for index in order]


def placement(
    graph: dict[int, dict[int, int]],
    groups: list[list[int]],
    num_blocks: int,
    num_ports: int = 2,
) -> list[int | None]:
    
    #Order Algorithm 1's groups, then map them to a single DBC's blocks.
    
    _validate_group_members(groups, graph)
    _validate_placement_capacity(groups, num_blocks, num_ports)
    return _map_groups(_order_groups(graph, groups), num_blocks)


def chen(
    dbcs: list[list[int]],
    seq: list[int],
    num_variables: int,
    *,
    num_ports: int = 2,
    alpha: float = 0.5,
    beta: float = 0.5,
    threshold_schedule: Iterable[tuple[float, float]] | None = None,
) -> list[list[int | None]]:
    """Reorder each DBC with S-DBC-P; the first three arguments match ofu().
    Filter seq per DBC, then build its access graph, group and place variables.
    Preserve DBC membership, global IDs and inputs; empty DBCs return [].
    Round each nonempty DBC's capacity up to a multiple of num_ports; retain
    unused physical slots as None instead of changing segment boundaries.

    Keyword options follow grouping(), including its explicit relaxation policy.
    Ordering uses the approved project rule in order_groups(). The result has
    one physical layout per DBC; common.shift_cost() supports only one port.
    """
    if (isinstance(num_variables, bool) or not isinstance(num_variables, int)
            or num_variables < 0):
        raise ValueError("num_variables must be a nonnegative integer")
    if isinstance(num_ports, bool) or not isinstance(num_ports, int) or num_ports <= 0:
        raise ValueError("num_ports must be a positive integer")
    _validate_thresholds(alpha, beta)
    _validate_group_members(dbcs)

    # Local IDs follow each input DBC's order, preserving deterministic ties.
    locations = {}
    for dbc_index, dbc in enumerate(dbcs):
        for local_index, vertex in enumerate(dbc):
            if vertex >= num_variables:
                raise ValueError(f"Invalid variable index: {vertex}")
            locations[vertex] = (dbc_index, local_index)

    subsequences = [[] for _ in dbcs]
    for vertex in seq:
        if (isinstance(vertex, bool) or not isinstance(vertex, int)
                or not 0 <= vertex < num_variables):
            raise ValueError(f"Invalid variable index: {vertex}")
        if vertex in locations:
            dbc_index, local_index = locations[vertex]
            subsequences[dbc_index].append(local_index)

    # Materialize a caller-supplied generator once so every DBC gets its schedule.
    schedule = None if threshold_schedule is None else tuple(threshold_schedule)
    layouts = []
    for dbc, subsequence in zip(dbcs, subsequences):
        if not dbc:
            layouts.append([])
            continue
        num_blocks = ((len(dbc) + num_ports - 1) // num_ports) * num_ports
        graph = build_access_graph(subsequence, [str(vertex) for vertex in dbc])
        groups = grouping(graph, num_blocks, num_ports, alpha, beta,
                          threshold_schedule=schedule)
        local_layout = placement(graph, groups, num_blocks, num_ports)
        layouts.append([None if vertex is None else dbc[vertex]
                        for vertex in local_layout])
    return layouts
