"""DAP: Distortion-Aware Partitioner.

Report-aligned implementation of the DAP method used in the thesis.

DAP partitions a graph into exactly K federated clients while trying to reduce
structural loss caused by cross-client edges and keep client sizes balanced.
It starts from a Louvain partition and refines only boundary nodes.

Objective
---------
For a partition P, DAP minimizes

    J(P) = sum_v d_cut(v) + lambda * imbalance(P)

where d_cut(v) is the number of edges incident to node v that cross client
boundaries. Because every undirected cut edge is incident to two nodes, the
first term counts each cross-client edge twice. This is equivalent to the
report expression degree(v) * cut_ratio(v), summed over all nodes.

Algorithm used in the report
----------------------------
1. Convert the input to the undirected structural view used for partitioning.
2. Initialize with Louvain at resolution 0.8.
3. Pack Louvain communities into exactly K clients.
4. Identify boundary nodes (nodes with at least one neighbor in another client).
5. Visit boundary nodes in deterministic high-degree-first order.
6. For each boundary node, test moves only to neighboring clients.
7. Accept a move only if J strictly decreases.
8. Repeat until no improving move remains or max_sweeps is reached.

The default balance penalty is normalized L1 deviation from the ideal client
size. The thesis writes the balance term abstractly as imbalance(P); keeping
the implementation explicit here makes the experimental setting reproducible.

This module contains no dataset-specific loading logic. It expects an already
built NetworkX graph and returns exactly K node sets.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import networkx as nx
import numpy as np


SEED = 42
LOUVAIN_RESOLUTION = 0.8
DEFAULT_LAMBDA = 1.0
DEFAULT_MAX_SWEEPS = 100
DEFAULT_MIN_CLIENT_SIZE = 1


def _structural_graph(graph: nx.Graph) -> nx.Graph:
    """Return the undirected structural graph used by DAP.

    If the input is directed, reciprocal edges are merged and their weights are
    summed. Self-loops are ignored because they do not affect cross-client
    structural loss.
    """
    if not graph.is_directed() and not graph.is_multigraph():
        undirected = graph.copy()
        undirected.remove_edges_from(nx.selfloop_edges(undirected))
        return undirected

    undirected = nx.Graph()
    undirected.add_nodes_from(graph.nodes(data=True))

    for u, v, data in graph.edges(data=True):
        if u == v:
            continue
        weight = float(data.get("weight", 1.0))
        if undirected.has_edge(u, v):
            undirected[u][v]["weight"] += weight
        else:
            undirected.add_edge(u, v, weight=weight)

    return undirected


def _split_until_k(communities: Iterable[Set], k: int, seed: int) -> List[Set]:
    """Ensure there are at least k non-empty groups before client packing."""
    communities = [set(c) for c in communities if c]
    rng = np.random.default_rng(seed)

    while len(communities) < k:
        communities.sort(key=len, reverse=True)
        largest = communities.pop(0)
        if len(largest) < 2:
            raise ValueError(
                "Cannot create exactly k non-empty clients: there are fewer "
                "nodes than requested clients."
            )

        nodes = list(largest)
        rng.shuffle(nodes)
        mid = len(nodes) // 2
        communities.append(set(nodes[:mid]))
        communities.append(set(nodes[mid:]))

    return communities


def communities_to_clients(
    communities: Iterable[Set],
    k: int,
    *,
    seed: int = SEED,
) -> List[Set]:
    """Greedily pack communities into exactly k non-empty clients.

    Larger communities are assigned first to the currently smallest client.
    This is the balanced community-packing step used before DAP refinement.
    """
    if k < 1:
        raise ValueError("k must be at least 1")

    communities = _split_until_k(communities, k, seed)
    clients = [set() for _ in range(k)]
    sizes = [0] * k

    for community in sorted(communities, key=len, reverse=True):
        client = sizes.index(min(sizes))
        clients[client].update(community)
        sizes[client] += len(community)

    if any(len(client) == 0 for client in clients):
        raise RuntimeError("Internal error: client packing produced an empty client")

    return clients


def initial_partition_louvain(
    graph: nx.Graph,
    k: int,
    *,
    seed: int = SEED,
    resolution: float = LOUVAIN_RESOLUTION,
) -> List[Set]:
    """Build the report-aligned Louvain starting partition.

    Louvain uses resolution 0.8 by default and detected communities are packed
    into exactly k clients.
    """
    undirected = _structural_graph(graph)
    n = undirected.number_of_nodes()

    if n == 0:
        raise ValueError("graph must contain at least one node")
    if k > n:
        raise ValueError(f"k={k} cannot exceed the number of nodes ({n})")

    communities = nx.community.louvain_communities(
        undirected,
        seed=seed,
        resolution=resolution,
        weight="weight",
    )
    return communities_to_clients(communities, k, seed=seed)


def _parts_to_assignment(parts: Sequence[Set]) -> Dict[object, int]:
    assignment: Dict[object, int] = {}
    for client, nodes in enumerate(parts):
        for node in nodes:
            if node in assignment:
                raise ValueError(f"Node {node!r} appears in more than one client")
            assignment[node] = client
    return assignment


def _assignment_to_parts(
    assignment: Mapping[object, int],
    k: int,
) -> List[Set]:
    parts = [set() for _ in range(k)]
    for node, client in assignment.items():
        if client < 0 or client >= k:
            raise ValueError(f"Invalid client id {client} for node {node!r}")
        parts[int(client)].add(node)
    return parts


def _client_sizes(assignment: Mapping[object, int], k: int) -> np.ndarray:
    sizes = np.zeros(k, dtype=int)
    for client in assignment.values():
        sizes[int(client)] += 1
    return sizes


def _imbalance_cost(
    sizes: Sequence[int],
    total_nodes: int,
    k: int,
    mode: str = "l1",
) -> float:
    """Client-size imbalance penalty.

    l1 (default): normalized absolute deviation from the ideal client size.
    variance: mean squared normalized deviation from the ideal client size.
    """
    target = total_nodes / k
    sizes_arr = np.asarray(sizes, dtype=float)

    if mode == "l1":
        return float(np.abs(sizes_arr - target).sum() / target)
    if mode == "variance":
        return float(np.mean(((sizes_arr - target) / target) ** 2))

    raise ValueError("imbalance_mode must be 'l1' or 'variance'")


def _structural_loss(
    graph: nx.Graph,
    assignment: Mapping[object, int],
) -> float:
    """Return sum_v d_cut(v).

    Every undirected cross-client edge contributes 2 because it is incident to
    two nodes. This is equivalent to sum_v degree(v) * cut_ratio(v).
    """
    return float(
        2
        * sum(
            1
            for u, v in graph.edges()
            if assignment[u] != assignment[v]
        )
    )


def _objective_from_assignment(
    graph: nx.Graph,
    assignment: Mapping[object, int],
    k: int,
    lam: float,
    imbalance_mode: str,
) -> float:
    sizes = _client_sizes(assignment, k)
    return _structural_loss(graph, assignment) + lam * _imbalance_cost(
        sizes, graph.number_of_nodes(), k, imbalance_mode
    )


def objective_value(
    graph: nx.Graph,
    parts: Sequence[Set],
    lam: float = DEFAULT_LAMBDA,
    imbalance_mode: str = "l1",
) -> float:
    """Compute the report-aligned DAP objective for an existing partition."""
    undirected = _structural_graph(graph)
    k = len(parts)

    if k == 0:
        raise ValueError("parts must contain at least one client")

    assignment = _parts_to_assignment(parts)
    graph_nodes = set(undirected.nodes())

    if set(assignment) != graph_nodes:
        missing = graph_nodes - set(assignment)
        extra = set(assignment) - graph_nodes
        raise ValueError(
            "Partition must cover every graph node exactly once. "
            f"Missing={len(missing)}, extra={len(extra)}"
        )

    return _objective_from_assignment(
        undirected, assignment, k, lam, imbalance_mode
    )


def dap_partition(
    graph: nx.Graph,
    k: int,
    lam: float = DEFAULT_LAMBDA,
    *,
    resolution: float = LOUVAIN_RESOLUTION,
    imbalance_mode: str = "l1",
    max_sweeps: int = DEFAULT_MAX_SWEEPS,
    min_client_size: int = DEFAULT_MIN_CLIENT_SIZE,
    seed: int = SEED,
) -> List[Set]:
    """Run the report-aligned Distortion-Aware Partitioner.

    Parameters
    ----------
    graph:
        NetworkX Graph/DiGraph. DAP operates on its undirected structural view.
    k:
        Number of federated clients. Exactly k non-empty node sets are returned.
    lam:
        Weight of the client-size balance penalty.
    resolution:
        Louvain resolution. The report-aligned default is 0.8.
    imbalance_mode:
        'l1' (default) or 'variance'. The thesis writes this term abstractly as
        imbalance(P); the repository experiment uses normalized L1 by default.
    max_sweeps:
        Maximum number of complete boundary-node refinement sweeps.
    min_client_size:
        A move is rejected if it would shrink a client below this size.
    seed:
        Seed used by Louvain and by the rare community-splitting fallback.

    Returns
    -------
    list[set]
        Exactly k non-empty client node sets.

    Notes
    -----
    Boundary nodes are processed in deterministic high-degree-first order.
    Candidate destinations are restricted to neighboring clients. A move is
    accepted only when the objective strictly decreases, so the final DAP
    objective cannot be worse than the Louvain initialization under the same
    objective and settings. The method is still a local search and therefore
    does not guarantee the global optimum.
    """
    if lam < 0:
        raise ValueError("lam must be non-negative")
    if max_sweeps < 1:
        raise ValueError("max_sweeps must be at least 1")
    if min_client_size < 1:
        raise ValueError("min_client_size must be at least 1")

    undirected = _structural_graph(graph)
    initial_parts = initial_partition_louvain(
        undirected,
        k,
        seed=seed,
        resolution=resolution,
    )
    assignment = _parts_to_assignment(initial_parts)

    current = _objective_from_assignment(
        undirected, assignment, k, lam, imbalance_mode
    )

    for _ in range(max_sweeps):
        moved = 0

        boundary = [
            node
            for node in undirected.nodes()
            if any(assignment[nbr] != assignment[node]
                   for nbr in undirected.neighbors(node))
        ]
        boundary.sort(key=lambda node: (-undirected.degree(node), str(node)))

        for node in boundary:
            old_client = assignment[node]
            sizes = _client_sizes(assignment, k)
            if sizes[old_client] <= min_client_size:
                continue

            candidate_clients = sorted({
                assignment[nbr]
                for nbr in undirected.neighbors(node)
                if assignment[nbr] != old_client
            })

            best_client = old_client
            best_objective = current

            for new_client in candidate_clients:
                assignment[node] = new_client
                candidate_objective = _objective_from_assignment(
                    undirected, assignment, k, lam, imbalance_mode
                )
                assignment[node] = old_client

                if candidate_objective < best_objective - 1e-12:
                    best_objective = candidate_objective
                    best_client = new_client

            if best_client != old_client:
                assignment[node] = best_client
                current = best_objective
                moved += 1

        if moved == 0:
            break

    result = _assignment_to_parts(assignment, k)

    if len(result) != k or any(len(part) < min_client_size for part in result):
        raise RuntimeError("DAP produced an invalid client partition")

    return result


if __name__ == "__main__":
    # Small smoke test. Real experiments should build the graph using the same
    # preprocessing and graph-construction pipeline described in the report.
    example_graph = nx.gnm_random_graph(200, 800, seed=SEED)
    k = 4

    louvain_parts = initial_partition_louvain(example_graph, k)
    dap_parts = dap_partition(example_graph, k)

    print("Louvain J:", objective_value(example_graph, louvain_parts))
    print("DAP J:", objective_value(example_graph, dap_parts))
    print("DAP client sizes:", [len(part) for part in dap_parts])
