from typing import Dict, Tuple, List
import numpy as np

try:
    import networkx as nx
except Exception:  # optional dependency
    nx = None  # type: ignore


def create_track_graph(node_positions: Dict[str, Tuple[float, float]],
                       edges: List[Tuple[str, str]]):
    """
    Create a track graph from node positions and edge list.
    """
    if nx is None:
        raise ImportError("networkx is required for create_track_graph")
    G = nx.Graph()
    for nid, (x, y) in node_positions.items():
        G.add_node(nid, pos=(float(x), float(y)))
    G.add_edges_from(edges)
    return G


def _project_point_to_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray):
    """Project point p onto segment a-b. Returns (t, distance, projection)."""
    ap = p - a
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom == 0.0:
        t = 0.0
        proj = a
    else:
        t = float(np.clip(np.dot(ap, ab) / denom, 0.0, 1.0))
        proj = a + t * ab
    dist = float(np.linalg.norm(p - proj))
    return t, dist, proj


def linearize_positions_along_graph(G, xy: np.ndarray):
    """
    Linearize 2D positions to 1D distances along the closest edge.
    Returns (linear_pos, edge_indices).
    """
    if nx is None:
        raise ImportError("networkx is required for linearize_positions_along_graph")
    edges = list(G.edges)
    node_pos = nx.get_node_attributes(G, "pos")
    linear = np.zeros(xy.shape[0], dtype=float)
    edge_ids = np.zeros(xy.shape[0], dtype=int)

    # precompute edge lengths and cumulative mapping
    edge_lengths = []
    for (u, v) in edges:
        a = np.array(node_pos[u], dtype=float)
        b = np.array(node_pos[v], dtype=float)
        edge_lengths.append(np.linalg.norm(b - a))
    cum_lengths = np.cumsum([0.0] + edge_lengths[:-1])

    for i, p in enumerate(xy):
        best = (1e18, 0, 0.0)  # (dist, edge_idx, t)
        for ei, (u, v) in enumerate(edges):
            a = np.array(node_pos[u], dtype=float)
            b = np.array(node_pos[v], dtype=float)
            t, dist, _ = _project_point_to_segment(p, a, b)
            if dist < best[0]:
                best = (dist, ei, t)
        dist_along = cum_lengths[best[1]] + best[2] * edge_lengths[best[1]]
        linear[i] = dist_along
        edge_ids[i] = best[1]
    return linear, edge_ids


def classify_track_segments(G, xy: np.ndarray, diagonal_bias: float = 0.0):
    """
    Simple nearest-edge classifier: returns edge index for each sample.
    diagonal_bias can be used to bias tie-breaks.
    """
    if nx is None:
        raise ImportError("networkx is required for classify_track_segments")
    edges = list(G.edges)
    node_pos = nx.get_node_attributes(G, "pos")

    edge_ids = np.zeros(xy.shape[0], dtype=int)
    for i, p in enumerate(xy):
        best = (1e18, 0)
        for ei, (u, v) in enumerate(edges):
            a = np.array(node_pos[u], dtype=float)
            b = np.array(node_pos[v], dtype=float)
            _, dist, _ = _project_point_to_segment(p, a, b)
            score = dist + diagonal_bias
            if score < best[0]:
                best = (score, ei)
        edge_ids[i] = best[1]
    return edge_ids
