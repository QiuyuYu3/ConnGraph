"""
Graph layouts that keep each group of nodes (e.g. a network) in its own region.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

_GOLDEN_ANGLE = np.pi * (3 - np.sqrt(5))


def grouped_layout(
    G: nx.Graph,
    groups,
    dim: int = 2,
    seed: int = 42,
    margin: float = 1.0,
) -> dict:
    """Give each group (labels in G.nodes() order) its own region, placed closer to groups it shares more |weight| with."""
    nodes = list(G.nodes())
    groups = list(groups)
    if len(groups) != len(nodes):
        raise ValueError(f"groups has {len(groups)} entries; the graph has {len(nodes)} nodes.")

    group_of = dict(zip(nodes, groups))
    names = list(dict.fromkeys(groups))
    radius = np.array([np.sqrt(groups.count(g)) for g in names])
    centres = _region_centres(G, group_of, names, radius, dim, seed, margin)

    rng = np.random.default_rng(seed)
    pos = {}
    for name, centre, r in zip(names, centres, radius):
        members = [nd for nd in nodes if group_of[nd] == name]
        local = _local_layout(G, members, dim, seed, rng)
        for nd, xyz in zip(members, centre + local * r * 0.9):
            pos[nd] = xyz
    return pos


def _region_centres(G, group_of, names, radius, dim, seed, margin) -> np.ndarray:
    H = nx.Graph()
    H.add_nodes_from(names)
    for u, v, w in G.edges(data="weight", default=1.0):
        a, b = group_of[u], group_of[v]
        if a != b:
            H.add_edge(a, b, weight=H.get_edge_data(a, b, {"weight": 0.0})["weight"] + abs(w))
    start = nx.spring_layout(H, dim=dim, seed=seed, weight="weight")
    centres = np.array([start[g] for g in names], dtype=float) * np.sqrt((radius ** 2).sum())

    rng = np.random.default_rng(seed)
    for _ in range(1000):
        moved = False
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                d = centres[j] - centres[i]
                dist = np.linalg.norm(d)
                short = radius[i] + radius[j] + margin - dist
                if short > 1e-9:
                    unit = d / dist if dist > 1e-9 else rng.normal(size=dim)
                    unit = unit / np.linalg.norm(unit)
                    centres[i] -= unit * short / 2
                    centres[j] += unit * short / 2
                    moved = True
        if not moved:
            break
    return centres


def _local_layout(G, members, dim, seed, rng) -> np.ndarray:
    m = len(members)
    if m == 1:
        return np.zeros((1, dim))

    sub = nx.Graph()
    sub.add_nodes_from(members)
    sub.add_weighted_edges_from((u, v, abs(w)) for u, v, w in G.subgraph(members).edges(data="weight", default=1.0))
    if sub.number_of_edges() == 0:
        return _fill_evenly(m, dim, rng)

    p = nx.spring_layout(sub, dim=dim, seed=seed, weight="weight")
    local = np.array([p[nd] for nd in members], dtype=float)
    local -= local.mean(axis=0)
    peak = np.linalg.norm(local, axis=1).max()
    return local / peak if peak > 0 else local


def _fill_evenly(m: int, dim: int, rng) -> np.ndarray:
    if dim == 2:
        k = np.arange(m)
        r = np.sqrt((k + 0.5) / m)
        return np.c_[r * np.cos(k * _GOLDEN_ANGLE), r * np.sin(k * _GOLDEN_ANGLE)]
    direction = rng.normal(size=(m, dim))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    return direction * rng.random((m, 1)) ** (1 / dim)
