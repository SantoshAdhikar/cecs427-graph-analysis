# graph_analysis.py  — Step 1: load & save GML, print basic info
import argparse
import random
import statistics
from math import sqrt
import sys
import os
import csv
from typing import List, Tuple

import networkx as nx # type: ignore
import matplotlib.pyplot as plt # type: ignore
from networkx.algorithms.community import girvan_newman # type: ignore

# SciPy for p-values in homophily test
try:
    from scipy.stats import norm  # type: ignore
except Exception:
    norm = None

def read_gml(path: str) -> nx.Graph:
    """Read a .gml file, ensure undirected simple graph with string node labels."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input not found: {path}")
    G = nx.read_gml(path)
    G = nx.Graph(G)  # ensure simple undirected
    # normalize labels to strings
    if any(not isinstance(n, str) for n in G.nodes):
        G = nx.relabel_nodes(G, {n: str(n) for n in G.nodes})
    return G

def _safe(v):
    """GML cannot store None; coerce to safe types."""
    if v is None:
        return ""
    if isinstance(v, (str, int, float, bool)):
        return v
    return str(v)

def verify_balanced_graph(G: nx.Graph, sign_attr: str = "sign"):
    """
    BFS-style balance check:
    within-group edges must be '+', between-group edges must be '-'.
    Prints result and returns bool.
    """
    color = {}
    from collections import deque

    for s in G.nodes():
        if s in color:
            continue
        color[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for v in G.neighbors(u):
                sign = G[u][v].get(sign_attr, '+')  # default '+'
                expected = color[u] if sign == '+' else 1 - color[u]
                if v not in color:
                    color[v] = expected
                    q.append(v)
                else:
                    if color[v] != expected:
                        print("[balance] structurally_balanced=False")
                        return False
    print("[balance] structurally_balanced=True")
    return True


def robustness_check(G: nx.Graph, k: int, trials: int = 10):
    """
    Perform multiple random edge removals (k edges per trial).
    Reports:
      - average number of components
      - min/max component sizes
    """
    if k <= 0 or k > G.number_of_edges():
        print(f"[warn] invalid k={k} for robustness_check")
        return

    num_comps_list = []
    min_sizes, max_sizes = [], []

    for t in range(trials):
        H = G.copy()
        edges = list(H.edges())
        removed = random.sample(edges, k)
        H.remove_edges_from(removed)

        comps = list(nx.connected_components(H))
        sizes = [len(c) for c in comps]

        num_comps_list.append(len(comps))
        min_sizes.append(min(sizes))
        max_sizes.append(max(sizes))

    avg_comps = statistics.mean(num_comps_list)
    print(f"[robustness] trials={trials}, k={k}")
    print(f"[robustness] avg_num_components={avg_comps:.2f}")
    print(f"[robustness] min_component_size_range=({min(min_sizes)}, {max(min_sizes)})")
    print(f"[robustness] max_component_size_range=({min(max_sizes)}, {max(max_sizes)})")

def apply_temporal_csv(G: nx.Graph, csv_path: str) -> List[Tuple[str, nx.Graph]]:
    """
    CSV columns: source,target,timestamp,action
    action in {'add','remove'}.
    Returns a list of (timestamp, snapshot_graph).
    """
    steps: List[Tuple[str, nx.Graph]] = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            u = str(row.get("source", "")).strip()
            v = str(row.get("target", "")).strip()
            ts = str(row.get("timestamp", "")).strip()
            action = str(row.get("action", "add")).strip().lower()

            if not u or not v:
                continue

            # auto-add nodes if missing
            if u not in G:
                G.add_node(u)
            if v not in G:
                G.add_node(v)

            if action == "add":
                G.add_edge(u, v)
            elif action == "remove":
                if G.has_edge(u, v):
                    G.remove_edge(u, v)

            steps.append((ts or f"t{len(steps)+1}", G.copy()))
    return steps


def render_temporal_frames(steps, out_dir="plots"):
    os.makedirs(out_dir, exist_ok=True)
    for i, (ts, Gi) in enumerate(steps):
        out_path = os.path.join(out_dir, f"temporal_{i:03d}.png")
        try:
            plot_attributes  # if defined
            plot_attributes(Gi, out_path=out_path, quiet=True)
        except NameError:
            pos = nx.spring_layout(Gi, seed=42)
            plt.figure(figsize=(8,6))
            nx.draw(Gi, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=600)
            plt.title(f"Temporal snapshot @ {ts}")
            plt.axis("off"); plt.tight_layout()
            plt.savefig(out_path, dpi=300); plt.close()
        print(f"[temporal] saved -> {out_path}")
    print(f"[temporal] wrote {len(steps)} frames to '{out_dir}'")


def verify_homophily(G: nx.Graph, attr: str = "color", alternative: str = "greater"):
    """
    Z-test on edge homophily for node attribute `attr`.
    alternative: 'greater' (homophily), 'less' (heterophily), or 'two-sided'
    """
    colors_attr = nx.get_node_attributes(G, attr)
    if not colors_attr:
        print(f"[homophily] no '{attr}' attribute on nodes")
        return

    edges = list(G.edges())
    if not edges:
        print("[homophily] no edges")
        return

    # observed fraction of same-attribute edges
    same = sum(1 for u, v in edges if G.nodes[u].get(attr) == G.nodes[v].get(attr))
    obs = same / len(edges)

    # expected under random mixing based on node proportions
    values = [colors_attr.get(n) for n in G.nodes()]
    total = len(values)
    from collections import Counter
    freq = Counter(values)
    p0 = sum((cnt/total) ** 2 for cnt in freq.values())  # sum p_i^2

    # variance of Bernoulli proportion under H0
    m = len(edges)
    var = p0 * (1 - p0) / m
    if var == 0:
        print(f"[homophily] expected variance is zero (p0={p0:.3f}, m={m}); test not defined")
        return

    z = (obs - p0) / sqrt(var)

    # p-value (if SciPy available)
    if norm is None:
        p = None
        pnote = " (install scipy for p-value)"
    else:
        if alternative == "greater":
            p = 1 - norm.cdf(z)
        elif alternative == "less":
            p = norm.cdf(z)
        else:  # two-sided
            p = 2 * min(norm.cdf(z), 1 - norm.cdf(z))
        pnote = ""

    print(f"[homophily] edges={m} observed_same={obs:.3f} expected_same={p0:.3f}")
    print(f"[homophily] z={z:.3f}" + (f" p={p:.4f}" if p is not None else pnote))

    if alternative == "greater":
        verdict = "evidence of homophily" if z > 0 and (p is None or p < 0.05) else "no evidence of homophily"
    elif alternative == "less":
        verdict = "evidence of heterophily" if z < 0 and (p is None or p < 0.05) else "no evidence of heterophily"
    else:
        verdict = "different from random mixing" if (p is not None and p < 0.05) else "no significant difference"
    print(f"[homophily] {verdict}")

def verify_balanced_graph(G: nx.Graph, sign_attr: str = "sign"):
    """
    BFS-style check: within-group edges must be '+', between-group edges must be '-'.
    Returns True if structurally balanced, else False.
    """
    color = {}
    from collections import deque

    for s in G.nodes():
        if s in color: 
            continue
        color[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for v in G.neighbors(u):
                sign = G[u][v].get(sign_attr, '+')  # default plus
                expected = color[u] if sign == '+' else 1 - color[u]
                if v not in color:
                    color[v] = expected
                    q.append(v)
                else:
                    if color[v] != expected:
                        print("[balance] structurally_balanced=False")
                        return False
    print("[balance] structurally_balanced=True")
    return True


def plot_clustering(G, out_path="plots/clustering.png"):
    """Visualize graph with node size = clustering coefficient, color = degree."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # compute clustering coefficients
    clustering = nx.clustering(G)
    nx.set_node_attributes(G, clustering, "clustering")

    # compute node degrees
    degrees = dict(G.degree())
    nx.set_node_attributes(G, degrees, "degree")

    # prepare layout
    pos = nx.spring_layout(G, seed=42)

    # scale node size by clustering coefficient
    node_sizes = [300 + 2000 * clustering[n] for n in G.nodes()]
    node_colors = [degrees[n] for n in G.nodes()]

    nx.draw(
        G, pos,
        with_labels=True,
        node_color=node_colors,
        cmap=plt.cm.viridis,
        node_size=node_sizes,
        edge_color="gray",
        font_size=10
    )
    plt.title("Clustering Coefficient Visualization")
    plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.viridis),
                 ax=plt.gca(), label="Node degree")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[plot] saved -> {out_path}")


def write_gml(G: nx.Graph, path: str) -> None:
    """Write graph to .gml with sanitized attributes."""
    H = nx.Graph()
    for n, data in G.nodes(data=True):
        H.add_node(str(n), **{k: _safe(v) for k, v in data.items()})
    for u, v, data in G.edges(data=True):
        H.add_edge(str(u), str(v), **{k: _safe(vv) for k, vv in data.items()})
    H.graph.update({k: _safe(v) for k, v in G.graph.items()})
    nx.write_gml(H, path)



def simulate_failures(G: nx.Graph, k: int):
    """
    Remove k random edges from the graph, then analyze:
    - average shortest path length (if connected)
    - number of connected components
    - impact on betweenness centrality (average score)
    """
    if k <= 0 or k > G.number_of_edges():
        print(f"[warn] invalid k={k} for simulate_failures")
        return

    # copy graph so we don't modify the original
    H = G.copy()
    edges = list(H.edges())
    removed = random.sample(edges, k)
    H.remove_edges_from(removed)

    # components
    comps = list(nx.connected_components(H))
    num_comps = len(comps)
    comp_sizes = [len(c) for c in comps]

    # avg shortest path length (only if connected)
    try:
        if nx.is_connected(H):
            aspl = nx.average_shortest_path_length(H)
        else:
            aspl = None
    except Exception:
        aspl = None

    # betweenness centrality (average value across nodes)
    bc = nx.betweenness_centrality(H)
    avg_bc = statistics.mean(bc.values()) if bc else 0.0

    print(f"[failures] removed={removed}")
    print(f"[failures] components={num_comps} sizes={comp_sizes}")
    if aspl is not None:
        print(f"[failures] avg_shortest_path_length={aspl:.3f}")
    else:
        print(f"[failures] avg_shortest_path_length=disconnected")
    print(f"[failures] avg_betweenness={avg_bc:.4f}")


def girvan_newman_partition(G, n: int):
    """
    Partition graph into n communities using Girvan-Newman.
    Returns list of sets of nodes.
    """
    if n < 1:
        raise ValueError("Number of components must be >= 1")

    comp_gen = girvan_newman(G)
    limited = None
    for i in range(n - 1):  # run until we have n groups
        try:
            limited = next(comp_gen)
        except StopIteration:
            break

    if limited is None:
        return [set(G.nodes())]

    communities = [set(c) for c in limited]

    # annotate nodes
    for idx, comm in enumerate(communities):
        for node in comm:
            G.nodes[node]["community_id"] = idx

    return communities

def export_communities(G, communities, out_dir="components"):
    os.makedirs(out_dir, exist_ok=True)
    for idx, comm in enumerate(communities):
        subG = G.subgraph(comm).copy()
        out_path = os.path.join(out_dir, f"community_{idx}.gml")
        write_gml(subG, out_path)
        print(f"[output] wrote -> {out_path}")

def plot_attributes(G, out_path="plots/attributes.png",
                    node_color_attr="color", edge_sign_attr="sign",
                    quiet: bool = False):

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pos = nx.spring_layout(G, seed=42)
    node_colors = [G.nodes[n].get(node_color_attr, "gray") for n in G.nodes()]
    edge_colors = ["green" if G[u][v].get(edge_sign_attr, "+") == "+" else "red" for u, v in G.edges()]
    plt.figure(figsize=(8,6))
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=2, alpha=0.9)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=320, edgecolors="black")
    nx.draw_networkx_labels(G, pos, font_size=9)
    plt.title(f"Attributes: nodes by '{node_color_attr}', edges by '{edge_sign_attr}' (+=green, -=red)")
    plt.axis("off"); plt.tight_layout()
    plt.savefig(out_path, dpi=300); plt.close()
    if not quiet:
        print(f"[plot] saved -> {out_path}")


def compute_neighborhood_overlap(G):
    """Compute edge-wise Jaccard overlap and store as edge attribute 'overlap'."""
    for u, v in G.edges():
        Nu = set(G.neighbors(u)) - {v}
        Nv = set(G.neighbors(v)) - {u}
        inter = len(Nu & Nv)
        union = len((Nu | Nv))
        ov = 0.0 if union == 0 else inter / union
        G[u][v]["overlap"] = float(ov)
    return G

def plot_overlap(G, out_path="plots/overlap.png"):
    """Plot edge thickness = neighborhood overlap; edge color = sum of endpoint degrees."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # ensure overlap is computed
    if not all("overlap" in G[u][v] for u, v in G.edges()):
        compute_neighborhood_overlap(G)

    deg = dict(G.degree())
    pos = nx.spring_layout(G, seed=42)

    # widths by overlap; colors by sum of endpoint degrees
    widths = [1.0 + 6.0 * G[u][v]["overlap"] for u, v in G.edges()]
    ecolors = [deg[u] + deg[v] for u, v in G.edges()]

    plt.figure(figsize=(8, 6))
    ec = nx.draw_networkx_edges(
        G, pos, width=widths, edge_color=ecolors, edge_cmap=plt.cm.plasma, alpha=0.9
    )
    nx.draw_networkx_nodes(G, pos, node_size=300, node_color="lightgray", edgecolors="black")
    nx.draw_networkx_labels(G, pos, font_size=9)

    cbar = plt.colorbar(ec, ax=plt.gca(), label="Sum of endpoint degrees")
    plt.title("Neighborhood Overlap (edge thickness)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[plot] saved -> {out_path}")

def parse_args():
    p = argparse.ArgumentParser(description="CECS 427 Graph Analysis")
    p.add_argument("graph_file", help="Path to input .gml graph")
    p.add_argument("--output", default="out.gml", help="Where to save the enriched graph (.gml)")
    p.add_argument("--components", type=int, help="Partition into N components (Girvan-Newman)")
    p.add_argument("--split_output_dir", help="Optional: export each community to this directory")
    p.add_argument("--simulate_failures", type=int, help="Remove k random edges and analyze graph")
    p.add_argument("--robustness_check", type=int, help="Simulate repeated failures of k edges")
    p.add_argument("--verify_homophily", action="store_true", help="Z-test for homophily (node attribute 'color')")
    p.add_argument("--verify_balanced_graph", action="store_true",
               help="Check structural balance (edge attr 'sign' = '+'/'-')")

    p.add_argument("--temporal_simulation", help="CSV file with source,target,timestamp,action")
    p.add_argument(
    "--plot",
    choices=list("CNPT"),
    help="C=clustering, N=overlap, P=attributes, T=temporal (requires --temporal_simulation)"
    )

    return p.parse_args()


def visualize_graph(G, out_path="plots/overview.png"):
    """Draws the graph and saves to a PNG file."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)  # make sure folder exists
    pos = nx.spring_layout(G, seed=42)  # layout (fixed seed = stable drawing)
    nx.draw(
        G, pos,
        with_labels=True,
        node_color="lightblue",
        edge_color="gray",
        node_size=800,
        font_size=10
    )
    plt.title("Graph Overview")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[plot] saved -> {out_path}")

def main() -> int:
    args = parse_args()
    print(f"[info] loading: {args.graph_file}")
    try:
        G = read_gml(args.graph_file)
    except Exception as e:
        print(f"[error] {e}", file=sys.stderr)
        return 2

    # --- analysis flags ---
    if args.components:
        comms = girvan_newman_partition(G, args.components)
        sizes = [len(c) for c in comms]
        print(f"[partition] GN components={len(comms)} sizes={sizes}")
        if args.split_output_dir:
            export_communities(G, comms, args.split_output_dir)

    if args.simulate_failures:
        simulate_failures(G, args.simulate_failures)

    if args.robustness_check:
        robustness_check(G, args.robustness_check)

    if args.verify_homophily:
        verify_homophily(G)

    if args.verify_balanced_graph:
        verify_balanced_graph(G)

    # --- summary ---
    n, m = G.number_of_nodes(), G.number_of_edges()
    print(f"[info] loaded graph: n={n} m={m}")

    # --- plotting switch ---
    if args.plot == "C":
        plot_clustering(G)
    elif args.plot == "N":
        compute_neighborhood_overlap(G)
        plot_overlap(G)
    elif args.plot == "P":
        plot_attributes(G)
    elif args.plot == "T":
        if not args.temporal_simulation:
            print("[plot] 'T' requires --temporal_simulation <csv>", file=sys.stderr)
        else:
            steps = apply_temporal_csv(G.copy(), args.temporal_simulation)
            render_temporal_frames(steps, out_dir="plots")
    else:
        visualize_graph(G)

    # --- save graph ---
    try:
        write_gml(G, args.output)
        print(f"[output] wrote -> {args.output}")
    except Exception as e:
        print(f"[error] failed to write output: {e}", file=sys.stderr)
        return 3

    return 0



if __name__ == "__main__":
    sys.exit(main())
