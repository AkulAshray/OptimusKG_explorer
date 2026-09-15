"""Typed graph exploration helpers for the OptimusKG tutorial layer.

The functions in this module operate on the deliberately scoped Parquet
snapshot used by the project.  They preserve node and relationship types and
make query limits visible.  They are exploratory helpers, not causal or
clinical inference methods.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, Sequence, TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from .core import OptimusExplorer


NODE_TYPES = {
    "disease",
    "drug",
    "gene",
    "biological_process",
    "pathway",
}

EDGE_COLUMNS = [
    "source_type", "source_id", "source_name", "relationship",
    "target_type", "target_id", "target_name", "source_table", "derived",
]


def _normalise_type(node_type: str) -> str:
    value = node_type.strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "process": "biological_process",
        "biologicalprocess": "biological_process",
        "compound": "drug",
        "protein": "gene",
    }
    value = aliases.get(value, value)
    if value not in NODE_TYPES:
        raise ValueError(f"Unknown node type {node_type!r}. Choose from {sorted(NODE_TYPES)}.")
    return value


def _edge_frame(
    frame: pl.DataFrame,
    *,
    source_type: str,
    source_id: str,
    source_name: str,
    relationship: str | pl.Expr,
    target_type: str,
    target_id: str,
    target_name: str,
    source_table: str,
    derived: bool = False,
) -> pl.DataFrame:
    relation_expr = (
        pl.col(relationship).cast(pl.String).str.to_uppercase()
        if isinstance(relationship, str) and relationship in frame.columns
        else relationship
        if isinstance(relationship, pl.Expr)
        else pl.lit(relationship)
    )
    source_name_expr = pl.col(source_name) if source_name in frame.columns else pl.col(source_id)
    target_name_expr = pl.col(target_name) if target_name in frame.columns else pl.col(target_id)
    return (
        frame.select(
            pl.lit(source_type).alias("source_type"),
            pl.col(source_id).cast(pl.String).alias("source_id"),
            source_name_expr.cast(pl.String).alias("source_name"),
            relation_expr.alias("relationship"),
            pl.lit(target_type).alias("target_type"),
            pl.col(target_id).cast(pl.String).alias("target_id"),
            target_name_expr.cast(pl.String).alias("target_name"),
            pl.lit(source_table).alias("source_table"),
            pl.lit(derived).alias("derived"),
        )
        .drop_nulls(["source_id", "target_id", "relationship"])
        .unique()
    )


def build_edge_catalog(explorer: "OptimusExplorer") -> pl.DataFrame:
    """Create one canonical, typed edge table from the scoped snapshot."""
    tables = explorer.tables
    edges = [
        _edge_frame(
            tables["disease_gene"],
            source_type="disease", source_id="disease_id", source_name="disease_name",
            relationship="ASSOCIATED_WITH",
            target_type="gene", target_id="gene_id", target_name="gene_symbol",
            source_table="disease_gene", derived=False,
        ),
        _edge_frame(
            tables["drug_gene"],
            source_type="drug", source_id="drug_id", source_name="drug_name",
            relationship="drug_gene_relation",
            target_type="gene", target_id="gene_id", target_name="gene_symbol",
            source_table="drug_gene", derived=False,
        ),
        _edge_frame(
            tables["process"],
            source_type="gene", source_id="gene_id", source_name="gene_symbol",
            relationship="PARTICIPATES_IN_PROCESS",
            target_type="biological_process", target_id="annotation_id",
            target_name="annotation_name", source_table="process", derived=False,
        ),
        _edge_frame(
            tables["pathway"],
            source_type="gene", source_id="gene_id", source_name="gene_symbol",
            relationship="PARTICIPATES_IN_PATHWAY",
            target_type="pathway", target_id="annotation_id",
            target_name="annotation_name", source_table="pathway", derived=False,
        ),
    ]

    pairs = tables["pair_summary"]
    edges.append(_edge_frame(
        pairs,
        source_type="drug", source_id="drug_id", source_name="drug_name",
        relationship="CANDIDATE_PAIR",
        target_type="disease", target_id="disease_id", target_name="disease_name",
        source_table="pair_summary", derived=True,
    ))
    for flag, relationship in [
        ("has_recorded_indication", "RECORDED_INDICATION"),
        ("has_recorded_off_label_use", "RECORDED_OFF_LABEL_USE"),
        ("has_recorded_contraindication", "RECORDED_CONTRAINDICATION"),
    ]:
        if flag in pairs.columns:
            flagged = pairs.filter(pl.col(flag).fill_null(False))
            edges.append(_edge_frame(
                flagged,
                source_type="drug", source_id="drug_id", source_name="drug_name",
                relationship=relationship,
                target_type="disease", target_id="disease_id", target_name="disease_name",
                source_table="pair_summary", derived=False,
            ))
    return pl.concat(edges, how="vertical_relaxed").select(EDGE_COLUMNS).unique()


def build_node_catalog(explorer: "OptimusExplorer") -> pl.DataFrame:
    """Create a searchable catalogue with user-friendly aliases."""
    t = explorer.tables
    process_gene_name = pl.col("gene_symbol") if "gene_symbol" in t["process"].columns else pl.col("gene_id")
    pathway_gene_name = pl.col("gene_symbol") if "gene_symbol" in t["pathway"].columns else pl.col("gene_id")
    frames = [
        t["disease_gene"].select(
            pl.lit("disease").alias("node_type"),
            pl.col("disease_id").cast(pl.String).alias("node_id"),
            pl.col("disease_name").cast(pl.String).alias("name"),
            pl.col("disease_code").cast(pl.String).alias("alias"),
        ),
        t["drug_gene"].select(
            pl.lit("drug").alias("node_type"),
            pl.col("drug_id").cast(pl.String).alias("node_id"),
            pl.col("drug_name").cast(pl.String).alias("name"),
            pl.col("drug_name").cast(pl.String).alias("alias"),
        ),
        t["disease_gene"].select(
            pl.lit("gene").alias("node_type"),
            pl.col("gene_id").cast(pl.String).alias("node_id"),
            pl.col("gene_symbol").cast(pl.String).alias("name"),
            pl.col("gene_symbol").cast(pl.String).alias("alias"),
        ),
        t["process"].select(
            pl.lit("biological_process").alias("node_type"),
            pl.col("annotation_id").cast(pl.String).alias("node_id"),
            pl.col("annotation_name").cast(pl.String).alias("name"),
            pl.col("annotation_name").cast(pl.String).alias("alias"),
        ),
        t["process"].select(
            pl.lit("gene").alias("node_type"),
            pl.col("gene_id").cast(pl.String).alias("node_id"),
            process_gene_name.cast(pl.String).alias("name"),
            process_gene_name.cast(pl.String).alias("alias"),
        ),
        t["pathway"].select(
            pl.lit("gene").alias("node_type"),
            pl.col("gene_id").cast(pl.String).alias("node_id"),
            pathway_gene_name.cast(pl.String).alias("name"),
            pathway_gene_name.cast(pl.String).alias("alias"),
        ),
        t["pathway"].select(
            pl.lit("pathway").alias("node_type"),
            pl.col("annotation_id").cast(pl.String).alias("node_id"),
            pl.col("annotation_name").cast(pl.String).alias("name"),
            pl.col("annotation_name").cast(pl.String).alias("alias"),
        ),
    ]
    return pl.concat(frames, how="vertical_relaxed").drop_nulls(["node_id"]).unique()


@dataclass(frozen=True)
class PathQueryResult:
    """Bounded paths returned by a generic typed query."""

    data: pl.DataFrame
    start: "NodeView"
    target_type: str
    target_id: str | None
    pattern: tuple[str, ...] | None
    max_hops: int
    max_paths: int
    expansions: int
    truncated: bool
    include_derived: bool

    def __len__(self) -> int:
        return self.data.height

    def table(self, n: int | None = 50) -> pl.DataFrame:
        return self.data if n is None else self.data.head(n)

    def status(self) -> pl.DataFrame:
        return pl.DataFrame({
            "returned_paths": [len(self)],
            "max_paths": [self.max_paths],
            "max_hops": [self.max_hops],
            "expansions": [self.expansions],
            "truncated": [self.truncated],
            "include_derived": [self.include_derived],
        })

    def summary(self) -> pl.DataFrame:
        if self.data.is_empty():
            return pl.DataFrame({
                "hop_count": pl.Series([], dtype=pl.Int64),
                "path_pattern": pl.Series([], dtype=pl.String),
                "paths": pl.Series([], dtype=pl.UInt32),
            })
        return (
            self.data.group_by("hop_count", "path_pattern")
            .agg(pl.len().alias("paths"))
            .sort("hop_count", "paths", descending=[False, True])
        )

    def provenance(self) -> pl.DataFrame:
        if self.data.is_empty():
            return pl.DataFrame()
        return (
            self.data.explode("source_tables")
            .group_by("source_tables")
            .agg(pl.len().alias("path_uses"))
            .sort("path_uses", descending=True)
        )

    def warnings(self) -> list[str]:
        messages = [
            "Returned paths are recorded graph connections, not causal mechanisms.",
            "Path counts depend on the selected node types, relationships and snapshot.",
        ]
        if self.truncated:
            messages.append("The query reached a safety limit; counts are incomplete.")
        if self.include_derived:
            messages.append("The query allowed derived analytical edges such as CANDIDATE_PAIR.")
        return messages

    def plot(self, n: int = 5):
        """Plot paths as separate horizontal cards to avoid network hairballs."""
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError("Install matplotlib to plot paths.") from exc
        sample = self.data.head(n)
        if sample.is_empty():
            raise ValueError("There are no paths to plot.")
        figure, axes = plt.subplots(sample.height, 1, figsize=(14, 2.2 * sample.height))
        if sample.height == 1:
            axes = [axes]
        for axis, row in zip(axes, sample.iter_rows(named=True)):
            nodes = row["node_names"]
            relations = row["relationships"]
            directions = row["directions"]
            count = len(nodes)
            for index, label in enumerate(nodes):
                x = index / max(count - 1, 1)
                axis.text(
                    x, 0.55, label, ha="center", va="center", fontsize=9,
                    bbox={"boxstyle": "round,pad=0.4", "facecolor": "#dbeafe", "edgecolor": "#60a5fa"},
                    transform=axis.transAxes,
                )
                if index < len(relations):
                    next_x = (index + 1) / max(count - 1, 1)
                    arrow = "->" if directions[index] == "forward" else "<-"
                    axis.annotate(
                        "", xy=(next_x - 0.04, 0.55), xytext=(x + 0.04, 0.55),
                        xycoords="axes fraction", textcoords="axes fraction",
                        arrowprops={"arrowstyle": arrow, "color": "#64748b"},
                    )
                    axis.text((x + next_x) / 2, 0.72, relations[index], ha="center", fontsize=7,
                              color="#475569", transform=axis.transAxes)
            axis.set_title(f"Path {row['path_id']} · {row['path_pattern']}", loc="left", fontsize=9)
            axis.axis("off")
        figure.tight_layout()
        return figure


@dataclass(frozen=True)
class GraphProjection:
    """A documented graph projection used for topology calculations."""

    nodes: pl.DataFrame
    edges: pl.DataFrame
    directed: bool
    specification: dict
    truncated: bool = False

    def to_networkx(self):
        try:
            import networkx as nx
        except ImportError as exc:
            raise ImportError("Install networkx to materialise a projection.") from exc
        graph = nx.DiGraph() if self.directed else nx.Graph()
        for row in self.nodes.iter_rows(named=True):
            graph.add_node((row["node_type"], row["node_id"]), **row)
        for row in self.edges.iter_rows(named=True):
            a = (row["source_type"], row["source_id"])
            b = (row["target_type"], row["target_id"])
            if graph.has_edge(a, b):
                graph[a][b]["multiplicity"] = graph[a][b].get("multiplicity", 1) + 1
                graph[a][b].setdefault("relationships", set()).add(row["relationship"])
            else:
                graph.add_edge(
                    a, b, multiplicity=1, relationships={row["relationship"]},
                    source_table=row["source_table"], derived=row["derived"],
                )
        return graph

    def status(self) -> pl.DataFrame:
        return pl.DataFrame({
            "nodes": [self.nodes.height], "typed_edges": [self.edges.height],
            "directed": [self.directed], "truncated": [self.truncated],
        })

    def summary(self) -> pl.DataFrame:
        import networkx as nx
        graph = self.to_networkx()
        if self.directed:
            components = list(nx.weakly_connected_components(graph))
        else:
            components = list(nx.connected_components(graph))
        return pl.DataFrame({
            "measure": [
                "projection nodes", "typed edges", "collapsed graph edges",
                "connected components", "largest component nodes", "isolated nodes",
            ],
            "value": [
                self.nodes.height, self.edges.height, graph.number_of_edges(),
                len(components), max((len(c) for c in components), default=0),
                len(list(nx.isolates(graph))),
            ],
        })

    def components(self) -> pl.DataFrame:
        import networkx as nx
        graph = self.to_networkx()
        groups = (
            nx.weakly_connected_components(graph)
            if self.directed else nx.connected_components(graph)
        )
        rows = []
        for component_id, members in enumerate(sorted(groups, key=len, reverse=True), start=1):
            for node_type, node_id in members:
                rows.append({"component": component_id, "node_type": node_type, "node_id": node_id})
        return pl.DataFrame(rows) if rows else pl.DataFrame()

    def centrality(self, metric: str = "degree", top: int | None = 30, approximate: bool = False) -> pl.DataFrame:
        import networkx as nx
        graph = self.to_networkx()
        metric = metric.lower()
        if metric == "degree":
            scores = nx.degree_centrality(graph)
        elif metric == "betweenness":
            k = min(100, graph.number_of_nodes()) if approximate and graph.number_of_nodes() else None
            scores = nx.betweenness_centrality(graph, k=k, seed=7)
        elif metric == "pagerank":
            scores = nx.pagerank(graph)
        else:
            raise ValueError("metric must be 'degree', 'betweenness', or 'pagerank'.")
        names = {(r["node_type"], r["node_id"]): r["name"] for r in self.nodes.iter_rows(named=True)}
        rows = [
            {
                "node_type": key[0], "node_id": key[1], "name": names.get(key),
                "metric": metric, "score": float(score), "degree": int(graph.degree(key)),
            }
            for key, score in scores.items()
        ]
        result = pl.DataFrame(rows).sort("score", "degree", descending=[True, True]) if rows else pl.DataFrame()
        if not result.is_empty():
            result = result.with_row_index("rank", offset=1)
        return result if top is None else result.head(top)

    def hub_report(self, top: int = 20) -> pl.DataFrame:
        return self.centrality("degree", top=top).rename({"score": "degree_centrality"})

    def degree_distribution(self) -> pl.DataFrame:
        graph = self.to_networkx()
        values = [degree for _, degree in graph.degree]
        if not values:
            return pl.DataFrame()
        return (
            pl.DataFrame({"degree": values})
            .group_by("degree").agg(pl.len().alias("nodes"))
            .sort("degree")
        )

    def provenance(self) -> pl.DataFrame:
        if self.edges.is_empty():
            return pl.DataFrame()
        return (
            self.edges.group_by("source_table", "relationship", "derived")
            .agg(pl.len().alias("typed_edges"))
            .sort("typed_edges", descending=True)
        )

    def warnings(self) -> list[str]:
        relationships = self.edges.get_column("relationship").n_unique() if not self.edges.is_empty() else 0
        messages = ["Topology metrics describe this projection, not the complete biological system."]
        if relationships > 1:
            messages.append("This projection mixes relationship types; interpret centrality cautiously.")
        if self.truncated:
            messages.append("The projection reached max_nodes and is incomplete.")
        return messages

    def plot(self, max_nodes: int = 100, seed: int = 7, with_labels: bool = False):
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
        except ImportError as exc:
            raise ImportError("Install matplotlib and networkx to plot projections.") from exc
        graph = self.to_networkx()
        display_graph = graph
        if graph.number_of_nodes() > max_nodes:
            ranked = sorted(graph.degree, key=lambda item: (-item[1], str(item[0])))[:max_nodes]
            display_graph = graph.subgraph([node for node, _ in ranked]).copy()
        colors_by_type = {
            "disease": "#ef4444", "drug": "#2563eb", "gene": "#8b5cf6",
            "biological_process": "#10b981", "pathway": "#f59e0b",
        }
        colors = [colors_by_type.get(node[0], "#cbd5e1") for node in display_graph.nodes]
        positions = nx.spring_layout(display_graph, seed=seed)
        figure, axis = plt.subplots(figsize=(12, 8))
        labels = {
            node: display_graph.nodes[node].get("name", node[1])
            for node in display_graph.nodes
        } if with_labels else None
        nx.draw_networkx(
            display_graph, positions, ax=axis, node_color=colors, node_size=180,
            width=0.7, edge_color="#94a3b8", labels=labels, font_size=7,
            arrows=self.directed,
        )
        axis.set_title(f"Projection view ({display_graph.number_of_nodes()} displayed nodes)")
        axis.axis("off")
        figure.tight_layout()
        return figure


@dataclass(frozen=True)
class NodeView:
    """A resolved entity that can be explored without writing graph queries."""

    explorer: "OptimusExplorer"
    node_type: str
    node_id: str
    name: str

    def info(self) -> pl.DataFrame:
        return pl.DataFrame({
            "node_type": [self.node_type], "node_id": [self.node_id], "name": [self.name],
        })

    def neighbours(
        self,
        node_type: str | None = None,
        relations: Iterable[str] | None = None,
        direction: str = "both",
        include_derived: bool = False,
        limit: int | None = 100,
    ) -> pl.DataFrame:
        direction = direction.lower()
        if direction not in {"both", "incoming", "outgoing"}:
            raise ValueError("direction must be 'both', 'incoming', or 'outgoing'.")
        target_type = _normalise_type(node_type) if node_type else None
        relation_set = {r.upper() for r in relations} if relations else None
        edges = self.explorer.edge_catalog()
        if not include_derived:
            edges = edges.filter(~pl.col("derived"))
        if relation_set:
            edges = edges.filter(pl.col("relationship").is_in(sorted(relation_set)))
        frames = []
        if direction in {"both", "outgoing"}:
            frames.append(
                edges.filter(
                    (pl.col("source_type") == self.node_type) & (pl.col("source_id") == self.node_id)
                ).select(
                    "relationship", pl.lit("outgoing").alias("direction"),
                    pl.col("target_type").alias("neighbour_type"),
                    pl.col("target_id").alias("neighbour_id"),
                    pl.col("target_name").alias("neighbour_name"), "source_table", "derived",
                )
            )
        if direction in {"both", "incoming"}:
            frames.append(
                edges.filter(
                    (pl.col("target_type") == self.node_type) & (pl.col("target_id") == self.node_id)
                ).select(
                    "relationship", pl.lit("incoming").alias("direction"),
                    pl.col("source_type").alias("neighbour_type"),
                    pl.col("source_id").alias("neighbour_id"),
                    pl.col("source_name").alias("neighbour_name"), "source_table", "derived",
                )
            )
        result = pl.concat(frames, how="vertical_relaxed").unique() if frames else pl.DataFrame()
        if target_type and not result.is_empty():
            result = result.filter(pl.col("neighbour_type") == target_type)
        if not result.is_empty():
            result = result.sort("neighbour_type", "relationship", "neighbour_name")
        return result if limit is None else result.head(limit)

    def degree(self, by_relation: bool = True, include_derived: bool = False) -> pl.DataFrame:
        neighbours = self.neighbours(include_derived=include_derived, limit=None)
        if neighbours.is_empty():
            return pl.DataFrame()
        if not by_relation:
            return pl.DataFrame({
                "node_id": [self.node_id], "distinct_neighbours": [neighbours["neighbour_id"].n_unique()],
                "typed_edges": [neighbours.height],
            })
        return (
            neighbours.group_by("direction", "relationship", "neighbour_type")
            .agg(
                pl.col("neighbour_id").n_unique().alias("distinct_neighbours"),
                pl.len().alias("typed_edges"),
            )
            .sort("typed_edges", descending=True)
        )

    def ego(
        self,
        radius: int = 1,
        relations: Iterable[str] | None = None,
        node_types: Iterable[str] | None = None,
        include_derived: bool = False,
        directed: bool = False,
        max_nodes: int = 250,
    ) -> GraphProjection:
        return self.explorer.project(
            node_types=node_types, relations=relations, seeds=[self], radius=radius,
            include_derived=include_derived, directed=directed, max_nodes=max_nodes,
        )

    def paths_to(
        self,
        target_type: str,
        target: str | None = None,
        pattern: Sequence[str] | None = None,
        relations: Iterable[str] | None = None,
        max_hops: int = 3,
        max_paths: int = 100,
        include_derived: bool = False,
        direction: str = "both",
        max_expansions: int = 200_000,
    ) -> PathQueryResult:
        return _find_paths(
            self, target_type=target_type, target=target, pattern=pattern,
            relations=relations, max_hops=max_hops, max_paths=max_paths,
            include_derived=include_derived, direction=direction,
            max_expansions=max_expansions,
        )


def _find_paths(
    start: NodeView,
    *,
    target_type: str,
    target: str | None,
    pattern: Sequence[str] | None,
    relations: Iterable[str] | None,
    max_hops: int,
    max_paths: int,
    include_derived: bool,
    direction: str,
    max_expansions: int,
) -> PathQueryResult:
    target_type = _normalise_type(target_type)
    if not 1 <= max_hops <= 4:
        raise ValueError("max_hops must be between 1 and 4 for exploratory queries.")
    if not 1 <= max_paths <= 5000:
        raise ValueError("max_paths must be between 1 and 5000.")
    if direction not in {"both", "incoming", "outgoing"}:
        raise ValueError("direction must be 'both', 'incoming', or 'outgoing'.")
    typed_pattern = tuple(_normalise_type(value) for value in pattern) if pattern else None
    if typed_pattern:
        if typed_pattern[0] != start.node_type or typed_pattern[-1] != target_type:
            raise ValueError("pattern must start at this node type and end at target_type.")
        if len(typed_pattern) - 1 > max_hops:
            raise ValueError("pattern requires more hops than max_hops permits.")
    target_id = start.explorer.node(target_type, target).node_id if target else None
    relation_set = {r.upper() for r in relations} if relations else None
    edges = start.explorer.edge_catalog()
    if not include_derived:
        edges = edges.filter(~pl.col("derived"))
    if relation_set:
        edges = edges.filter(pl.col("relationship").is_in(sorted(relation_set)))

    allowed_type_pairs = None
    if typed_pattern:
        allowed_type_pairs = {
            pair
            for a, b in zip(typed_pattern[:-1], typed_pattern[1:])
            for pair in ((a, b), (b, a))
        }

    adjacency: dict[tuple[str, str], list[dict]] = {}
    for row in edges.iter_rows(named=True):
        if allowed_type_pairs and (row["source_type"], row["target_type"]) not in allowed_type_pairs:
            continue
        source = (row["source_type"], row["source_id"])
        target_key = (row["target_type"], row["target_id"])
        if direction in {"both", "outgoing"}:
            adjacency.setdefault(source, []).append({
                "next": target_key, "next_name": row["target_name"],
                "relationship": row["relationship"], "direction": "forward",
                "source_table": row["source_table"], "derived": row["derived"],
            })
        if direction in {"both", "incoming"}:
            adjacency.setdefault(target_key, []).append({
                "next": source, "next_name": row["source_name"],
                "relationship": row["relationship"], "direction": "reverse",
                "source_table": row["source_table"], "derived": row["derived"],
            })

    start_key = (start.node_type, start.node_id)
    queue = deque([(start_key, [start_key], [start.name], [])])
    rows = []
    expansions = 0
    truncated = False
    while queue:
        current, node_path, name_path, edge_path = queue.popleft()
        depth = len(edge_path)
        if depth and current[0] == target_type and (target_id is None or current[1] == target_id):
            exact_pattern = tuple(node[0] for node in node_path)
            if typed_pattern is None or exact_pattern == typed_pattern:
                rows.append({
                    "path_id": len(rows) + 1,
                    "hop_count": depth,
                    "path_pattern": " -> ".join(exact_pattern),
                    "start_type": start.node_type,
                    "start_id": start.node_id,
                    "end_type": current[0],
                    "end_id": current[1],
                    "end_name": name_path[-1],
                    "node_ids": [node[1] for node in node_path],
                    "node_names": name_path,
                    "relationships": [edge["relationship"] for edge in edge_path],
                    "directions": [edge["direction"] for edge in edge_path],
                    "source_tables": sorted({edge["source_table"] for edge in edge_path}),
                    "uses_derived_edge": any(edge["derived"] for edge in edge_path),
                    "readable_path": " | ".join(
                        f"{name_path[i]} [{'->' if edge['direction'] == 'forward' else '<-'} {edge['relationship']}]"
                        for i, edge in enumerate(edge_path)
                    ) + f" | {name_path[-1]}",
                })
                if len(rows) >= max_paths:
                    truncated = bool(queue)
                    break
            continue
        if depth >= max_hops:
            continue
        for edge in adjacency.get(current, []):
            next_key = edge["next"]
            if next_key in node_path:
                continue
            if typed_pattern and depth + 1 < len(typed_pattern):
                if next_key[0] != typed_pattern[depth + 1]:
                    continue
            expansions += 1
            if expansions >= max_expansions:
                truncated = True
                queue.clear()
                break
            queue.append((
                next_key,
                node_path + [next_key],
                name_path + [edge["next_name"] or next_key[1]],
                edge_path + [edge],
            ))
    data = pl.DataFrame(rows) if rows else pl.DataFrame({
        "path_id": pl.Series([], dtype=pl.Int64),
        "hop_count": pl.Series([], dtype=pl.Int64),
        "path_pattern": pl.Series([], dtype=pl.String),
        "readable_path": pl.Series([], dtype=pl.String),
    })
    return PathQueryResult(
        data=data, start=start, target_type=target_type, target_id=target_id,
        pattern=typed_pattern, max_hops=max_hops, max_paths=max_paths,
        expansions=expansions, truncated=truncated, include_derived=include_derived,
    )


def build_projection(
    explorer: "OptimusExplorer",
    *,
    node_types: Iterable[str] | None = None,
    relations: Iterable[str] | None = None,
    seeds: Iterable[NodeView | tuple[str, str]] | None = None,
    radius: int | None = None,
    include_derived: bool = False,
    directed: bool = False,
    max_nodes: int = 5000,
) -> GraphProjection:
    """Build a documented, bounded graph projection."""
    if max_nodes < 1:
        raise ValueError("max_nodes must be positive.")
    allowed_types = {_normalise_type(t) for t in node_types} if node_types else None
    allowed_relations = {r.upper() for r in relations} if relations else None
    edges = explorer.edge_catalog()
    if not include_derived:
        edges = edges.filter(~pl.col("derived"))
    if allowed_types:
        edges = edges.filter(
            pl.col("source_type").is_in(sorted(allowed_types))
            & pl.col("target_type").is_in(sorted(allowed_types))
        )
    if allowed_relations:
        edges = edges.filter(pl.col("relationship").is_in(sorted(allowed_relations)))

    truncated = False
    selected_keys: set[tuple[str, str]] | None = None
    seed_keys: list[tuple[str, str]] = []
    if seeds:
        for seed in seeds:
            if isinstance(seed, NodeView):
                seed_keys.append((seed.node_type, seed.node_id))
            else:
                resolved = explorer.node(seed[0], seed[1])
                seed_keys.append((resolved.node_type, resolved.node_id))
        if radius is None or radius < 0 or radius > 4:
            raise ValueError("Seeded projections require radius between 0 and 4.")
        adjacency: dict[tuple[str, str], set[tuple[str, str]]] = {}
        for row in edges.iter_rows(named=True):
            a = (row["source_type"], row["source_id"])
            b = (row["target_type"], row["target_id"])
            adjacency.setdefault(a, set()).add(b)
            adjacency.setdefault(b, set()).add(a)
        selected_keys = set(seed_keys)
        queue = deque((key, 0) for key in seed_keys)
        while queue:
            current, depth = queue.popleft()
            if depth >= radius:
                continue
            for neighbour in sorted(adjacency.get(current, set())):
                if neighbour in selected_keys:
                    continue
                if len(selected_keys) >= max_nodes:
                    truncated = True
                    queue.clear()
                    break
                selected_keys.add(neighbour)
                queue.append((neighbour, depth + 1))
        key_frame = pl.DataFrame({
            "node_type": [k[0] for k in selected_keys],
            "node_id": [k[1] for k in selected_keys],
        })
        sources = key_frame.rename({"node_type": "source_type", "node_id": "source_id"})
        targets = key_frame.rename({"node_type": "target_type", "node_id": "target_id"})
        edges = edges.join(sources, on=["source_type", "source_id"], how="inner").join(
            targets, on=["target_type", "target_id"], how="inner"
        )

    all_nodes = explorer.node_catalog()
    endpoint_rows = pl.concat([
        edges.select(
            pl.col("source_type").alias("node_type"), pl.col("source_id").alias("node_id"),
            pl.col("source_name").alias("name"),
        ),
        edges.select(
            pl.col("target_type").alias("node_type"), pl.col("target_id").alias("node_id"),
            pl.col("target_name").alias("name"),
        ),
    ], how="vertical_relaxed").group_by("node_type", "node_id").agg(
        pl.col("name").drop_nulls().first().alias("name")
    )
    if selected_keys:
        seed_only = all_nodes.join(
            pl.DataFrame({
                "node_type": [k[0] for k in selected_keys],
                "node_id": [k[1] for k in selected_keys],
            }), on=["node_type", "node_id"], how="inner",
        ).select("node_type", "node_id", "name").unique()
        endpoint_rows = pl.concat([endpoint_rows, seed_only], how="vertical_relaxed").group_by(
            "node_type", "node_id"
        ).agg(pl.col("name").drop_nulls().first().alias("name"))
    return GraphProjection(
        nodes=endpoint_rows.sort("node_type", "name"),
        edges=edges.select(EDGE_COLUMNS).unique(),
        directed=directed,
        specification={
            "node_types": sorted(allowed_types) if allowed_types else None,
            "relations": sorted(allowed_relations) if allowed_relations else None,
            "seeds": seed_keys or None,
            "radius": radius,
            "include_derived": include_derived,
            "max_nodes": max_nodes,
        },
        truncated=truncated,
    )
