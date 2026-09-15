"""Progressive exploration layer for OptimusKG and project snapshots.

The snapshot API used by the validated Tab 3 path audit is retained. Version
0.4 adds a live, deliberately incremental API on top of the official
``optimuskg`` client: no node or edge table is downloaded until the user asks
for it, and loaded tables can be activated, deactivated, or removed from the
current analytical graph. It also supports evidence-aware neighbourhood
inspection and portable CSV, Parquet, and DuckDB exports.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import polars as pl


DEFAULT_FILES = {
    "disease_gene": "disease_gene_evidence.parquet",
    "drug_gene": "candidate_drug_gene_evidence.parquet",
    "pair_summary": "candidate_drug_disease_summary.parquet",
    "clinical": "candidate_drug_clinical_relations.parquet",
    "direct_paths": "direct_target_convergence_paths.parquet",
    "process": "gene_biological_process_evidence.parquet",
    "pathway": "gene_pathway_evidence.parquet",
}

REQUIRED_COLUMNS = {
    "disease_gene": {
        "disease_code", "disease_id", "disease_name", "gene_id",
        "gene_symbol", "evidence_score",
    },
    "drug_gene": {
        "drug_id", "drug_name", "gene_id", "gene_symbol",
        "drug_gene_relation",
    },
    "pair_summary": {
        "disease_code", "disease_id", "disease_name", "drug_id",
        "drug_name", "has_recorded_indication",
    },
    "direct_paths": {
        "disease_code", "disease_id", "drug_id", "gene_id",
        "drug_gene_relation",
    },
    "process": {"gene_id", "annotation_id", "annotation_name"},
    "pathway": {"gene_id", "annotation_id", "annotation_name"},
}

PATH_FAMILIES = {
    "direct_convergence": {
        "label": "Direct convergence",
        "structure": "Disease -> Gene <- Drug",
        "interpretation": "The drug acts on a gene recorded as associated with the disease.",
        "warning": "A recorded overlap does not establish therapeutic effect.",
    },
    "shared_pathway": {
        "label": "Shared pathway",
        "structure": "Disease -> Gene -> Pathway <- Gene <- Drug",
        "interpretation": "A disease gene and drug target share a pathway annotation.",
        "warning": "A shared pathway is a context, not a causal mechanism.",
    },
    "shared_process": {
        "label": "Shared process",
        "structure": "Disease -> Gene -> Process <- Gene <- Drug",
        "interpretation": "A disease gene and drug target share a biological-process annotation.",
        "warning": "Broad GO terms can generate many redundant paths.",
    },
    "gene_interaction": {
        "label": "Gene interaction",
        "structure": "Disease -> Gene -- Gene <- Drug",
        "interpretation": "A disease gene interacts with a recorded drug target.",
        "warning": "An interaction does not specify therapeutic direction or tissue context.",
    },
}

PATH_COLUMNS = [
    "disease_code", "disease_id", "disease_name", "drug_id", "drug_name",
    "path_family", "hop_count", "disease_gene_id", "disease_gene_symbol",
    "drug_target_gene_id", "drug_target_gene_symbol", "drug_gene_relation",
    "annotation_type", "annotation_id", "annotation_name",
    "disease_gene_evidence_score",
]


def _empty_paths() -> pl.DataFrame:
    return pl.DataFrame({
        "disease_code": pl.Series([], dtype=pl.String),
        "disease_id": pl.Series([], dtype=pl.String),
        "disease_name": pl.Series([], dtype=pl.String),
        "drug_id": pl.Series([], dtype=pl.String),
        "drug_name": pl.Series([], dtype=pl.String),
        "path_family": pl.Series([], dtype=pl.String),
        "hop_count": pl.Series([], dtype=pl.Int32),
        "disease_gene_id": pl.Series([], dtype=pl.String),
        "disease_gene_symbol": pl.Series([], dtype=pl.String),
        "drug_target_gene_id": pl.Series([], dtype=pl.String),
        "drug_target_gene_symbol": pl.Series([], dtype=pl.String),
        "drug_gene_relation": pl.Series([], dtype=pl.String),
        "annotation_type": pl.Series([], dtype=pl.String),
        "annotation_id": pl.Series([], dtype=pl.String),
        "annotation_name": pl.Series([], dtype=pl.String),
        "disease_gene_evidence_score": pl.Series([], dtype=pl.Float64),
    })


def _normalise_annotation(frame: pl.DataFrame, annotation_type: str) -> pl.DataFrame:
    return (
        frame.select("gene_id", "annotation_id", "annotation_name")
        .drop_nulls(["gene_id", "annotation_id"])
        .unique()
        .with_columns(pl.lit(annotation_type).alias("annotation_type"))
    )

    
def _first_existing(frame: pl.DataFrame, names: Sequence[str]) -> str | None:
    return next((name for name in names if name in frame.columns), None)


def _edge_type_list(edge_types: Iterable[str] | str | None) -> list[str] | None:
    """Normalise one edge-table name or an iterable of names.

    Strings are iterable in Python, so accepting them without this guard turns
    ``"drug_gene"`` into individual characters.  Keeping the normalisation in
    one place makes the beginner-facing API forgiving and predictable.
    """
    if edge_types is None:
        return None
    if isinstance(edge_types, str):
        return [edge_types]
    return list(edge_types)


def _flatten_struct_columns(frame: pl.DataFrame) -> pl.DataFrame:
    """Recursively expand Polars Struct columns using stable ``__`` names."""
    result = frame
    while True:
        struct_columns = [
            (name, dtype)
            for name, dtype in result.schema.items()
            if isinstance(dtype, pl.Struct)
        ]
        if not struct_columns:
            return result
        name, dtype = struct_columns[0]
        expressions = [
            pl.col(name).struct.field(field.name).alias(f"{name}__{field.name}")
            for field in dtype.fields
        ]
        result = result.with_columns(expressions).drop(name)


def _csv_safe(frame: pl.DataFrame) -> pl.DataFrame:
    """Convert nested/list columns to readable strings for CSV/BI tools."""
    result = _flatten_struct_columns(frame)
    expressions = []
    for name, dtype in result.schema.items():
        if isinstance(dtype, pl.List):
            expressions.append(
                pl.col(name)
                .map_elements(
                    lambda value: json.dumps(value.to_list() if hasattr(value, "to_list") else value),
                    return_dtype=pl.String,
                )
                .alias(name)
            )
    return result.with_columns(expressions) if expressions else result


@dataclass(frozen=True)
class PathSelection:
    """Representative paths and an explicit record of what selection removed."""

    paths: "PathSet"
    audit: pl.DataFrame


@dataclass(frozen=True)
class PathSet:
    """A collection of explicit, typed paths for one drug-disease pair."""

    data: pl.DataFrame
    requested_families: tuple[str, ...]
    total_before_limit: int
    truncated: bool = False

    def __len__(self) -> int:
        return self.data.height

    def summary(self) -> pl.DataFrame:
        if self.data.is_empty():
            return pl.DataFrame({
                "path_family": [], "available_paths": [], "drug_targets": [],
                "disease_genes": [], "annotation_contexts": [],
            })
        return (
            self.data.group_by("path_family")
            .agg(
                pl.len().alias("available_paths"),
                pl.col("drug_target_gene_id").n_unique().alias("drug_targets"),
                pl.col("disease_gene_id").n_unique().alias("disease_genes"),
                pl.col("annotation_id").drop_nulls().n_unique().alias("annotation_contexts"),
            )
            .sort("path_family")
        )

    def status(self) -> pl.DataFrame:
        return pl.DataFrame({
            "available_paths": [self.data.height],
            "total_before_limit": [self.total_before_limit],
            "truncated": [self.truncated],
            "requested_families": [list(self.requested_families)],
        })

    def readable(self, n: int | None = 20) -> pl.DataFrame:
        readable = self.data.with_columns(
            pl.when(pl.col("path_family") == "Direct convergence")
            .then(pl.concat_str([
                pl.col("disease_name"), pl.lit(" -> "),
                pl.col("disease_gene_symbol"), pl.lit(" <- "),
                pl.col("drug_gene_relation"), pl.lit(" <- "),
                pl.col("drug_name"),
            ]))
            .otherwise(pl.concat_str([
                pl.col("disease_name"), pl.lit(" -> "),
                pl.col("disease_gene_symbol"), pl.lit(" -> "),
                pl.col("annotation_name"), pl.lit(" <- "),
                pl.col("drug_target_gene_symbol"), pl.lit(" <- "),
                pl.col("drug_gene_relation"), pl.lit(" <- "),
                pl.col("drug_name"),
            ]))
            .alias("readable_path")
        )
        columns = [
            "path_family", "readable_path", "disease_gene_evidence_score",
            "annotation_id", "drug_target_gene_id", "drug_gene_relation",
        ]
        result = readable.select(columns)
        return result if n is None else result.head(n)

    def context_redundancy(self, n: int = 20) -> pl.DataFrame:
        if self.data.is_empty():
            return pl.DataFrame()
        return (
            self.data.filter(pl.col("annotation_id").is_not_null())
            .group_by("path_family", "annotation_id", "annotation_name")
            .agg(
                pl.len().alias("paths"),
                pl.col("disease_gene_id").n_unique().alias("disease_genes"),
                pl.col("drug_target_gene_id").n_unique().alias("drug_targets"),
            )
            .with_columns(
                (pl.col("paths") / pl.col("drug_targets").clip(lower_bound=1))
                .round(2)
                .alias("paths_per_target")
            )
            .sort(["paths", "annotation_name"], descending=[True, False])
            .head(n)
        )

    def representatives(self) -> PathSelection:
        direct = self.data.filter(pl.col("path_family") == "Direct convergence")
        shared = (
            self.data.filter(pl.col("path_family") != "Direct convergence")
            .sort(
                ["disease_gene_evidence_score", "disease_gene_id"],
                descending=[True, False],
                nulls_last=True,
            )
            .unique(
                ["path_family", "annotation_id", "drug_target_gene_id", "drug_gene_relation"],
                keep="first",
                maintain_order=True,
            )
        )
        selected = pl.concat([direct, shared], how="diagonal_relaxed").sort(
            ["hop_count", "path_family", "disease_gene_evidence_score"],
            descending=[False, False, True],
            nulls_last=True,
        )
        original_contexts = self.data.get_column("annotation_id").drop_nulls().n_unique()
        selected_contexts = selected.get_column("annotation_id").drop_nulls().n_unique()
        audit = pl.DataFrame({
            "measure": [
                "available paths", "representative paths", "path retention pct",
                "original drug targets", "retained drug targets",
                "original disease genes", "retained disease genes",
                "original annotation contexts", "retained annotation contexts",
            ],
            "value": pl.Series([
                self.data.height,
                selected.height,
                round(100 * selected.height / max(self.data.height, 1), 2),
                self.data.get_column("drug_target_gene_id").n_unique(),
                selected.get_column("drug_target_gene_id").n_unique(),
                self.data.get_column("disease_gene_id").n_unique(),
                selected.get_column("disease_gene_id").n_unique(),
                original_contexts,
                selected_contexts,
            ], dtype=pl.Float64),
        })
        return PathSelection(
            paths=PathSet(
                data=selected,
                requested_families=self.requested_families,
                total_before_limit=selected.height,
                truncated=self.truncated,
            ),
            audit=audit,
        )

    def plot_family_counts(self):
        """Return a Plotly bar chart. Plotly is imported only when requested."""
        try:
            import plotly.express as px
        except ImportError as exc:
            raise ImportError("Install plotly to use plot_family_counts().") from exc
        frame = self.summary().to_pandas()
        return px.bar(
            frame,
            x="path_family",
            y="available_paths",
            text_auto=",.0f",
            title="Explicit paths by family",
            labels={"path_family": "Path family", "available_paths": "Paths"},
        )

    def plot_paths(self, n: int = 5, layout: str = "small_multiples", seed: int = 7):
        """Draw separate path cards by default; optionally draw one network."""
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError("Install matplotlib to use plot_paths().") from exc

        sample = self.data.head(n)
        if sample.is_empty():
            raise ValueError("There are no paths to plot.")
        if layout not in {"small_multiples", "network"}:
            raise ValueError("layout must be 'small_multiples' or 'network'.")

        if layout == "small_multiples":
            figure, axes = plt.subplots(sample.height, 1, figsize=(14, 2.2 * sample.height))
            if sample.height == 1:
                axes = [axes]
            for axis, row in zip(axes, sample.iter_rows(named=True)):
                if row["path_family"] == "Direct convergence":
                    labels = [row["disease_name"], row["disease_gene_symbol"], row["drug_name"]]
                    edge_labels = ["associated with", row["drug_gene_relation"]]
                    arrows = ["->", "<-"]
                else:
                    labels = [
                        row["disease_name"], row["disease_gene_symbol"], row["annotation_name"],
                        row["drug_target_gene_symbol"], row["drug_name"],
                    ]
                    edge_labels = [
                        "associated with", "annotated to", "annotated to", row["drug_gene_relation"]
                    ]
                    arrows = ["->", "->", "<-", "<-"]
                for index, label in enumerate(labels):
                    x = index / max(len(labels) - 1, 1)
                    axis.text(
                        x, 0.5, label or "unknown", ha="center", va="center", fontsize=9,
                        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#dbeafe", "edgecolor": "#60a5fa"},
                        transform=axis.transAxes,
                    )
                    if index < len(edge_labels):
                        next_x = (index + 1) / max(len(labels) - 1, 1)
                        axis.annotate(
                            "", xy=(next_x - 0.04, 0.5), xytext=(x + 0.04, 0.5),
                            xycoords="axes fraction", textcoords="axes fraction",
                            arrowprops={"arrowstyle": arrows[index], "color": "#64748b"},
                        )
                        axis.text(
                            (x + next_x) / 2, 0.68, edge_labels[index], ha="center",
                            fontsize=7, color="#475569", transform=axis.transAxes,
                        )
                axis.set_title(row["path_family"], loc="left", fontsize=9)
                axis.axis("off")
            figure.tight_layout()
            return figure

        try:
            import networkx as nx
        except ImportError as exc:
            raise ImportError("Install networkx to use layout='network'.") from exc
        graph = nx.DiGraph()
        for row in sample.iter_rows(named=True):
            disease = row["disease_name"]
            drug = row["drug_name"]
            dgene = row["disease_gene_symbol"]
            target = row["drug_target_gene_symbol"]
            graph.add_edge(disease, dgene, label="associated gene")
            graph.add_edge(drug, target, label=row["drug_gene_relation"])
            if row["path_family"] != "Direct convergence":
                context = row["annotation_name"]
                graph.add_edge(dgene, context, label="annotated to")
                graph.add_edge(target, context, label="annotated to")
        figure, axis = plt.subplots(figsize=(12, 7))
        positions = nx.spring_layout(graph, seed=seed)
        nx.draw_networkx(
            graph, positions, ax=axis, node_size=1800, font_size=8,
            arrows=True, edge_color="#718096", node_color="#dbeafe",
        )
        nx.draw_networkx_edge_labels(
            graph, positions, edge_labels=nx.get_edge_attributes(graph, "label"),
            font_size=7, ax=axis,
        )
        axis.set_title(f"First {min(n, sample.height)} representative paths")
        axis.axis("off")
        figure.tight_layout()
        return figure


@dataclass(frozen=True)
class InteractionProjection:
    """A documented undirected gene-interaction projection."""

    nodes: pl.DataFrame
    edges: pl.DataFrame
    disease_code: str
    drug_id: str | None
    include_one_hop: bool

    def summary(self) -> pl.DataFrame:
        graph = self.to_networkx()
        import networkx as nx
        component_sizes = [len(component) for component in nx.connected_components(graph)]
        return pl.DataFrame({
            "measure": [
                "projection genes", "projection interactions",
                "connected components", "largest component genes",
            ],
            "value": [
                graph.number_of_nodes(), graph.number_of_edges(),
                len(component_sizes), max(component_sizes, default=0),
            ],
        })

    def to_networkx(self):
        try:
            import networkx as nx
        except ImportError as exc:
            raise ImportError("Install networkx to materialise the interaction graph.") from exc
        graph = nx.Graph()
        for row in self.nodes.iter_rows(named=True):
            graph.add_node(row["gene_id"], **row)
        for row in self.edges.iter_rows(named=True):
            graph.add_edge(row["gene_a"], row["gene_b"], **row)
        return graph

    def as_projection(self):
        """Expose the interaction network through the common projection API."""
        from .concepts import GraphProjection
        node_rows = self.nodes.select(
            pl.lit("gene").alias("node_type"),
            pl.col("gene_id").alias("node_id"),
            pl.coalesce([pl.col("gene_symbol"), pl.col("gene_id")]).alias("name"),
        )
        relation = (
            pl.col("relation").cast(pl.String)
            if "relation" in self.edges.columns else pl.lit("INTERACTS_WITH")
        )
        edge_rows = self.edges.select(
            pl.lit("gene").alias("source_type"),
            pl.col("gene_a").alias("source_id"),
            pl.col("gene_a").alias("source_name"),
            relation.alias("relationship"),
            pl.lit("gene").alias("target_type"),
            pl.col("gene_b").alias("target_id"),
            pl.col("gene_b").alias("target_name"),
            pl.lit("gene_gene").alias("source_table"),
            pl.lit(False).alias("derived"),
        )
        return GraphProjection(
            nodes=node_rows, edges=edge_rows, directed=False,
            specification={
                "projection": "gene_interaction", "disease_code": self.disease_code,
                "drug_id": self.drug_id, "include_one_hop": self.include_one_hop,
            },
        )

    def components(self) -> pl.DataFrame:
        return self.as_projection().components()

    def centrality(self, metric: str = "degree", top: int | None = 30, approximate: bool = False) -> pl.DataFrame:
        return self.as_projection().centrality(metric=metric, top=top, approximate=approximate)

    def hub_report(self, top: int = 20) -> pl.DataFrame:
        return self.as_projection().hub_report(top=top)

    def degree_distribution(self) -> pl.DataFrame:
        return self.as_projection().degree_distribution()

    def provenance(self) -> pl.DataFrame:
        return self.as_projection().provenance()

    def warnings(self) -> list[str]:
        return [
            "Centrality describes the selected undirected interaction projection, not biological importance.",
            "Gene interactions do not establish therapeutic direction, tissue context or clinical effect.",
        ]

    def plot(self, max_nodes: int = 100, seed: int = 7):
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
        except ImportError as exc:
            raise ImportError("Install matplotlib and networkx to plot the projection.") from exc
        graph = self.to_networkx()
        if graph.number_of_nodes() > max_nodes:
            ranked = sorted(graph.degree, key=lambda item: (-item[1], str(item[0])))[:max_nodes]
            graph = graph.subgraph([node for node, _ in ranked]).copy()
        positions = nx.spring_layout(graph, seed=seed)
        colors = []
        for node in graph.nodes:
            attrs = graph.nodes[node]
            if attrs.get("is_disease_gene") and attrs.get("is_drug_target"):
                colors.append("#8b5cf6")
            elif attrs.get("is_disease_gene"):
                colors.append("#ef4444")
            elif attrs.get("is_drug_target"):
                colors.append("#2563eb")
            else:
                colors.append("#cbd5e1")
        figure, axis = plt.subplots(figsize=(12, 8))
        nx.draw_networkx(
            graph, positions, ax=axis, node_color=colors, node_size=170,
            width=0.7, edge_color="#94a3b8", with_labels=False,
        )
        axis.set_title(f"Gene-interaction projection ({graph.number_of_nodes()} displayed genes)")
        axis.axis("off")
        figure.tight_layout()
        return figure


class OptimusExplorer:
    """Entry point for exploring a versioned OptimusKG extraction snapshot."""

    def __init__(self, snapshot_dir: str | Path, files: dict[str, str] | None = None):
        self.snapshot_dir = Path(snapshot_dir).expanduser().resolve()
        self.files = dict(DEFAULT_FILES if files is None else files)
        missing = [name for name in self.files.values() if not (self.snapshot_dir / name).exists()]
        if missing:
            raise FileNotFoundError(
                "Missing extraction outputs:\n- " + "\n- ".join(missing)
                + f"\nSnapshot directory: {self.snapshot_dir}"
            )
        self.tables = {
            name: pl.read_parquet(self.snapshot_dir / filename)
            for name, filename in self.files.items()
        }
        self._validate()
        self.annotations = pl.concat([
            _normalise_annotation(self.tables["process"], "Biological process"),
            _normalise_annotation(self.tables["pathway"], "Pathway"),
        ], how="diagonal_relaxed")
        self._node_catalog_cache = None
        self._edge_catalog_cache = None

    @classmethod
    def from_snapshot(
        cls, snapshot_dir: str | Path, files: dict[str, str] | None = None
    ) -> "OptimusExplorer":
        """Open the original, project-specific audited extraction."""
        return cls(snapshot_dir, files=files)

    @classmethod
    def from_optimuskg(
        cls,
        *,
        cache_dir: str | Path | None = None,
        doi: str | None = None,
        server: str | None = None,
        client: Any | None = None,
    ) -> "ProgressiveExplorer":
        """Connect to the published graph without downloading any graph table.

        This corrects the misleading v0.2 behaviour where ``from_optimuskg``
        accepted a local snapshot directory. Use ``OptimusExplorer(path)`` for
        the original project-snapshot workflow.
        """
        return ProgressiveExplorer.connect(
            cache_dir=cache_dir, doi=doi, server=server, client=client
        )

    def _validate(self) -> None:
        for table_name, required in REQUIRED_COLUMNS.items():
            missing = required - set(self.tables[table_name].columns)
            if missing:
                raise ValueError(f"{table_name} is missing columns: {sorted(missing)}")

    def inventory(self) -> pl.DataFrame:
        return pl.DataFrame({
            "table": list(self.tables),
            "rows": [self.tables[name].height for name in self.tables],
            "columns": [len(self.tables[name].columns) for name in self.tables],
        })

    def schema(self) -> pl.DataFrame:
        rows = []
        for table_name, frame in self.tables.items():
            for column_name, dtype in frame.schema.items():
                rows.append({
                    "table": table_name,
                    "column": column_name,
                    "dtype": str(dtype),
                })
        return pl.DataFrame(rows).sort("table", "column")

    def table_schema(self) -> pl.DataFrame:
        """Physical columns and dtypes, distinct from the graph metagraph."""
        return self.schema()

    def node_catalog(self) -> pl.DataFrame:
        from .concepts import build_node_catalog
        if self._node_catalog_cache is None:
            self._node_catalog_cache = build_node_catalog(self)
        return self._node_catalog_cache

    def edge_catalog(self) -> pl.DataFrame:
        from .concepts import build_edge_catalog
        if self._edge_catalog_cache is None:
            self._edge_catalog_cache = build_edge_catalog(self)
        return self._edge_catalog_cache

    def metagraph(self, include_derived: bool = True) -> pl.DataFrame:
        """Summarise type-to-type transitions and their observed counts."""
        edges = self.edge_catalog()
        if not include_derived:
            edges = edges.filter(~pl.col("derived"))
        return (
            edges.group_by(
                "source_type", "relationship", "target_type", "source_table", "derived"
            )
            .agg(
                pl.len().alias("typed_edges"),
                pl.col("source_id").n_unique().alias("source_nodes"),
                pl.col("target_id").n_unique().alias("target_nodes"),
            )
            .with_columns(pl.lit(True).alias("directed_as_recorded"))
            .sort("source_type", "target_type", "relationship")
        )

    def plot_metagraph(self, include_derived: bool = True):
        """Plot the compact graph schema rather than the instance graph."""
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
        except ImportError as exc:
            raise ImportError("Install matplotlib and networkx to plot the metagraph.") from exc
        frame = self.metagraph(include_derived=include_derived)
        graph = nx.MultiDiGraph()
        for row in frame.iter_rows(named=True):
            graph.add_edge(
                row["source_type"], row["target_type"],
                label=f"{row['relationship']}\n{row['typed_edges']:,}",
                derived=row["derived"],
            )
        positions = {
            "disease": (-1.0, 0.8), "drug": (-1.0, -0.8), "gene": (0.0, 0.0),
            "biological_process": (1.0, 0.7), "pathway": (1.0, -0.7),
        }
        figure, axis = plt.subplots(figsize=(13, 7))
        nx.draw_networkx_nodes(
            graph, positions, ax=axis, node_size=3000, node_color="#dbeafe",
            edgecolors="#2563eb",
        )
        nx.draw_networkx_labels(graph, positions, ax=axis, font_size=9)
        for index, (source, target, _, attrs) in enumerate(graph.edges(keys=True, data=True)):
            radius = 0.12 * ((index % 3) - 1)
            nx.draw_networkx_edges(
                graph, positions, edgelist=[(source, target)], ax=axis,
                connectionstyle=f"arc3,rad={radius}",
                style="dashed" if attrs["derived"] else "solid",
                arrows=True, arrowstyle="-|>", edge_color="#64748b",
            )
            midpoint = (
                (positions[source][0] + positions[target][0]) / 2,
                (positions[source][1] + positions[target][1]) / 2 + radius,
            )
            axis.text(
                *midpoint, attrs["label"], fontsize=6.5, ha="center", va="center",
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
            )
        axis.set_title("Optimus Explorer metagraph · dashed edges are analytical/derived")
        axis.axis("off")
        figure.tight_layout()
        return figure

    def node(self, node_type: str, node: str):
        """Resolve one typed node by identifier, name, symbol or disease code."""
        from .concepts import NodeView, _normalise_type
        node_type = _normalise_type(node_type)
        value = node.strip().lower()
        if not value:
            raise ValueError("node must contain at least one non-space character.")
        candidates = self.node_catalog().filter(pl.col("node_type") == node_type)
        exact = candidates.filter(
            (pl.col("node_id").str.to_lowercase() == value)
            | (pl.col("name").fill_null("").str.to_lowercase() == value)
            | (pl.col("alias").fill_null("").str.to_lowercase() == value)
        ).unique(["node_type", "node_id"])
        if exact.is_empty():
            exact = candidates.filter(
                pl.col("name").fill_null("").str.to_lowercase().str.contains(value, literal=True)
                | pl.col("node_id").str.to_lowercase().str.contains(value, literal=True)
            ).unique(["node_type", "node_id"])
        if exact.height != 1:
            choices = exact.select("node_id", "name", "alias").sort("name").head(10).to_dicts()
            raise ValueError(
                f"Expected one {node_type} for {node!r}; found {exact.height}. Candidates: {choices}"
            )
        row = exact.row(0, named=True)
        return NodeView(self, node_type, row["node_id"], row["name"] or row["node_id"])

    def project(
        self, node_types=None, relations=None, seeds=None, radius=None,
        include_derived: bool = False, directed: bool = False, max_nodes: int = 5000,
    ):
        """Build an explicit, documented projection for topology analysis."""
        from .concepts import build_projection
        return build_projection(
            self, node_types=node_types, relations=relations, seeds=seeds,
            radius=radius, include_derived=include_derived,
            directed=directed, max_nodes=max_nodes,
        )

    def counts_explained(self, disease: str) -> pl.DataFrame:
        """Define the source and denominator behind disease-centred headline counts."""
        view = self.disease(disease)
        disease_genes = view.associated_genes()
        pairs = view.related_drugs(limit=None)
        rows = [
            {
                "measure": "associated genes", "value": disease_genes["gene_id"].n_unique(),
                "source_table": "disease_gene", "filter": f"disease_code == {view.code}",
                "unit": "distinct gene_id",
                "interpretation": "Genes recorded in the extracted disease-gene evidence table.",
            },
            {
                "measure": "candidate drugs", "value": pairs["drug_id"].n_unique(),
                "source_table": "pair_summary", "filter": f"disease_code == {view.code}",
                "unit": "distinct drug_id",
                "interpretation": "Drugs retained by candidate-pair rules; not approved treatments.",
            },
        ]
        for flag, label in [
            ("has_recorded_indication", "recorded indications"),
            ("has_recorded_off_label_use", "recorded off-label uses"),
            ("has_recorded_contraindication", "recorded contraindications"),
        ]:
            if flag in pairs.columns:
                rows.append({
                    "measure": label,
                    "value": pairs.filter(pl.col(flag).fill_null(False))["drug_id"].n_unique(),
                    "source_table": "pair_summary",
                    "filter": f"disease_code == {view.code} and {flag}",
                    "unit": "distinct drug_id",
                    "interpretation": "Recorded relation after harmonisation; inspect provenance before clinical interpretation.",
                })
        return pl.DataFrame(rows)

    def path_families(self) -> pl.DataFrame:
        return pl.DataFrame([
            {"path_family": key, **definition}
            for key, definition in PATH_FAMILIES.items()
        ])

    def search_nodes(self, query: str, node_type: str | None = None, limit: int = 30) -> pl.DataFrame:
        query = query.strip()
        if not query:
            raise ValueError("query must contain at least one non-space character.")
        from .concepts import _normalise_type
        nodes = self.node_catalog()
        if node_type is not None:
            nodes = nodes.filter(pl.col("node_type") == _normalise_type(node_type))
        lowered = query.lower()
        return (
            nodes.filter(
                pl.col("name").str.to_lowercase().str.contains(lowered, literal=True)
                | pl.col("node_id").str.to_lowercase().str.contains(lowered, literal=True)
                | pl.col("alias").fill_null("").str.to_lowercase().str.contains(lowered, literal=True)
            )
            .sort("node_type", "name", "node_id")
            .select("node_type", "node_id", "name", "alias")
            .unique()
            .head(limit)
        )

    def disease(self, disease: str) -> "DiseaseView":
        candidates = self.tables["pair_summary"].select(
            "disease_code", "disease_id", "disease_name"
        ).unique()
        lowered = disease.strip().lower()
        exact = candidates.filter(
            (pl.col("disease_code").str.to_lowercase() == lowered)
            | (pl.col("disease_id").str.to_lowercase() == lowered)
            | (pl.col("disease_name").str.to_lowercase() == lowered)
        )
        if exact.is_empty():
            exact = candidates.filter(
                pl.col("disease_name").str.to_lowercase().str.contains(lowered, literal=True)
            )
        if exact.height != 1:
            choices = exact.sort("disease_name").head(10).to_dicts()
            raise ValueError(
                f"Expected one disease for {disease!r}; found {exact.height}. "
                f"Candidates: {choices}"
            )
        row = exact.row(0, named=True)
        return DiseaseView(self, row["disease_code"], row["disease_id"], row["disease_name"])


class DiseaseView:
    """Disease-centred view with a small set of exploratory operations."""

    def __init__(self, explorer: OptimusExplorer, code: str, disease_id: str, name: str):
        self.explorer = explorer
        self.code = code
        self.id = disease_id
        self.name = name

    def info(self) -> pl.DataFrame:
        return pl.DataFrame({
            "disease_code": [self.code],
            "disease_id": [self.id],
            "disease_name": [self.name],
        })

    def as_node(self):
        """Return this disease through the generic typed-node interface."""
        return self.explorer.node("disease", self.id)

    def summary(self) -> pl.DataFrame:
        disease_genes = self.associated_genes()
        pairs = self.related_drugs(limit=None)
        indications = (
            pairs.filter(pl.col("has_recorded_indication").fill_null(False)).height
            if "has_recorded_indication" in pairs.columns else 0
        )
        off_label = (
            pairs.filter(pl.col("has_recorded_off_label_use").fill_null(False)).height
            if "has_recorded_off_label_use" in pairs.columns else 0
        )
        contraindications = (
            pairs.filter(pl.col("has_recorded_contraindication").fill_null(False)).height
            if "has_recorded_contraindication" in pairs.columns else 0
        )
        return pl.DataFrame({
            "measure": [
                "associated genes", "candidate drugs", "recorded indications",
                "recorded off-label uses", "recorded contraindications",
            ],
            "value": [
                disease_genes.get_column("gene_id").n_unique(),
                pairs.get_column("drug_id").n_unique(), indications, off_label,
                contraindications,
            ],
        })

    def associated_genes(self, min_score: float = 0.0, limit: int | None = None) -> pl.DataFrame:
        frame = (
            self.explorer.tables["disease_gene"]
            .filter(
                (pl.col("disease_code") == self.code)
                & (pl.col("evidence_score").fill_null(0) >= min_score)
            )
            .sort(["evidence_score", "gene_symbol"], descending=[True, False], nulls_last=True)
        )
        return frame if limit is None else frame.head(limit)

    def related_drugs(self, limit: int | None = 30) -> pl.DataFrame:
        frame = self.explorer.tables["pair_summary"].filter(pl.col("disease_code") == self.code)
        sort_columns = [
            col for col in ["has_recorded_indication", "shared_target_count", "drug_name"]
            if col in frame.columns
        ]
        if sort_columns:
            descending = [True if col != "drug_name" else False for col in sort_columns]
            frame = frame.sort(sort_columns, descending=descending, nulls_last=True)
        return frame if limit is None else frame.head(limit)

    def annotation_summary(self) -> pl.DataFrame:
        genes = self.associated_genes().select("gene_id").unique()
        joined = genes.join(self.explorer.annotations, on="gene_id", how="inner")
        return (
            joined.group_by("annotation_type")
            .agg(
                pl.col("gene_id").n_unique().alias("disease_genes"),
                pl.col("annotation_id").n_unique().alias("annotations"),
                pl.len().alias("gene_annotation_rows"),
            )
            .sort("annotation_type")
        )

    def _resolve_drug(self, drug: str) -> dict:
        candidates = self.related_drugs(limit=None).select("drug_id", "drug_name").unique()
        lowered = drug.strip().lower()
        exact = candidates.filter(
            (pl.col("drug_id").str.to_lowercase() == lowered)
            | (pl.col("drug_name").str.to_lowercase() == lowered)
        )
        if exact.is_empty():
            exact = candidates.filter(
                pl.col("drug_name").str.to_lowercase().str.contains(lowered, literal=True)
            )
        if exact.height != 1:
            choices = exact.sort("drug_name").head(10).to_dicts()
            raise ValueError(
                f"Expected one drug for {drug!r}; found {exact.height}. Candidates: {choices}"
            )
        return exact.row(0, named=True)

    def paths_to_drug(
        self,
        drug: str,
        families: Iterable[str] = (
            "direct_convergence", "shared_pathway", "shared_process"
        ),
        max_paths: int | None = None,
    ) -> PathSet:
        requested = tuple(families)
        unknown = sorted(set(requested) - set(PATH_FAMILIES))
        if unknown:
            raise ValueError(f"Unknown path families: {unknown}. Use explorer.path_families().")
        unsupported = sorted(set(requested) - {
            "direct_convergence", "shared_pathway", "shared_process"
        })
        if unsupported:
            raise ValueError(
                f"These families require a separate projection: {unsupported}. "
                "Use gene_interaction_projection() for gene interactions."
            )
        drug_row = self._resolve_drug(drug)
        drug_id = drug_row["drug_id"]

        dg = (
            self.explorer.tables["disease_gene"]
            .filter(pl.col("disease_code") == self.code)
            .select(
                "disease_code", "disease_id", "disease_name",
                pl.col("gene_id").alias("disease_gene_id"),
                pl.col("gene_symbol").alias("disease_gene_symbol"),
                pl.col("evidence_score").alias("disease_gene_evidence_score"),
            )
            .unique(["disease_id", "disease_gene_id"])
        )
        tg = (
            self.explorer.tables["drug_gene"]
            .filter(pl.col("drug_id") == drug_id)
            .select(
                "drug_id", "drug_name",
                pl.col("gene_id").alias("drug_target_gene_id"),
                pl.col("gene_symbol").alias("drug_target_gene_symbol"),
                "drug_gene_relation",
            )
            .unique(["drug_id", "drug_target_gene_id", "drug_gene_relation"])
        )
        if dg.is_empty() or tg.is_empty():
            raise ValueError("The selected pair has an empty disease or drug side.")

        parts: list[pl.DataFrame] = []
        if "direct_convergence" in requested:
            direct = (
                dg.join(
                    tg, left_on="disease_gene_id", right_on="drug_target_gene_id",
                    how="inner", validate="m:m",
                )
                .with_columns(
                    pl.col("disease_gene_id").alias("drug_target_gene_id"),
                    pl.col("disease_gene_symbol").alias("drug_target_gene_symbol"),
                    pl.lit("Direct convergence").alias("path_family"),
                    pl.lit(2, dtype=pl.Int32).alias("hop_count"),
                    pl.lit(None, dtype=pl.String).alias("annotation_type"),
                    pl.lit(None, dtype=pl.String).alias("annotation_id"),
                    pl.lit(None, dtype=pl.String).alias("annotation_name"),
                )
                .select(PATH_COLUMNS)
            )
            parts.append(direct)

        requested_annotation_types = []
        if "shared_process" in requested:
            requested_annotation_types.append("Biological process")
        if "shared_pathway" in requested:
            requested_annotation_types.append("Pathway")
        if requested_annotation_types:
            annotations = self.explorer.annotations.filter(
                pl.col("annotation_type").is_in(requested_annotation_types)
            )
            dg_ann = dg.join(
                annotations, left_on="disease_gene_id", right_on="gene_id",
                how="inner", validate="m:m",
            ).unique()
            tg_ann = tg.join(
                annotations, left_on="drug_target_gene_id", right_on="gene_id",
                how="inner", validate="m:m",
            ).unique()
            shared = (
                dg_ann.join(
                    tg_ann,
                    on=["annotation_type", "annotation_id", "annotation_name"],
                    how="inner", validate="m:m",
                )
                .with_columns(
                    pl.when(pl.col("annotation_type") == "Biological process")
                    .then(pl.lit("Shared process"))
                    .otherwise(pl.lit("Shared pathway"))
                    .alias("path_family"),
                    pl.lit(4, dtype=pl.Int32).alias("hop_count"),
                )
                .select(PATH_COLUMNS)
            )
            parts.append(shared)

        paths = pl.concat(parts, how="diagonal_relaxed") if parts else _empty_paths()
        paths = (
            paths.unique([
                "disease_id", "drug_id", "path_family", "disease_gene_id",
                "drug_target_gene_id", "annotation_id", "drug_gene_relation",
            ])
            .sort(
                ["hop_count", "path_family", "disease_gene_evidence_score", "annotation_name"],
                descending=[False, False, True, False],
                nulls_last=True,
            )
        )
        total = paths.height
        truncated = max_paths is not None and total > max_paths
        if truncated:
            paths = paths.head(max_paths)
        return PathSet(paths, requested, total_before_limit=total, truncated=truncated)

    def verify_direct_paths(self, drug: str, paths: PathSet | None = None) -> pl.DataFrame:
        drug_row = self._resolve_drug(drug)
        drug_id = drug_row["drug_id"]
        if paths is None:
            paths = self.paths_to_drug(drug_id, families=["direct_convergence"])
        expected = (
            self.explorer.tables["direct_paths"]
            .filter(
                (pl.col("disease_code") == self.code)
                & (pl.col("drug_id") == drug_id)
            )
            .select(
                "disease_id", "drug_id", pl.col("gene_id").alias("gene_id"),
                "drug_gene_relation",
            )
            .unique()
        )
        observed = (
            paths.data.filter(pl.col("path_family") == "Direct convergence")
            .select(
                "disease_id", "drug_id",
                pl.col("disease_gene_id").alias("gene_id"), "drug_gene_relation",
            )
            .unique()
        )
        keys = ["disease_id", "drug_id", "gene_id", "drug_gene_relation"]
        missing = expected.join(observed, on=keys, how="anti")
        unexpected = observed.join(expected, on=keys, how="anti")
        return pl.DataFrame({
            "check": ["expected", "reconstructed", "missing", "unexpected"],
            "rows": [expected.height, observed.height, missing.height, unexpected.height],
        })

    def gene_interaction_projection(
        self,
        gene_gene_path: str | Path,
        drug: str | None = None,
        include_one_hop: bool = False,
    ) -> InteractionProjection:
        source = Path(gene_gene_path).expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(source)
        disease_frame = self.associated_genes().select("gene_id", "gene_symbol").unique()
        disease_ids = set(disease_frame.get_column("gene_id").to_list())
        target_ids: set[str] = set()
        drug_id = None
        target_frame = pl.DataFrame({
            "gene_id": pl.Series([], dtype=pl.String),
            "gene_symbol": pl.Series([], dtype=pl.String),
        })
        if drug is not None:
            drug_row = self._resolve_drug(drug)
            drug_id = drug_row["drug_id"]
            target_frame = (
                self.explorer.tables["drug_gene"]
                .filter(pl.col("drug_id") == drug_id)
                .select("gene_id", "gene_symbol")
                .unique()
            )
            target_ids = set(target_frame.get_column("gene_id").to_list())
        seeds = sorted(disease_ids | target_ids)
        if not seeds:
            raise ValueError("No seed genes were available for the interaction projection.")

        scan = pl.scan_parquet(source)
        columns = set(scan.collect_schema().names())
        if not {"from", "to"}.issubset(columns):
            raise ValueError("The gene-gene file must contain 'from' and 'to' columns.")
        condition = (
            pl.col("from").is_in(seeds) | pl.col("to").is_in(seeds)
            if include_one_hop
            else pl.col("from").is_in(seeds) & pl.col("to").is_in(seeds)
        )
        optional = [name for name in ["label", "relation", "undirected"] if name in columns]
        edges = (
            scan.filter(condition)
            .select(
                pl.col("from").alias("gene_a"),
                pl.col("to").alias("gene_b"),
                *optional,
            )
            .filter(pl.col("gene_a") != pl.col("gene_b"))
            .collect()
        )
        if "undirected" in edges.columns and edges.filter(
            pl.col("undirected") == False  # noqa: E712 - intentional Polars expression
        ).height:
            raise ValueError(
                "This experimental projection only supports relationships marked undirected."
            )
        if "relation" in edges.columns and edges.filter(
            pl.col("relation") != "INTERACTS_WITH"
        ).height:
            raise ValueError(
                "Unexpected gene-gene relation found; inspect it before combining relationship types."
            )
        # The current interaction projection is explicitly undirected. Normalise
        # endpoint order so reverse duplicates do not inflate edge counts.
        edges = (
            edges.with_columns(
                pl.when(pl.col("gene_a") <= pl.col("gene_b"))
                .then(pl.col("gene_a"))
                .otherwise(pl.col("gene_b"))
                .alias("_gene_a"),
                pl.when(pl.col("gene_a") <= pl.col("gene_b"))
                .then(pl.col("gene_b"))
                .otherwise(pl.col("gene_a"))
                .alias("_gene_b"),
            )
            .drop("gene_a", "gene_b")
            .rename({"_gene_a": "gene_a", "_gene_b": "gene_b"})
            .select("gene_a", "gene_b", *optional)
            .unique(["gene_a", "gene_b", *optional])
        )
        node_ids = sorted(set(edges.get_column("gene_a").to_list()) | set(edges.get_column("gene_b").to_list()))
        symbol_rows = pl.concat([disease_frame, target_frame], how="vertical_relaxed").unique("gene_id")
        symbol_map = dict(symbol_rows.iter_rows())
        nodes = pl.DataFrame({
            "gene_id": node_ids,
            "gene_symbol": [symbol_map.get(gene_id) for gene_id in node_ids],
            "is_disease_gene": [gene_id in disease_ids for gene_id in node_ids],
            "is_drug_target": [gene_id in target_ids for gene_id in node_ids],
        })
        return InteractionProjection(nodes, edges, self.code, drug_id, include_one_hop)

    def gene_interaction_connections(
        self,
        drug: str,
        gene_gene_path: str | Path,
    ) -> pl.DataFrame:
        drug_row = self._resolve_drug(drug)
        projection = self.gene_interaction_projection(
            gene_gene_path, drug=drug_row["drug_id"], include_one_hop=False
        )
        disease_symbols = dict(
            self.associated_genes().select("gene_id", "gene_symbol").unique().iter_rows()
        )
        targets = (
            self.explorer.tables["drug_gene"]
            .filter(pl.col("drug_id") == drug_row["drug_id"])
            .select("gene_id", "gene_symbol", "drug_gene_relation")
            .unique()
        )
        target_rows: dict[str, list[dict]] = {}
        for target_row in targets.iter_rows(named=True):
            target_rows.setdefault(target_row["gene_id"], []).append(target_row)
        records = []
        for row in projection.edges.iter_rows(named=True):
            for disease_gene, target_gene in [
                (row["gene_a"], row["gene_b"]),
                (row["gene_b"], row["gene_a"]),
            ]:
                if disease_gene in disease_symbols and target_gene in target_rows:
                    for target in target_rows[target_gene]:
                        records.append({
                            "disease_code": self.code,
                            "disease_name": self.name,
                            "drug_id": drug_row["drug_id"],
                            "drug_name": drug_row["drug_name"],
                            "disease_gene_id": disease_gene,
                            "disease_gene_symbol": disease_symbols[disease_gene],
                            "interacting_target_gene_id": target_gene,
                            "interacting_target_gene_symbol": target["gene_symbol"],
                            "drug_gene_relation": target["drug_gene_relation"],
                            "interaction_relation": row.get("relation", "INTERACTS_WITH"),
                        })
        if not records:
            return pl.DataFrame({
                "disease_code": pl.Series([], dtype=pl.String),
                "disease_name": pl.Series([], dtype=pl.String),
                "drug_id": pl.Series([], dtype=pl.String),
                "drug_name": pl.Series([], dtype=pl.String),
                "disease_gene_id": pl.Series([], dtype=pl.String),
                "disease_gene_symbol": pl.Series([], dtype=pl.String),
                "interacting_target_gene_id": pl.Series([], dtype=pl.String),
                "interacting_target_gene_symbol": pl.Series([], dtype=pl.String),
                "drug_gene_relation": pl.Series([], dtype=pl.String),
                "interaction_relation": pl.Series([], dtype=pl.String),
            })
        return pl.DataFrame(records).unique().sort(
            "disease_gene_symbol", "interacting_target_gene_symbol"
        )


# ---------------------------------------------------------------------------
# Live, progressive OptimusKG exploration
# ---------------------------------------------------------------------------

NODE_TYPE_SPECS = {
    "gene": ("GEN", "nodes/gene.parquet"),
    "disease": ("DIS", "nodes/disease.parquet"),
    "biological_process": ("BPO", "nodes/biological_process.parquet"),
    "phenotype": ("PHE", "nodes/phenotype.parquet"),
    "drug": ("DRG", "nodes/drug.parquet"),
    "anatomy": ("ANA", "nodes/anatomy.parquet"),
    "molecular_function": ("MFN", "nodes/molecular_function.parquet"),
    "cellular_component": ("CCO", "nodes/cellular_component.parquet"),
    "pathway": ("PWY", "nodes/pathway.parquet"),
    "exposure": ("EXP", "nodes/exposure.parquet"),
}

# The endpoint orientation follows the published edge label. ``undirected``
# remains an edge-level field and is respected when traversing a loaded table.
EDGE_TYPE_SPECS = {
    "disease_gene": ("disease", "gene", "DIS-GEN"),
    "anatomy_gene": ("anatomy", "gene", "ANA-GEN"),
    "drug_drug": ("drug", "drug", "DRG-DRG"),
    "phenotype_gene": ("phenotype", "gene", "PHE-GEN"),
    "gene_gene": ("gene", "gene", "GEN-GEN"),
    "disease_phenotype": ("disease", "phenotype", "DIS-PHE"),
    "biological_process_gene": ("biological_process", "gene", "BPO-GEN"),
    "drug_disease": ("drug", "disease", "DRG-DIS"),
    "molecular_function_gene": ("molecular_function", "gene", "MFN-GEN"),
    "drug_phenotype": ("drug", "phenotype", "DRG-PHE"),
    "pathway_gene": ("pathway", "gene", "PWY-GEN"),
    "biological_process_biological_process": (
        "biological_process", "biological_process", "BPO-BPO"
    ),
    "disease_disease": ("disease", "disease", "DIS-DIS"),
    "cellular_component_gene": ("cellular_component", "gene", "CCO-GEN"),
    "drug_gene": ("drug", "gene", "DRG-GEN"),
    "phenotype_phenotype": ("phenotype", "phenotype", "PHE-PHE"),
    "molecular_function_molecular_function": (
        "molecular_function", "molecular_function", "MFN-MFN"
    ),
    "pathway_pathway": ("pathway", "pathway", "PWY-PWY"),
    "exposure_gene": ("exposure", "gene", "EXP-GEN"),
    "exposure_disease": ("exposure", "disease", "EXP-DIS"),
    "exposure_exposure": ("exposure", "exposure", "EXP-EXP"),
    "exposure_biological_process": (
        "exposure", "biological_process", "EXP-BPO"
    ),
    "anatomy_anatomy": ("anatomy", "anatomy", "ANA-ANA"),
    "cellular_component_cellular_component": (
        "cellular_component", "cellular_component", "CCO-CCO"
    ),
    "exposure_molecular_function": (
        "exposure", "molecular_function", "EXP-MFN"
    ),
    "exposure_cellular_component": (
        "exposure", "cellular_component", "EXP-CCO"
    ),
    "drug_biological_process": ("drug", "biological_process", "DRG-BPO"),
}

TYPE_ALIASES = {
    "genes": "gene", "diseases": "disease", "drugs": "drug",
    "process": "biological_process", "processes": "biological_process",
    "biological process": "biological_process", "biological processes": "biological_process",
    "molecular function": "molecular_function", "cellular component": "cellular_component",
    "pathways": "pathway", "phenotypes": "phenotype", "anatomies": "anatomy",
    "exposures": "exposure",
}


def _live_type(value: str) -> str:
    key = value.strip().lower().replace("-", "_")
    key = TYPE_ALIASES.get(key, key.replace(" ", "_"))
    if key not in NODE_TYPE_SPECS:
        raise ValueError(
            f"Unknown node type {value!r}. Choose from {sorted(NODE_TYPE_SPECS)}."
        )
    return key


def _properties(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("properties")
    return value if isinstance(value, dict) else {}


def _entity_name(row: dict[str, Any]) -> str:
    props = _properties(row)
    for key in (
        "name", "symbol", "preferred_name", "primary_name", "display_name",
        "disease_name", "drug_name", "label",
    ):
        value = row.get(key, props.get(key))
        if value is not None and str(value).strip():
            return str(value)
    return str(row.get("id", ""))


@dataclass(frozen=True)
class DatasetSpec:
    """One documented OptimusKG parquet table."""

    name: str
    kind: str
    path: str
    source_type: str | None = None
    target_type: str | None = None
    label: str | None = None


@dataclass
class LoadPlan:
    """A route through the schema that can be reviewed before downloading."""

    explorer: "ProgressiveExplorer"
    node_types: tuple[str, ...]
    edge_types: tuple[str, ...]

    def show(self) -> pl.DataFrame:
        records = []
        for position, node_type in enumerate(self.node_types):
            records.append({
                "step": 2 * position,
                "kind": "node",
                "dataset": node_type,
                "path": NODE_TYPE_SPECS[node_type][1],
                "loaded": node_type in self.explorer.tables,
            })
            if position < len(self.edge_types):
                edge_type = self.edge_types[position]
                records.append({
                    "step": 2 * position + 1,
                    "kind": "edge",
                    "dataset": edge_type,
                    "path": f"edges/{edge_type}.parquet",
                    "loaded": edge_type in self.explorer.tables,
                })
        return pl.DataFrame(records).sort("step")

    def load(self, *, force: bool = False, activate: bool = True) -> "ProgressiveExplorer":
        """Load only the reviewed tables in this plan."""
        for node_type in self.node_types:
            if node_type not in self.explorer.tables:
                self.explorer.load_node(node_type, force=force, activate=activate)
        for edge_type in self.edge_types:
            if edge_type not in self.explorer.tables:
                self.explorer.load_edge(edge_type, force=force, activate=activate)
        return self.explorer


@dataclass(frozen=True)
class LiveEntity:
    """A resolved entity in one explicitly loaded node table."""

    explorer: "ProgressiveExplorer"
    node_type: str
    id: str
    name: str
    record: dict[str, Any]

    def info(self) -> pl.DataFrame:
        return pl.DataFrame([{
            "node_type": self.node_type,
            "id": self.id,
            "name": self.name,
            **{k: v for k, v in self.record.items() if k != "properties"},
        }])

    def properties(self, *, flatten: bool = True) -> pl.DataFrame:
        """Return this node's property record as a one-row table."""
        frame = pl.DataFrame([self.record])
        return _flatten_struct_columns(frame) if flatten else frame

    def available_edge_types(self) -> pl.DataFrame:
        """List loaded edge tables that can connect to this node type."""
        return self.explorer.available_edge_types(self.node_type)

    def neighbors(
        self,
        *,
        edge_types: Iterable[str] | str | None = None,
        direction: str = "both",
        limit: int | None = 50,
        exclude_self: bool = False,
    ) -> pl.DataFrame:
        return self.explorer.neighbors(
            self, edge_types=edge_types, direction=direction, limit=limit,
            exclude_self=exclude_self,
        )

    def edge_summary(
        self, *, edge_types: Iterable[str] | str | None = None,
        exclude_self: bool = True,
    ) -> pl.DataFrame:
        """Count returned neighbour rows and unique endpoints by edge table."""
        return self.explorer.edge_summary(
            self, edge_types=edge_types, exclude_self=exclude_self
        )

    def edge_records(
        self, *, edge_types: Iterable[str] | str | None = None,
        direction: str = "both", flatten_properties: bool = True,
        exclude_self: bool = False, limit: int | None = 50,
    ) -> pl.DataFrame:
        """Return source edge rows, endpoint names, and optional properties."""
        return self.explorer.edge_records(
            self, edge_types=edge_types, direction=direction,
            flatten_properties=flatten_properties, exclude_self=exclude_self,
            limit=limit,
        )

    def ego(
        self,
        *,
        radius: int = 1,
        edge_types: Iterable[str] | str | None = None,
        max_neighbors_per_hop: int = 50,
        exclude_self: bool = True,
    ) -> "WorkingGraph":
        return self.explorer.subgraph(
            [self], radius=radius, edge_types=edge_types,
            max_neighbors_per_hop=max_neighbors_per_hop,
            exclude_self=exclude_self,
        )

    def export_neighborhood(
        self, output_dir: str | Path, *, radius: int = 1,
        edge_types: Iterable[str] | str | None = None,
        max_neighbors_per_hop: int = 5000,
        exclude_self: bool = True,
        formats: Iterable[str] | str = ("csv", "parquet"),
    ) -> "EvidenceBundle":
        """Build, save, and return a bounded evidence bundle around this entity."""
        bundle = self.explorer.evidence_bundle(
            self, radius=radius, edge_types=edge_types,
            max_neighbors_per_hop=max_neighbors_per_hop,
            exclude_self=exclude_self,
        )
        bundle.write(output_dir, formats=formats)
        return bundle


@dataclass(frozen=True)
class WorkingGraph:
    """A bounded instance graph created from the currently active tables."""

    explorer: "ProgressiveExplorer"
    nodes: pl.DataFrame
    edges: pl.DataFrame
    seeds: tuple[str, ...]
    radius: int
    truncated: bool

    def summary(self) -> pl.DataFrame:
        return pl.DataFrame({
            "measure": ["nodes", "edges", "seed nodes", "radius", "truncated"],
            "value": [
                str(self.nodes.height), str(self.edges.height), str(len(self.seeds)),
                str(self.radius), str(self.truncated),
            ],
        })

    def to_networkx(self, *, directed: bool = True):
        try:
            import networkx as nx
        except ImportError as exc:
            raise ImportError("Install networkx to materialise a working graph.") from exc
        graph = nx.MultiDiGraph() if directed else nx.MultiGraph()
        for row in self.nodes.iter_rows(named=True):
            graph.add_node(row["id"], **row)
        for row in self.edges.iter_rows(named=True):
            graph.add_edge(row["from"], row["to"], **row)
            if directed and row.get("undirected") is True:
                graph.add_edge(row["to"], row["from"], **row)
        return graph

    def paths(
        self, source: LiveEntity | str, target: LiveEntity | str,
        *, max_hops: int = 4, max_paths: int = 20,
    ) -> list[list[str]]:
        """Enumerate bounded simple paths inside this working graph only."""
        import networkx as nx
        source_id = source.id if isinstance(source, LiveEntity) else str(source)
        target_id = target.id if isinstance(target, LiveEntity) else str(target)
        graph = nx.Graph(self.to_networkx(directed=False))
        if source_id not in graph or target_id not in graph:
            return []
        result = []
        for path in nx.all_simple_paths(graph, source_id, target_id, cutoff=max_hops):
            result.append(path)
            if len(result) >= max_paths:
                break
        return result

    def plot(self, *, seed: int = 7, with_edge_labels: bool = True):
        """Plot this bounded instance graph (not the metagraph/schema)."""
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
        except ImportError as exc:
            raise ImportError("Install matplotlib and networkx to plot graphs.") from exc
        # Collapse parallel records only for legible drawing; the underlying
        # ``edges`` table and ``to_networkx`` retain them.
        graph = nx.DiGraph(self.to_networkx(directed=True))
        figure, axis = plt.subplots(figsize=(13, 8))
        positions = nx.spring_layout(graph, seed=seed)
        type_colors = {
            "disease": "#ef4444", "drug": "#2563eb", "gene": "#8b5cf6",
            "pathway": "#16a34a", "biological_process": "#f59e0b",
        }
        colors = [type_colors.get(graph.nodes[n].get("node_type"), "#94a3b8") for n in graph]
        labels = {n: graph.nodes[n].get("name", n) for n in graph}
        nx.draw_networkx(
            graph, positions, labels=labels, node_color=colors, node_size=1500,
            font_size=7, arrows=True, edge_color="#94a3b8", ax=axis,
        )
        if with_edge_labels and graph.number_of_edges() <= 40:
            edge_labels = {
                (u, v): attrs.get("relation", attrs.get("edge_type", ""))
                for u, v, attrs in graph.edges(data=True)
            }
            nx.draw_networkx_edge_labels(
                graph, positions, edge_labels=edge_labels, font_size=6, ax=axis
            )
        axis.set_title("OptimusKG working instance graph")
        axis.axis("off")
        figure.tight_layout()
        return figure


@dataclass(frozen=True)
class EvidenceBundle:
    """Portable, question-scoped graph extract for BI and downstream analysis."""

    entities: pl.DataFrame
    relationships: pl.DataFrame
    evidence: pl.DataFrame
    manifest: dict[str, Any]

    def summary(self) -> pl.DataFrame:
        return pl.DataFrame({
            "table": ["entities", "relationships", "evidence"],
            "rows": [
                self.entities.height,
                self.relationships.height,
                self.evidence.height,
            ],
            "columns": [
                len(self.entities.columns),
                len(self.relationships.columns),
                len(self.evidence.columns),
            ],
        })

    def write(
        self, output_dir: str | Path, *,
        formats: Iterable[str] | str = ("csv", "parquet"),
        database_name: str = "evidence.duckdb",
    ) -> pl.DataFrame:
        """Write normalized tables plus a manifest.

        CSV is flattened for BI tools. Parquet retains data types. DuckDB is
        optional and requires the ``duckdb`` package.
        """
        requested = _edge_type_list(formats) or []
        requested = [item.lower().lstrip(".") for item in requested]
        allowed = {"csv", "parquet", "duckdb"}
        unknown = sorted(set(requested) - allowed)
        if unknown:
            raise ValueError(f"Unknown export formats {unknown}; choose from {sorted(allowed)}.")

        destination = Path(output_dir).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        tables = {
            "entities": self.entities,
            "relationships": self.relationships,
            "evidence": self.evidence,
        }
        written: list[dict[str, Any]] = []

        if "csv" in requested:
            for name, frame in tables.items():
                path = destination / f"{name}.csv"
                _csv_safe(frame).write_csv(path)
                written.append({"format": "csv", "table": name, "path": str(path)})

        if "parquet" in requested:
            for name, frame in tables.items():
                path = destination / f"{name}.parquet"
                frame.write_parquet(path)
                written.append({"format": "parquet", "table": name, "path": str(path)})

        if "duckdb" in requested:
            try:
                import duckdb
            except ImportError as exc:
                raise ImportError(
                    "Install DuckDB to request a database export: pip install duckdb pyarrow"
                ) from exc
            database_path = destination / database_name
            connection = duckdb.connect(str(database_path))
            try:
                for name, frame in tables.items():
                    connection.register(f"_{name}_input", frame.to_arrow())
                    connection.execute(
                        f'CREATE OR REPLACE TABLE "{name}" AS '
                        f'SELECT * FROM "_{name}_input"'
                    )
                    connection.unregister(f"_{name}_input")
            finally:
                connection.close()
            written.append({"format": "duckdb", "table": "all", "path": str(database_path)})

        manifest_path = destination / "manifest.json"
        manifest_path.write_text(json.dumps(self.manifest, indent=2), encoding="utf-8")
        written.append({"format": "json", "table": "manifest", "path": str(manifest_path)})
        return pl.DataFrame(written)


class ProgressiveExplorer:
    """Stateful, opt-in exploration of the published OptimusKG parquet files.

    Connecting downloads nothing. ``load_node`` and ``load_edge`` fetch one
    stratified table at a time through the official client. ``active`` controls
    which loaded tables participate in traversal, so users can revise their
    analytical projection without deleting cached downloads.
    """

    def __init__(self, client: Any):
        self.client = client
        self.tables: dict[str, pl.DataFrame] = {}
        self.specs: dict[str, DatasetSpec] = {}
        self.active: set[str] = set()

    @classmethod
    def connect(
        cls,
        *,
        cache_dir: str | Path | None = None,
        doi: str | None = None,
        server: str | None = None,
        client: Any | None = None,
    ) -> "ProgressiveExplorer":
        if client is None:
            try:
                import optimuskg as client
            except ImportError as exc:
                raise ImportError(
                    "Install the official client first: pip install optimuskg"
                ) from exc
        if cache_dir is not None:
            client.set_cache_dir(str(Path(cache_dir).expanduser()))
        if doi is not None:
            client.set_doi(doi)
        if server is not None:
            client.set_server(server)
        return cls(client)

    def status(self) -> pl.DataFrame:
        """Connection and in-memory state; this does not download graph data."""
        def configured(getter: str) -> str | None:
            function = getattr(self.client, getter, None)
            return str(function()) if callable(function) else None
        return pl.DataFrame({
            "measure": ["server", "doi", "cache directory", "loaded tables", "active tables"],
            "value": [
                configured("get_server"), configured("get_doi"),
                configured("get_cache_dir"), str(len(self.tables)), str(len(self.active)),
            ],
        })

    def catalog(
        self, *, kind: str | None = None, loaded: bool | None = None
    ) -> pl.DataFrame:
        """Return the documented stratified file catalogue, without downloading."""
        rows = []
        for name, (label, path) in NODE_TYPE_SPECS.items():
            rows.append({
                "dataset": name, "kind": "node", "path": path,
                "source_type": None, "target_type": None, "label": label,
                "loaded": name in self.tables, "active": name in self.active,
            })
        for name, (source, target, label) in EDGE_TYPE_SPECS.items():
            rows.append({
                "dataset": name, "kind": "edge", "path": f"edges/{name}.parquet",
                "source_type": source, "target_type": target, "label": label,
                "loaded": name in self.tables, "active": name in self.active,
            })
        frame = pl.DataFrame(rows)
        if kind is not None:
            if kind not in {"node", "edge"}:
                raise ValueError("kind must be 'node', 'edge', or None.")
            frame = frame.filter(pl.col("kind") == kind)
        if loaded is not None:
            frame = frame.filter(pl.col("loaded") == loaded)
        return frame.sort("kind", "dataset")

    def load_node(
        self, node_type: str, *, force: bool = False, activate: bool = True,
        columns: Sequence[str] | None = None,
    ) -> pl.DataFrame:
        node_type = _live_type(node_type)
        label, path = NODE_TYPE_SPECS[node_type]
        kwargs = {"columns": list(columns)} if columns is not None else {}
        frame = self.client.load_parquet(path, force=force, **kwargs)
        if "id" not in frame.columns:
            raise ValueError(f"{path} is missing the required 'id' column.")
        self.tables[node_type] = frame
        self.specs[node_type] = DatasetSpec(node_type, "node", path, label=label)
        if activate:
            self.active.add(node_type)
        return frame

    def load_edge(
        self, edge_type: str, *, force: bool = False, activate: bool = True,
        columns: Sequence[str] | None = None,
    ) -> pl.DataFrame:
        key = edge_type.strip().lower().replace("-", "_").replace(" ", "_")
        if key not in EDGE_TYPE_SPECS:
            raise ValueError(
                f"Unknown edge type {edge_type!r}. Choose from {sorted(EDGE_TYPE_SPECS)}."
            )
        source, target, label = EDGE_TYPE_SPECS[key]
        path = f"edges/{key}.parquet"
        kwargs = {"columns": list(columns)} if columns is not None else {}
        frame = self.client.load_parquet(path, force=force, **kwargs)
        missing = {"from", "to"} - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        self.tables[key] = frame
        self.specs[key] = DatasetSpec(key, "edge", path, source, target, label)
        if activate:
            self.active.add(key)
        return frame

    def register(
        self, name: str, frame: pl.DataFrame, spec: DatasetSpec,
        *, activate: bool = True,
    ) -> "ProgressiveExplorer":
        """Register a derived/local table explicitly; no biological meaning is inferred."""
        if name != spec.name:
            raise ValueError("name must equal spec.name so provenance stays unambiguous.")
        self.tables[name] = frame
        self.specs[name] = spec
        if activate:
            self.active.add(name)
        return self

    def loaded(self) -> pl.DataFrame:
        rows = []
        for name, frame in self.tables.items():
            spec = self.specs[name]
            rows.append({
                "dataset": name, "kind": spec.kind, "path": spec.path,
                "rows": frame.height, "columns": len(frame.columns),
                "active": name in self.active,
            })
        if not rows:
            return pl.DataFrame({
                "dataset": [], "kind": [], "path": [], "rows": [],
                "columns": [], "active": [],
            })
        return pl.DataFrame(rows).sort("kind", "dataset")

    def activate(self, *datasets: str) -> "ProgressiveExplorer":
        missing = set(datasets) - set(self.tables)
        if missing:
            raise KeyError(f"Load these datasets before activation: {sorted(missing)}")
        self.active.update(datasets)
        return self

    def deactivate(self, *datasets: str) -> "ProgressiveExplorer":
        self.active.difference_update(datasets)
        return self

    def unload(self, *datasets: str) -> "ProgressiveExplorer":
        """Remove tables from memory; the official client's disk cache remains."""
        for name in datasets:
            self.tables.pop(name, None)
            self.specs.pop(name, None)
            self.active.discard(name)
        return self

    def clear(self) -> "ProgressiveExplorer":
        return self.unload(*list(self.tables))

    def preview(self, dataset: str, n: int = 10) -> pl.DataFrame:
        self._require_loaded(dataset)
        return self.tables[dataset].head(n)

    def table_schema(self, dataset: str | None = None) -> pl.DataFrame:
        names = [dataset] if dataset is not None else list(self.tables)
        records = []
        for name in names:
            self._require_loaded(name)
            for column, dtype in self.tables[name].schema.items():
                records.append({"dataset": name, "column": column, "dtype": str(dtype)})
        if not records:
            return pl.DataFrame({"dataset": [], "column": [], "dtype": []})
        return pl.DataFrame(records).sort("dataset", "column")

    def expand_properties(
        self, frame_or_dataset: pl.DataFrame | str,
    ) -> pl.DataFrame:
        """Flatten nested Struct properties into ordinary tabular columns."""
        if isinstance(frame_or_dataset, str):
            self._require_loaded(frame_or_dataset)
            frame = self.tables[frame_or_dataset]
        else:
            frame = frame_or_dataset
        return _flatten_struct_columns(frame)

    def export_table(
        self, frame: pl.DataFrame, path: str | Path,
    ) -> Path:
        """Write one derived table as CSV or Parquet using safe conversions."""
        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        suffix = destination.suffix.lower()
        if suffix == ".csv":
            _csv_safe(frame).write_csv(destination)
        elif suffix in {".parquet", ".pq"}:
            frame.write_parquet(destination)
        else:
            raise ValueError("path must end in .csv, .parquet, or .pq")
        return destination

    def available_edge_types(self, node_type: str) -> pl.DataFrame:
        """Show which documented edge tables can touch a node type."""
        kind = _live_type(node_type)
        return (
            self.catalog(kind="edge")
            .filter(
                (pl.col("source_type") == kind)
                | (pl.col("target_type") == kind)
            )
            .select(
                "dataset", "source_type", "target_type", "label",
                "loaded", "active",
            )
        )

    def profile(self, dataset: str) -> pl.DataFrame:
        """Small, predictable table profile suitable for first-pass EDA."""
        self._require_loaded(dataset)
        frame = self.tables[dataset]
        rows = []
        for column, dtype in frame.schema.items():
            series = frame.get_column(column)
            rows.append({
                "column": column, "dtype": str(dtype), "rows": frame.height,
                "nulls": series.null_count(), "distinct": series.n_unique(),
            })
        return pl.DataFrame(rows)

    def search(
        self, query: str, *, node_type: str | None = None, limit: int = 30
    ) -> pl.DataFrame:
        """Search only node tables the user has deliberately loaded."""
        requested = [_live_type(node_type)] if node_type else [
            name for name, spec in self.specs.items()
            if spec.kind == "node" and name in self.active
        ]
        if node_type and requested[0] not in self.tables:
            raise RuntimeError(
                f"Node table {requested[0]!r} is not loaded. "
                f"Call load_node({requested[0]!r}) first."
            )
        needle = query.strip().lower()
        if not needle:
            raise ValueError("query must not be empty.")
        matches = []
        for kind in requested:
            for row in self.tables[kind].iter_rows(named=True):
                entity_id, name = str(row["id"]), _entity_name(row)
                props = _properties(row)
                synonyms = props.get("synonyms", props.get("alias", []))
                text = " ".join([entity_id, name, str(synonyms)]).lower()
                if needle in text:
                    matches.append({"node_type": kind, "id": entity_id, "name": name})
                    if len(matches) >= limit:
                        return pl.DataFrame(matches)
        return pl.DataFrame(matches) if matches else pl.DataFrame({
            "node_type": [], "id": [], "name": []
        })

    def entity(self, node_type: str, value: str) -> LiveEntity:
        node_type = _live_type(node_type)
        self._require_loaded(node_type)
        needle = value.strip().lower()
        exact, partial = [], []
        for row in self.tables[node_type].iter_rows(named=True):
            entity_id, name = str(row["id"]), _entity_name(row)
            props = _properties(row)
            synonyms = str(props.get("synonyms", props.get("alias", "")))
            values = [entity_id.lower(), name.lower(), synonyms.lower()]
            if needle in values:
                exact.append((row, entity_id, name))
            elif any(needle in candidate for candidate in values):
                partial.append((row, entity_id, name))
        candidates = exact or partial
        if len(candidates) != 1:
            choices = [{"id": item[1], "name": item[2]} for item in candidates[:10]]
            raise ValueError(
                f"Expected one {node_type} for {value!r}; found {len(candidates)}. "
                f"Candidates: {choices}"
            )
        row, entity_id, name = candidates[0]
        return LiveEntity(self, node_type, entity_id, name, row)

    def metagraph(self, *, scope: str = "loaded") -> pl.DataFrame:
        """Show type-level connectivity; no instance paths are implied."""
        if scope not in {"loaded", "catalog"}:
            raise ValueError("scope must be 'loaded' or 'catalog'.")
        records = []
        for name, (source, target, label) in EDGE_TYPE_SPECS.items():
            if scope == "loaded" and (name not in self.active or name not in self.tables):
                continue
            frame = self.tables.get(name)
            if frame is not None and "relation" in frame.columns:
                for relation_row in frame.group_by("relation").len().iter_rows(named=True):
                    records.append({
                        "edge_type": name, "source_type": source,
                        "relation": relation_row["relation"], "target_type": target,
                        "edges": relation_row["len"], "loaded": True,
                    })
            else:
                records.append({
                    "edge_type": name, "source_type": source, "relation": label,
                    "target_type": target, "edges": frame.height if frame is not None else None,
                    "loaded": frame is not None,
                })
        return pl.DataFrame(records) if records else pl.DataFrame({
            "edge_type": [], "source_type": [], "relation": [],
            "target_type": [], "edges": [], "loaded": [],
        })

    def plot_metagraph(self, *, scope: str = "loaded", seed: int = 7):
        """Plot the full documented schema or only currently active edge types."""
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
        except ImportError as exc:
            raise ImportError("Install matplotlib and networkx to plot metagraphs.") from exc
        frame = self.metagraph(scope=scope)
        # The type-level view needs one visible connection per type pair; the
        # detailed relation rows remain available from ``metagraph``.
        graph = nx.DiGraph()
        for row in frame.iter_rows(named=True):
            graph.add_edge(row["source_type"], row["target_type"], label=row["edge_type"])
        figure, axis = plt.subplots(figsize=(13, 8))
        positions = nx.spring_layout(graph, seed=seed)
        nx.draw_networkx(
            graph, positions, node_color="#dbeafe", edgecolors="#2563eb",
            node_size=2300, font_size=8, arrows=True, edge_color="#94a3b8", ax=axis,
        )
        if graph.number_of_edges() <= 15:
            labels = {(u, v): a["label"] for u, v, a in graph.edges(data=True)}
            nx.draw_networkx_edge_labels(graph, positions, edge_labels=labels, font_size=6, ax=axis)
        axis.set_title(f"OptimusKG metagraph ({scope} edge tables)")
        axis.axis("off")
        figure.tight_layout()
        return figure

    def plan_path(
        self, start_type: str, end_type: str, *, via: Sequence[str] = (),
        max_hops: int = 4,
    ) -> LoadPlan:
        """Plan a shortest schema route; return it without loading anything."""
        start, end = _live_type(start_type), _live_type(end_type)
        required_via = tuple(_live_type(item) for item in via)
        routes = self._schema_routes(start, end, max_hops=max_hops)
        routes = [route for route in routes if all(item in route[0] for item in required_via)]
        if not routes:
            raise ValueError(
                f"No schema route from {start} to {end} through {required_via} "
                f"within {max_hops} hops."
            )
        node_types, edge_types = routes[0]
        return LoadPlan(self, tuple(node_types), tuple(edge_types))

    def neighbors(
        self, entity: LiveEntity | tuple[str, str], *,
        edge_types: Iterable[str] | str | None = None, direction: str = "both",
        limit: int | None = 50,
        exclude_self: bool = False,
    ) -> pl.DataFrame:
        if direction not in {"both", "in", "out"}:
            raise ValueError("direction must be 'both', 'in', or 'out'.")
        node_type, node_id = (
            (entity.node_type, entity.id) if isinstance(entity, LiveEntity)
            else (_live_type(entity[0]), str(entity[1]))
        )
        chosen = self._active_edges(edge_types)
        records = []
        for name in chosen:
            spec, frame = self.specs[name], self.tables[name]
            relation_column = "relation" if "relation" in frame.columns else None
            undirected_column = "undirected" if "undirected" in frame.columns else None
            checks = []
            if direction in {"both", "out"} and spec.source_type == node_type:
                checks.append(("from", "to", spec.target_type, "out"))
            if direction in {"both", "in"} and spec.target_type == node_type:
                checks.append(("to", "from", spec.source_type, "in"))
            if spec.source_type == spec.target_type == node_type and direction == "both":
                checks = [("from", "to", node_type, "out"), ("to", "from", node_type, "in")]
            for own, other, other_type, observed_direction in checks:
                subset = frame.filter(pl.col(own).cast(pl.String) == node_id)
                if exclude_self:
                    subset = subset.filter(pl.col(other).cast(pl.String) != node_id)
                columns = ["from", "to"]
                if relation_column:
                    columns.append(relation_column)
                if undirected_column:
                    columns.append(undirected_column)
                for row in subset.select(columns).iter_rows(named=True):
                    records.append({
                        "edge_type": name,
                        "relation": row.get("relation", spec.label),
                        "direction": observed_direction,
                        "neighbor_type": other_type,
                        "neighbor_id": str(row[other]),
                        "neighbor_name": self._lookup_name(other_type, str(row[other])),
                        "undirected": row.get("undirected"),
                    })
        result = pl.DataFrame(records) if records else pl.DataFrame({
            "edge_type": [], "relation": [], "direction": [], "neighbor_type": [],
            "neighbor_id": [], "neighbor_name": [], "undirected": [],
        })
        result = result.unique().sort("edge_type", "neighbor_type", "neighbor_name")
        return result if limit is None else result.head(limit)

    def edge_records(
        self, entity: LiveEntity | tuple[str, str], *,
        edge_types: Iterable[str] | str | None = None,
        direction: str = "both", flatten_properties: bool = True,
        exclude_self: bool = False, limit: int | None = 50,
    ) -> pl.DataFrame:
        """Return matching source rows without discarding evidence properties."""
        if direction not in {"both", "in", "out"}:
            raise ValueError("direction must be 'both', 'in', or 'out'.")
        node_type, node_id = (
            (entity.node_type, entity.id) if isinstance(entity, LiveEntity)
            else (_live_type(entity[0]), str(entity[1]))
        )
        frames = []
        for name in self._active_edges(edge_types):
            spec, frame = self.specs[name], self.tables[name]
            checks = []
            if direction in {"both", "out"} and spec.source_type == node_type:
                checks.append(("from", "to", spec.target_type, "out"))
            if direction in {"both", "in"} and spec.target_type == node_type:
                checks.append(("to", "from", spec.source_type, "in"))
            if spec.source_type == spec.target_type == node_type and direction == "both":
                checks = [
                    ("from", "to", node_type, "out"),
                    ("to", "from", node_type, "in"),
                ]
            for own, other, other_type, observed_direction in checks:
                subset = frame.filter(pl.col(own).cast(pl.String) == node_id)
                if exclude_self:
                    subset = subset.filter(pl.col(other).cast(pl.String) != node_id)
                if subset.is_empty():
                    continue
                neighbor_ids = subset.get_column(other).cast(pl.String).to_list()
                neighbor_names = [
                    self._lookup_name(other_type, neighbor_id)
                    for neighbor_id in neighbor_ids
                ]
                subset = subset.with_columns(
                    pl.lit(name).alias("edge_type"),
                    pl.lit(observed_direction).alias("direction"),
                    pl.lit(other_type).alias("neighbor_type"),
                    pl.col(other).cast(pl.String).alias("neighbor_id"),
                    pl.Series("neighbor_name", neighbor_names, dtype=pl.String),
                )
                frames.append(
                    _flatten_struct_columns(subset)
                    if flatten_properties else subset
                )
        if not frames:
            return pl.DataFrame({
                "edge_type": [], "direction": [], "neighbor_type": [],
                "neighbor_id": [], "neighbor_name": [],
            })
        result = pl.concat(frames, how="diagonal_relaxed")
        return result if limit is None else result.head(limit)

    def edge_summary(
        self, entity: LiveEntity | tuple[str, str], *,
        edge_types: Iterable[str] | str | None = None,
        exclude_self: bool = True,
    ) -> pl.DataFrame:
        """Compare source edge rows with deduplicated direct endpoints."""
        neighbours = self.neighbors(
            entity, edge_types=edge_types, limit=None,
            exclude_self=exclude_self,
        )
        raw = self.edge_records(
            entity, edge_types=edge_types, limit=None,
            flatten_properties=False, exclude_self=exclude_self,
        )
        if neighbours.is_empty():
            return pl.DataFrame({
                "edge_type": [], "raw_edge_rows": [],
                "neighbour_rows": [], "unique_neighbours": [],
            })
        summary = neighbours.group_by("edge_type").agg(
            pl.len().alias("neighbour_rows"),
            pl.col("neighbor_id").n_unique().alias("unique_neighbours"),
        )
        raw_counts = raw.group_by("edge_type").agg(
            pl.len().alias("raw_edge_rows")
        )
        return (
            summary.join(raw_counts, on="edge_type", how="left")
            .select(
                "edge_type", "raw_edge_rows", "neighbour_rows",
                "unique_neighbours",
            )
            .sort("unique_neighbours", descending=True)
        )

    def subgraph(
        self, seeds: Iterable[LiveEntity | tuple[str, str]], *, radius: int = 1,
        edge_types: Iterable[str] | str | None = None,
        max_neighbors_per_hop: int = 50,
        exclude_self: bool = True,
    ) -> WorkingGraph:
        """Build a bounded graph progressively from active in-memory edge tables."""
        if radius < 0:
            raise ValueError("radius must be non-negative.")
        seed_pairs = [
            (item.node_type, item.id) if isinstance(item, LiveEntity)
            else (_live_type(item[0]), str(item[1]))
            for item in seeds
        ]
        known = {(kind, node_id) for kind, node_id in seed_pairs}
        frontier = set(known)
        edge_records: list[dict[str, Any]] = []
        truncated = False
        for _ in range(radius):
            next_frontier: set[tuple[str, str]] = set()
            for node_type, node_id in sorted(frontier):
                rows = self.neighbors(
                    (node_type, node_id), edge_types=edge_types,
                    direction="both", limit=None, exclude_self=exclude_self,
                )
                if rows.height > max_neighbors_per_hop:
                    rows = rows.head(max_neighbors_per_hop)
                    truncated = True
                for row in rows.iter_rows(named=True):
                    neighbor = (row["neighbor_type"], row["neighbor_id"])
                    next_frontier.add(neighbor)
                    if row["direction"] == "out":
                        source, target = node_id, row["neighbor_id"]
                        source_type, target_type = node_type, row["neighbor_type"]
                    else:
                        source, target = row["neighbor_id"], node_id
                        source_type, target_type = row["neighbor_type"], node_type
                    edge_records.append({
                        "from": source, "to": target,
                        "source_type": source_type, "target_type": target_type,
                        "edge_type": row["edge_type"], "relation": row["relation"],
                        "undirected": row["undirected"],
                    })
            next_frontier -= known
            known |= next_frontier
            frontier = next_frontier
            if not frontier:
                break
        node_records = [{
            "node_type": kind, "id": node_id,
            "name": self._lookup_name(kind, node_id),
            "seed": (kind, node_id) in seed_pairs,
        } for kind, node_id in sorted(known)]
        nodes = pl.DataFrame(node_records)
        edges = pl.DataFrame(edge_records).unique() if edge_records else pl.DataFrame({
            "from": [], "to": [], "source_type": [], "target_type": [],
            "edge_type": [], "relation": [], "undirected": [],
        })
        return WorkingGraph(
            self, nodes, edges, tuple(node_id for _, node_id in seed_pairs), radius, truncated
        )

    def evidence_bundle(
        self, seed: LiveEntity | tuple[str, str], *, radius: int = 1,
        edge_types: Iterable[str] | str | None = None,
        max_neighbors_per_hop: int = 5000,
        exclude_self: bool = True,
    ) -> EvidenceBundle:
        """Create normalized entity, relationship and evidence tables."""
        selected = self._active_edges(edge_types)
        graph = self.subgraph(
            [seed], radius=radius, edge_types=selected,
            max_neighbors_per_hop=max_neighbors_per_hop,
            exclude_self=exclude_self,
        )
        relationships = graph.edges
        if exclude_self and not relationships.is_empty():
            relationships = relationships.filter(pl.col("from") != pl.col("to"))
        if not relationships.is_empty():
            relationships = relationships.with_columns(
                pl.concat_str(
                    [
                        pl.col("edge_type"), pl.col("from"), pl.col("to"),
                        pl.col("relation").fill_null(""),
                    ],
                    separator="|",
                ).alias("relationship_key")
            ).select(
                "relationship_key", "from", "to", "source_type",
                "target_type", "edge_type", "relation", "undirected",
            ).unique()
        else:
            relationships = pl.DataFrame({
                "relationship_key": [], "from": [], "to": [],
                "source_type": [], "target_type": [], "edge_type": [],
                "relation": [], "undirected": [],
            })

        endpoint_ids = set(graph.seeds)
        if not relationships.is_empty():
            endpoint_ids.update(relationships.get_column("from").to_list())
            endpoint_ids.update(relationships.get_column("to").to_list())
        entities = graph.nodes.filter(pl.col("id").is_in(sorted(endpoint_ids)))

        evidence_frames = []
        for name in selected:
            chosen = relationships.filter(pl.col("edge_type") == name)
            if chosen.is_empty():
                continue
            raw = self.tables[name]
            join_keys = ["from", "to"]
            if "relation" in raw.columns:
                join_keys.append("relation")
            keys = chosen.select(join_keys).unique()
            details = raw.join(keys, on=join_keys, how="inner")
            if "relation" not in details.columns:
                details = details.with_columns(pl.lit(self.specs[name].label).alias("relation"))
            details = details.with_columns(
                pl.lit(name).alias("edge_type"),
                pl.concat_str(
                    [
                        pl.lit(name), pl.col("from").cast(pl.String),
                        pl.col("to").cast(pl.String),
                        pl.col("relation").cast(pl.String).fill_null(""),
                    ],
                    separator="|",
                ).alias("relationship_key"),
            )
            evidence_frames.append(_flatten_struct_columns(details))
        evidence = (
            pl.concat(evidence_frames, how="diagonal_relaxed")
            if evidence_frames else pl.DataFrame({"relationship_key": [], "edge_type": []})
        )

        seed_type, seed_id = (
            (seed.node_type, seed.id) if isinstance(seed, LiveEntity)
            else (_live_type(seed[0]), str(seed[1]))
        )
        getter = lambda name: (
            str(getattr(self.client, name)())
            if callable(getattr(self.client, name, None)) else None
        )
        manifest = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_server": getter("get_server"),
            "source_doi": getter("get_doi"),
            "seed": {"node_type": seed_type, "id": seed_id},
            "radius": radius,
            "edge_types": selected,
            "exclude_self": exclude_self,
            "max_neighbors_per_hop": max_neighbors_per_hop,
            "truncated": graph.truncated,
            "counts": {
                "entities": entities.height,
                "relationships": relationships.height,
                "evidence_rows": evidence.height,
            },
            "interpretation": (
                "Question-scoped recorded evidence; paths and proximity do not establish causality."
            ),
        }
        return EvidenceBundle(entities, relationships, evidence, manifest)

    def _active_edges(
        self, edge_types: Iterable[str] | str | None
    ) -> list[str]:
        normalised = _edge_type_list(edge_types)
        chosen = normalised if normalised is not None else [
            name for name in self.active
            if name in self.specs and self.specs[name].kind == "edge"
        ]
        missing = [name for name in chosen if name not in self.tables]
        if missing:
            raise RuntimeError(f"Load these edge tables first: {sorted(missing)}")
        return sorted(chosen)

    def _lookup_name(self, node_type: str, node_id: str) -> str:
        if node_type not in self.tables:
            return node_id
        frame = self.tables[node_type].filter(pl.col("id").cast(pl.String) == node_id).head(1)
        return _entity_name(frame.row(0, named=True)) if frame.height else node_id

    def _require_loaded(self, dataset: str) -> None:
        if dataset not in self.tables:
            raise RuntimeError(
                f"Dataset {dataset!r} is not loaded. Inspect catalog(), then call "
                "load_node(...) or load_edge(...)."
            )

    @staticmethod
    def _schema_routes(start: str, end: str, *, max_hops: int):
        routes = [([start], [])]
        completed = []
        for _ in range(max_hops):
            next_routes = []
            for nodes, edges in routes:
                current = nodes[-1]
                for edge_name, (source, target, _) in EDGE_TYPE_SPECS.items():
                    if current == source:
                        neighbor = target
                    elif current == target:
                        neighbor = source
                    else:
                        continue
                    if neighbor in nodes and neighbor != end:
                        continue
                    candidate = (nodes + [neighbor], edges + [edge_name])
                    if neighbor == end:
                        completed.append(candidate)
                    else:
                        next_routes.append(candidate)
            routes = next_routes
        return sorted(completed, key=lambda item: (len(item[1]), item[1]))
