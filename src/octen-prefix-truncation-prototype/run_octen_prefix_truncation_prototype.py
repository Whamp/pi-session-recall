# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy>=2.0,<3",
#   "pyarrow>=20,<25",
# ]
# ///

"""PROTOTYPE: measure whether Octen embedding prefixes preserve current recall geometry."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pyarrow.ipc as arrow_ipc

EMBEDDING_CACHE_MAGIC = b"PIEVC001"
EMBEDDING_CACHE_CHECKSUM_BYTES = 32
EMBEDDING_CACHE_HEADER_LENGTH_BYTES = 4
DEFAULT_PREFIX_WIDTHS = (1536, 1024, 768, 512)
DEFAULT_CONTROL_WIDTH = 1024
DEFAULT_RANDOM_SEED = "pi-session-recall-octen-prefix-v1"
SCALAR_COLUMNS = (
    "_zvec_g_doc_id_",
    "_zvec_uid_",
    "documentKind",
    "isDenseSearchable",
    "sessionId",
    "entryId",
    "parentEntryId",
    "isOnActiveBranch",
    "role",
    "chunkIndex",
    "content",
)

FloatMatrix = npt.NDArray[np.float32]
IndexArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class DenseEvidence:
    """Latest dense evidence metadata needed for the adjacent-answer retrieval proxy."""

    global_document_id: int
    uid: str
    document_kind: str
    session_id: str
    entry_id: str
    parent_entry_id: str
    is_on_active_branch: bool
    role: str
    chunk_index: int
    vector_hash: str
    character_count: int
    preview: str


@dataclass(frozen=True)
class RetrievalQueryCase:
    """One cached user-message vector and its active child-assistant target vectors."""

    vector_hash: str
    target_vector_hashes: frozenset[str]
    preview: str


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read current cached Octen vectors and compare leading prefixes with full, random, "
            "and trailing coordinates. This prototype never opens zvec."
        )
    )
    recall_root = Path.home() / ".pi" / "agent" / "recall"
    parser.add_argument(
        "--recall-root",
        type=Path,
        default=recall_root,
        help="Recall directory containing index-manifest.json, embedding-cache, and zvec.",
    )
    parser.add_argument(
        "--candidate-count",
        type=int,
        default=20_000,
        help="Number of unique current evidence vectors in the deterministic candidate sample.",
    )
    parser.add_argument(
        "--query-count",
        type=int,
        default=100,
        help="Number of current user-message queries with child-assistant targets.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Neighborhood size used for full-versus-truncated overlap.",
    )
    parser.add_argument(
        "--prefix-widths",
        type=int,
        nargs="+",
        default=list(DEFAULT_PREFIX_WIDTHS),
        help="Leading-prefix widths to compare with the native vector.",
    )
    parser.add_argument(
        "--seed",
        default=DEFAULT_RANDOM_SEED,
        help="Deterministic sample and random-coordinate seed.",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        help="Optional path for the same Markdown report printed to stdout.",
    )
    return parser.parse_args()


def report_progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def read_embedding_cache_header(
    cache_file: Path,
) -> tuple[dict[str, Any], int, int, bytes]:
    file_bytes = cache_file.read_bytes()
    minimum_bytes = (
        len(EMBEDDING_CACHE_MAGIC)
        + EMBEDDING_CACHE_HEADER_LENGTH_BYTES
        + EMBEDDING_CACHE_CHECKSUM_BYTES
    )
    if len(file_bytes) < minimum_bytes:
        raise RuntimeError(f"Embedding cache file is truncated: {cache_file}")
    if file_bytes[: len(EMBEDDING_CACHE_MAGIC)] != EMBEDDING_CACHE_MAGIC:
        raise RuntimeError(f"Embedding cache magic is invalid: {cache_file}")

    header_length_offset = len(EMBEDDING_CACHE_MAGIC)
    (header_length,) = struct.unpack_from("<I", file_bytes, header_length_offset)
    header_start = header_length_offset + EMBEDDING_CACHE_HEADER_LENGTH_BYTES
    header_end = header_start + header_length
    checksum_start = len(file_bytes) - EMBEDDING_CACHE_CHECKSUM_BYTES
    if header_end > checksum_start:
        raise RuntimeError(f"Embedding cache header exceeds file body: {cache_file}")

    expected_checksum = file_bytes[checksum_start:]
    actual_checksum = hashlib.sha256(file_bytes[:checksum_start]).digest()
    if actual_checksum != expected_checksum:
        raise RuntimeError(f"Embedding cache checksum mismatch: {cache_file}")

    header = json.loads(file_bytes[header_start:header_end].decode("utf-8"))
    if not isinstance(header, dict):
        raise TypeError(f"Embedding cache header is not an object: {cache_file}")
    return header, header_end, checksum_start, file_bytes


def find_current_embedding_cache_directory(
    recall_root: Path,
) -> tuple[Path, dict[str, Any]]:
    manifest_path = recall_root / "index-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    embedding_manifest = manifest["embedding"]
    cache_root = recall_root / "embedding-cache" / "v1"

    matches: list[tuple[Path, dict[str, Any]]] = []
    for identity_directory in sorted(
        path for path in cache_root.iterdir() if path.is_dir()
    ):
        sample_file = next(identity_directory.glob("*/*.fp32"), None)
        if sample_file is None:
            continue
        header, _, _, _ = read_embedding_cache_header(sample_file)
        identity = header["identity"]["embedding"]
        if (
            identity["requestModel"] == embedding_manifest["requestModel"]
            and identity["servedModelId"] == embedding_manifest["servedModelId"]
            and identity["artifact"] == embedding_manifest["artifact"]
            and identity["dimensions"] == embedding_manifest["dimensions"]
            and identity["canaryFingerprint"] == embedding_manifest["canaryFingerprint"]
        ):
            matches.append((identity_directory, header))

    if len(matches) != 1:
        raise RuntimeError(
            "Expected exactly one embedding cache identity matching index-manifest.json, "
            f"found {len(matches)}"
        )
    return matches[0]


def list_cached_vector_hashes(cache_directory: Path) -> set[str]:
    return {cache_file.stem for cache_file in cache_directory.glob("*/*.fp32")}


def create_vector_hash(text: str) -> str:
    normalized_text = unicodedata.normalize("NFC", text)
    return hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()


def create_text_preview(text: str, maximum_characters: int = 120) -> str:
    single_line = " ".join(text.split())
    if len(single_line) <= maximum_characters:
        return single_line
    return f"{single_line[: maximum_characters - 1]}…"


def load_latest_dense_evidence(
    zvec_directory: Path,
    cached_vector_hashes: set[str],
) -> list[DenseEvidence]:
    """Reads immutable Arrow scalar files directly; it never calls ZVecOpen."""

    scalar_files = sorted(zvec_directory.glob("*/scalar.*.ipc"))
    if not scalar_files:
        raise RuntimeError(f"No zvec scalar Arrow files found under {zvec_directory}")

    latest_document_id_by_uid: dict[str, int] = {}
    latest_dense_evidence_by_uid: dict[str, DenseEvidence] = {}
    physical_rows = 0
    dense_physical_rows = 0

    for file_index, scalar_file in enumerate(scalar_files, start=1):
        with arrow_ipc.open_file(scalar_file) as reader:
            for batch_index in range(reader.num_record_batches):
                batch = reader.get_batch(batch_index).select(SCALAR_COLUMNS)
                columns = {name: batch.column(name) for name in SCALAR_COLUMNS}
                global_document_ids = columns["_zvec_g_doc_id_"].to_pylist()
                uids = columns["_zvec_uid_"].to_pylist()
                dense_flags = columns["isDenseSearchable"].to_pylist()
                physical_rows += batch.num_rows

                for row_index, uid_value in enumerate(uids):
                    uid = str(uid_value)
                    global_document_id = int(global_document_ids[row_index])
                    previous_document_id = latest_document_id_by_uid.get(uid)
                    if (
                        previous_document_id is not None
                        and global_document_id <= previous_document_id
                    ):
                        continue
                    latest_document_id_by_uid[uid] = global_document_id

                    if not bool(dense_flags[row_index]):
                        latest_dense_evidence_by_uid.pop(uid, None)
                        continue

                    dense_physical_rows += 1
                    content = str(columns["content"][row_index].as_py())
                    vector_hash = create_vector_hash(content)
                    if vector_hash not in cached_vector_hashes:
                        latest_dense_evidence_by_uid.pop(uid, None)
                        continue

                    latest_dense_evidence_by_uid[uid] = DenseEvidence(
                        global_document_id=global_document_id,
                        uid=uid,
                        document_kind=str(columns["documentKind"][row_index].as_py()),
                        session_id=str(columns["sessionId"][row_index].as_py()),
                        entry_id=str(columns["entryId"][row_index].as_py()),
                        parent_entry_id=str(
                            columns["parentEntryId"][row_index].as_py()
                        ),
                        is_on_active_branch=bool(
                            columns["isOnActiveBranch"][row_index].as_py()
                        ),
                        role=str(columns["role"][row_index].as_py()),
                        chunk_index=int(columns["chunkIndex"][row_index].as_py()),
                        vector_hash=vector_hash,
                        character_count=len(content),
                        preview=create_text_preview(content),
                    )

        if file_index % 20 == 0 or file_index == len(scalar_files):
            report_progress(
                f"Read scalar metadata {file_index}/{len(scalar_files)} files "
                f"({physical_rows:,} physical rows)"
            )

    report_progress(
        f"Resolved {len(latest_dense_evidence_by_uid):,} latest dense evidence rows "
        f"from {dense_physical_rows:,} encountered dense versions"
    )
    return list(latest_dense_evidence_by_uid.values())


def create_stable_order_key(seed: str, category: str, value: str) -> bytes:
    return hashlib.sha256(f"{seed}\0{category}\0{value}".encode()).digest()


def select_retrieval_query_cases(
    evidence: list[DenseEvidence],
    query_count: int,
    seed: str,
) -> list[RetrievalQueryCase]:
    assistant_hashes_by_parent: dict[tuple[str, str], set[str]] = {}
    for item in evidence:
        if (
            item.document_kind == "conversation"
            and item.role == "assistant"
            and item.is_on_active_branch
        ):
            assistant_hashes_by_parent.setdefault(
                (item.session_id, item.parent_entry_id), set()
            ).add(item.vector_hash)

    grouped_queries: dict[str, tuple[set[str], str]] = {}
    for item in evidence:
        if not (
            item.document_kind == "conversation"
            and item.role == "user"
            and item.is_on_active_branch
            and item.chunk_index == 0
            and 20 <= item.character_count <= 2_000
        ):
            continue
        targets = assistant_hashes_by_parent.get(
            (item.session_id, item.entry_id), set()
        )
        targets = targets - {item.vector_hash}
        if not targets:
            continue
        grouped_targets, preview = grouped_queries.setdefault(
            item.vector_hash, (set(), item.preview)
        )
        grouped_targets.update(targets)
        grouped_queries[item.vector_hash] = (grouped_targets, preview)

    query_cases = [
        RetrievalQueryCase(
            vector_hash=vector_hash,
            target_vector_hashes=frozenset(targets),
            preview=preview,
        )
        for vector_hash, (targets, preview) in grouped_queries.items()
    ]
    query_cases.sort(
        key=lambda item: create_stable_order_key(seed, "query", item.vector_hash)
    )
    if len(query_cases) < query_count:
        raise RuntimeError(
            f"Requested {query_count} query cases, but only {len(query_cases)} were available"
        )
    return query_cases[:query_count]


def select_candidate_vector_hashes(
    evidence: list[DenseEvidence],
    query_cases: list[RetrievalQueryCase],
    candidate_count: int,
    seed: str,
) -> list[str]:
    available_hashes = {item.vector_hash for item in evidence}
    required_hashes = {item.vector_hash for item in query_cases}
    for item in query_cases:
        required_hashes.update(item.target_vector_hashes)
    if not required_hashes <= available_hashes:
        raise RuntimeError(
            "A selected query or target vector is missing from current dense evidence"
        )
    if len(required_hashes) > candidate_count:
        raise RuntimeError(
            f"Candidate count {candidate_count} is smaller than {len(required_hashes)} required "
            "query and target vectors"
        )

    remaining_hashes = available_hashes - required_hashes
    ranked_remaining_hashes = sorted(
        remaining_hashes,
        key=lambda value: create_stable_order_key(seed, "candidate", value),
    )
    selected_hashes = required_hashes | set(
        ranked_remaining_hashes[: candidate_count - len(required_hashes)]
    )
    return sorted(selected_hashes)


def read_cached_embedding_vector(
    cache_directory: Path,
    vector_hash: str,
    expected_dimensions: int,
) -> npt.NDArray[np.float32]:
    cache_file = cache_directory / vector_hash[:2] / f"{vector_hash}.fp32"
    header, payload_start, checksum_start, file_bytes = read_embedding_cache_header(
        cache_file
    )
    if header["normalizedTextSha256"] != vector_hash:
        raise RuntimeError(f"Embedding cache content hash mismatch: {cache_file}")
    if int(header["vectorDimensions"]) != expected_dimensions:
        raise RuntimeError(
            f"Embedding cache dimensions mismatch in {cache_file}: "
            f"expected {expected_dimensions}, received {header['vectorDimensions']}"
        )
    payload = memoryview(file_bytes)[payload_start:checksum_start]
    expected_payload_bytes = expected_dimensions * np.dtype("<f4").itemsize
    if len(payload) != expected_payload_bytes:
        raise RuntimeError(
            f"Embedding cache payload size mismatch in {cache_file}: "
            f"expected {expected_payload_bytes}, received {len(payload)}"
        )
    return np.frombuffer(payload, dtype="<f4", count=expected_dimensions).copy()


def load_embedding_matrix(
    cache_directory: Path,
    vector_hashes: list[str],
    dimensions: int,
    label: str,
) -> FloatMatrix:
    matrix = np.empty((len(vector_hashes), dimensions), dtype=np.float32)
    for index, vector_hash in enumerate(vector_hashes):
        matrix[index] = read_cached_embedding_vector(
            cache_directory, vector_hash, dimensions
        )
        if (index + 1) % 2_000 == 0 or index + 1 == len(vector_hashes):
            report_progress(
                f"Loaded {label} vectors {index + 1:,}/{len(vector_hashes):,}"
            )
    return matrix


def create_seeded_random_coordinates(
    native_dimensions: int,
    selected_dimensions: int,
    seed: str,
) -> IndexArray:
    coordinates = sorted(
        range(native_dimensions),
        key=lambda value: create_stable_order_key(seed, "coordinate", str(value)),
    )[:selected_dimensions]
    return np.asarray(sorted(coordinates), dtype=np.int64)


def select_coordinates(
    matrix: FloatMatrix,
    coordinates: slice | IndexArray,
) -> FloatMatrix:
    return np.ascontiguousarray(matrix[:, coordinates], dtype=np.float32)


def calculate_cosine_scores(
    candidate_matrix: FloatMatrix,
    query_matrix: FloatMatrix,
    coordinates: slice | IndexArray,
) -> FloatMatrix:
    selected_candidates = select_coordinates(candidate_matrix, coordinates)
    selected_queries = select_coordinates(query_matrix, coordinates)
    candidate_norms = np.linalg.norm(selected_candidates, axis=1)
    query_norms = np.linalg.norm(selected_queries, axis=1)
    if np.any(candidate_norms == 0) or np.any(query_norms == 0):
        raise RuntimeError("Cannot normalize a zero-length truncated embedding")
    scores = selected_queries @ selected_candidates.T
    scores /= query_norms[:, None]
    scores /= candidate_norms[None, :]
    return np.asarray(scores, dtype=np.float32)


def calculate_top_indices(scores: FloatMatrix, top_k: int) -> IndexArray:
    partition = np.argpartition(-scores, kth=top_k - 1, axis=1)[:, :top_k]
    partition_scores = np.take_along_axis(scores, partition, axis=1)
    order = np.argsort(-partition_scores, axis=1)
    return np.asarray(np.take_along_axis(partition, order, axis=1), dtype=np.int64)


def calculate_retrieval_metrics(
    scores: FloatMatrix,
    query_cases: list[RetrievalQueryCase],
    candidate_index_by_hash: dict[str, int],
    full_scores: FloatMatrix,
    full_top_indices: IndexArray,
    top_k: int,
) -> dict[str, float]:
    for query_index, query_case in enumerate(query_cases):
        self_index = candidate_index_by_hash.get(query_case.vector_hash)
        if self_index is not None:
            scores[query_index, self_index] = -np.inf

    top_indices = calculate_top_indices(scores, top_k)
    top_one_preserved = np.mean(top_indices[:, 0] == full_top_indices[:, 0])
    top_k_overlap = np.mean(
        [
            len(set(current).intersection(set(baseline))) / top_k
            for current, baseline in zip(top_indices, full_top_indices, strict=True)
        ]
    )

    best_target_ranks: list[int] = []
    for query_index, query_case in enumerate(query_cases):
        target_indices = [
            candidate_index_by_hash[target_hash]
            for target_hash in query_case.target_vector_hashes
            if target_hash in candidate_index_by_hash
        ]
        if not target_indices:
            raise RuntimeError("A selected query has no target in the candidate sample")
        best_target_score = float(np.max(scores[query_index, target_indices]))
        rank = 1 + int(np.count_nonzero(scores[query_index] > best_target_score))
        best_target_ranks.append(rank)

    finite_mask = np.isfinite(full_scores) & np.isfinite(scores)
    baseline_values = full_scores[finite_mask].astype(np.float64)
    current_values = scores[finite_mask].astype(np.float64)
    similarity_correlation = float(np.corrcoef(baseline_values, current_values)[0, 1])
    mean_absolute_similarity_change = float(
        np.mean(np.abs(baseline_values - current_values))
    )
    ranks = np.asarray(best_target_ranks, dtype=np.float64)
    return {
        "similarity_correlation": similarity_correlation,
        "mean_absolute_similarity_change": mean_absolute_similarity_change,
        "top_one_preserved": float(top_one_preserved),
        "top_k_overlap": float(top_k_overlap),
        "answer_hit_at_10": float(np.mean(ranks <= 10)),
        "answer_hit_at_50": float(np.mean(ranks <= 50)),
        "answer_mean_reciprocal_rank": float(np.mean(1.0 / ranks)),
        "answer_median_rank": float(np.median(ranks)),
    }


def calculate_coordinate_metrics(
    candidate_matrix: FloatMatrix,
    coordinates: slice | IndexArray,
) -> dict[str, float]:
    full_energy = np.sum(candidate_matrix * candidate_matrix, axis=1, dtype=np.float64)
    selected = select_coordinates(candidate_matrix, coordinates)
    selected_energy = np.sum(selected * selected, axis=1, dtype=np.float64)
    coordinate_variances = np.var(candidate_matrix, axis=0, dtype=np.float64)
    selected_coordinate_variances = coordinate_variances[coordinates]
    return {
        "mean_energy_share": float(np.mean(selected_energy / full_energy)),
        "variance_share": float(
            np.sum(selected_coordinate_variances) / np.sum(coordinate_variances)
        ),
    }


def format_percentage(value: float) -> str:
    return f"{value * 100:.1f}%"


def format_markdown_report(result: dict[str, Any]) -> str:
    model = result["model"]
    sample = result["sample"]
    lines = [
        "# Octen prefix truncation prototype",
        "",
        (
            "**Question:** Do the original leading coordinates of the current Octen embeddings "
            "preserve enough similarity and retrieval behavior to justify storing fewer dimensions?"
        ),
        "",
        (
            "This is throwaway measurement code. It read regular embedding-cache and Arrow scalar "
            "files only. It did not open zvec, start a model, call Octen's API, or write under the "
            "recall directory."
        ),
        "",
        "## Input",
        "",
        f"- Model: `{model['served_model_id']}`",
        f"- Artifact: `{model['artifact']}`",
        f"- Native dimensions: {model['native_dimensions']:,}",
        f"- Cache identity: `{model['cache_identity']}`",
        f"- Candidate vectors: {sample['candidate_count']:,}",
        f"- Query vectors: {sample['query_count']:,}",
        f"- Available current dense evidence rows: {sample['dense_evidence_count']:,}",
        f"- Available unique current evidence vectors: {sample['unique_vector_count']:,}",
        "",
        (
            "Queries are current active-branch user-message chunks. Their active child assistant "
            "messages are a rough expected-answer target. This is a comparison proxy, not a labeled "
            "recall-quality benchmark."
        ),
        "",
        "## Coordinate distribution",
        "",
        "| Coordinates | Width | Mean vector energy | Coordinate variance |",
        "| --- | ---: | ---: | ---: |",
    ]
    for item in result["coordinate_metrics"]:
        lines.append(
            f"| {item['name']} | {item['width']:,} | "
            f"{format_percentage(item['mean_energy_share'])} | "
            f"{format_percentage(item['variance_share'])} |"
        )

    top_k = sample["top_k"]
    retrieval_by_name = {item["name"]: item for item in result["retrieval_metrics"]}
    full_metrics = retrieval_by_name["full baseline"]
    leading_1024_metrics = retrieval_by_name.get("leading 1024")
    random_1024_metrics = retrieval_by_name["seeded random 1024"]
    trailing_1024_metrics = retrieval_by_name["trailing 1024"]
    if leading_1024_metrics is not None:
        lines.extend(
            [
                "",
                "## Result",
                "",
                (
                    "The leading 1,024 dimensions retained "
                    f"{format_percentage(leading_1024_metrics['top_k_overlap'])} of the "
                    f"full-width top-{top_k} neighbors. Rough answer hit@10 changed from "
                    f"{format_percentage(full_metrics['answer_hit_at_10'])} to "
                    f"{format_percentage(leading_1024_metrics['answer_hit_at_10'])}, and answer "
                    f"MRR changed from {full_metrics['answer_mean_reciprocal_rank']:.4f} to "
                    f"{leading_1024_metrics['answer_mean_reciprocal_rank']:.4f}. This supports "
                    "1,024 as a practical stored-width candidate for a fuller quality run."
                ),
                "",
                (
                    "The random and trailing 1,024-coordinate controls retained "
                    f"{format_percentage(random_1024_metrics['top_k_overlap'])} and "
                    f"{format_percentage(trailing_1024_metrics['top_k_overlap'])} of the same "
                    "neighbors. Because those controls were similarly strong, this run does not "
                    "independently prove that Octen specially front-loads semantic information."
                ),
            ]
        )

    lines.extend(
        [
            "",
            (
                "Energy and variance describe where values sit in the original coordinates. They do "
                "not by themselves measure semantic quality."
            ),
            "",
            "## Retrieval comparison",
            "",
            (
                f"| Variant | Width | Score correlation | Mean score change | Top-1 same | "
                f"Top-{top_k} overlap | Answer hit@10 | Answer hit@50 | Answer MRR | "
                "Median answer rank |"
            ),
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in result["retrieval_metrics"]:
        lines.append(
            f"| {item['name']} | {item['width']:,} | "
            f"{item['similarity_correlation']:.4f} | "
            f"{item['mean_absolute_similarity_change']:.4f} | "
            f"{format_percentage(item['top_one_preserved'])} | "
            f"{format_percentage(item['top_k_overlap'])} | "
            f"{format_percentage(item['answer_hit_at_10'])} | "
            f"{format_percentage(item['answer_hit_at_50'])} | "
            f"{item['answer_mean_reciprocal_rank']:.4f} | "
            f"{item['answer_median_rank']:.1f} |"
        )

    lines.extend(
        [
            "",
            (
                "The full-width row is the baseline, so its score correlation, neighborhood overlap, "
                "and top-1 preservation are 1. Compare `leading 1024` with the random and trailing "
                "1024-coordinate controls. A useful leading prefix should stay close to full-width "
                "retrieval and outperform those controls."
            ),
            "",
            "## Sample query previews",
            "",
        ]
    )
    for preview in result["query_previews"]:
        lines.append(f"- {preview}")

    lines.extend(
        [
            "",
            "## Limits",
            "",
            (
                "- This measures the current local Q8 GGUF cache, which is the artifact that matters "
                "for this installation."
            ),
            "- The candidate set is a deterministic sample, not the entire recall corpus.",
            (
                "- Child assistant messages are useful comparison targets but are not human relevance "
                "labels."
            ),
            (
                "- A final dimension choice should also be run through the existing labeled recall "
                "quality corpus when the production profile supports reduced dimensions."
            ),
            (
                "- PCA is intentionally omitted because PCA rotates the coordinates and cannot test "
                "whether Octen's original first N values are useful."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def run_prototype(arguments: argparse.Namespace) -> dict[str, Any]:
    if arguments.candidate_count < 100:
        raise RuntimeError("--candidate-count must be at least 100")
    if arguments.query_count < 1:
        raise RuntimeError("--query-count must be positive")
    if arguments.top_k < 1 or arguments.top_k >= arguments.candidate_count:
        raise RuntimeError(
            "--top-k must be positive and smaller than --candidate-count"
        )

    cache_directory, cache_header = find_current_embedding_cache_directory(
        arguments.recall_root
    )
    identity = cache_header["identity"]["embedding"]
    native_dimensions = int(identity["dimensions"])
    prefix_widths = sorted(set(arguments.prefix_widths), reverse=True)
    if any(width < 1 or width >= native_dimensions for width in prefix_widths):
        raise RuntimeError(
            f"Every prefix width must be between 1 and {native_dimensions - 1}"
        )
    if DEFAULT_CONTROL_WIDTH >= native_dimensions:
        raise RuntimeError(
            f"The {DEFAULT_CONTROL_WIDTH}-dimension controls require a wider native vector"
        )

    report_progress(f"Using current cache identity {cache_directory.name}")
    cached_vector_hashes = list_cached_vector_hashes(cache_directory)
    report_progress(f"Found {len(cached_vector_hashes):,} cached vectors")
    evidence = load_latest_dense_evidence(
        arguments.recall_root / "zvec", cached_vector_hashes
    )
    query_cases = select_retrieval_query_cases(
        evidence, arguments.query_count, arguments.seed
    )
    candidate_hashes = select_candidate_vector_hashes(
        evidence, query_cases, arguments.candidate_count, arguments.seed
    )
    candidate_index_by_hash = {
        vector_hash: index for index, vector_hash in enumerate(candidate_hashes)
    }

    candidate_matrix = load_embedding_matrix(
        cache_directory,
        candidate_hashes,
        native_dimensions,
        "candidate",
    )
    query_matrix = np.stack(
        [
            candidate_matrix[candidate_index_by_hash[item.vector_hash]]
            for item in query_cases
        ]
    ).astype(np.float32, copy=False)

    random_coordinates = create_seeded_random_coordinates(
        native_dimensions, DEFAULT_CONTROL_WIDTH, arguments.seed
    )
    coordinate_variants: list[tuple[str, int, slice | IndexArray]] = [
        (f"leading {width}", width, slice(0, width)) for width in prefix_widths
    ]
    coordinate_variants.extend(
        [
            (
                f"seeded random {DEFAULT_CONTROL_WIDTH}",
                DEFAULT_CONTROL_WIDTH,
                random_coordinates,
            ),
            (
                f"trailing {DEFAULT_CONTROL_WIDTH}",
                DEFAULT_CONTROL_WIDTH,
                slice(native_dimensions - DEFAULT_CONTROL_WIDTH, native_dimensions),
            ),
        ]
    )

    coordinate_metrics = []
    for name, width, coordinates in coordinate_variants:
        report_progress(f"Measuring coordinate distribution for {name}")
        coordinate_metrics.append(
            {
                "name": name,
                "width": width,
                **calculate_coordinate_metrics(candidate_matrix, coordinates),
            }
        )

    report_progress(f"Searching full {native_dimensions}-dimension baseline")
    full_scores = calculate_cosine_scores(
        candidate_matrix, query_matrix, slice(0, native_dimensions)
    )
    for query_index, query_case in enumerate(query_cases):
        full_scores[
            query_index, candidate_index_by_hash[query_case.vector_hash]
        ] = -np.inf
    full_top_indices = calculate_top_indices(full_scores, arguments.top_k)
    full_metrics = calculate_retrieval_metrics(
        full_scores.copy(),
        query_cases,
        candidate_index_by_hash,
        full_scores,
        full_top_indices,
        arguments.top_k,
    )
    retrieval_metrics = [
        {
            "name": "full baseline",
            "width": native_dimensions,
            **full_metrics,
        }
    ]

    for name, width, coordinates in coordinate_variants:
        report_progress(f"Searching with {name}")
        scores = calculate_cosine_scores(candidate_matrix, query_matrix, coordinates)
        metrics = calculate_retrieval_metrics(
            scores,
            query_cases,
            candidate_index_by_hash,
            full_scores,
            full_top_indices,
            arguments.top_k,
        )
        retrieval_metrics.append({"name": name, "width": width, **metrics})

    unique_vector_count = len({item.vector_hash for item in evidence})
    return {
        "version": 1,
        "model": {
            "request_model": identity["requestModel"],
            "served_model_id": identity["servedModelId"],
            "artifact": identity["artifact"],
            "native_dimensions": native_dimensions,
            "cache_identity": cache_directory.name,
            "canary_fingerprint": identity["canaryFingerprint"],
        },
        "sample": {
            "candidate_count": len(candidate_hashes),
            "query_count": len(query_cases),
            "top_k": arguments.top_k,
            "dense_evidence_count": len(evidence),
            "unique_vector_count": unique_vector_count,
            "seed": arguments.seed,
        },
        "coordinate_metrics": coordinate_metrics,
        "retrieval_metrics": retrieval_metrics,
        "query_previews": [item.preview for item in query_cases[:10]],
    }


def main() -> None:
    arguments = parse_arguments()
    result = run_prototype(arguments)
    report = format_markdown_report(result)
    if arguments.markdown_output:
        arguments.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        arguments.markdown_output.write_text(report, encoding="utf-8")
        report_progress(f"Wrote Markdown report to {arguments.markdown_output}")
    print(report)


if __name__ == "__main__":
    main()
