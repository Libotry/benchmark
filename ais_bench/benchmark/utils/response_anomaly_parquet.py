"""Parquet storage for response-anomaly token and logprob payloads."""

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional


SCHEMA_VERSION = 1


def _load_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required for response anomaly Parquet storage. "
            "Install the AISBench response_anomaly extra."
        ) from exc
    return pa, pq


def payload_schema():
    pa, _ = _load_pyarrow()
    return pa.schema(
        [
            pa.field("schema_version", pa.int16(), nullable=False),
            pa.field("data_abbr", pa.string(), nullable=False),
            pa.field("model_abbr", pa.string(), nullable=False),
            pa.field("id", pa.string(), nullable=False),
            pa.field("uuid", pa.string(), nullable=False),
            pa.field("token_count", pa.int32(), nullable=False),
            pa.field("tokens", pa.list_(pa.int64()), nullable=False),
            pa.field(
                "topk_token_ids",
                pa.list_(pa.list_(pa.int64())),
                nullable=False,
            ),
            pa.field(
                "topk_logprobs",
                pa.list_(pa.list_(pa.float32())),
                nullable=False,
            ),
        ]
    )


class ResponseAnomalyParquetWriter:
    """Write bounded, process-local Parquet shards for one model."""

    def __init__(self, runtime: Dict[str, Any]) -> None:
        self.runtime = dict(runtime)
        self.work_dir = Path(runtime["work_dir"])
        self.model_abbr = str(runtime["model_abbr"])
        self.compression = runtime.get("compression", "zstd")
        self.compression_level = int(runtime.get("compression_level", 3))
        self.write_batch_size = int(runtime.get("write_batch_size", 64))
        self.rows_per_shard = int(runtime.get("rows_per_shard", 2000))
        self.max_buffered_rows = int(runtime.get("max_buffered_rows", 256))
        self.session_id = uuid.uuid4().hex[:8]
        self.buffers: Dict[str, List[Dict[str, Any]]] = {}
        self.writers: Dict[str, Any] = {}
        self.inprogress_paths: Dict[str, Path] = {}
        self.shard_indexes: Dict[str, int] = {}
        self.shard_rows: Dict[str, int] = {}
        self.schema = payload_schema()
        self._closed = False

    def write(self, record: Dict[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("response anomaly Parquet writer is closed")
        normalized = self._normalize_record(record)
        data_abbr = normalized["data_abbr"]
        self.buffers.setdefault(data_abbr, []).append(normalized)
        if len(self.buffers[data_abbr]) >= self.write_batch_size:
            self._flush_dataset(data_abbr)
        while self._buffered_rows() > self.max_buffered_rows:
            largest = max(self.buffers, key=lambda key: len(self.buffers[key]))
            self._flush_dataset(largest)

    def close(self) -> None:
        if self._closed:
            return
        try:
            for data_abbr in list(self.buffers):
                self._flush_dataset(data_abbr)
            for data_abbr in list(self.writers):
                self._close_shard(data_abbr)
        except Exception:
            # Do not publish a shard whose final flush or validation failed.
            # Closing the raw writers releases their file handles while the
            # .inprogress suffix keeps the incomplete files out of detection.
            for writer in list(self.writers.values()):
                try:
                    writer.close()
                except Exception:
                    pass
            self.writers.clear()
            raise
        finally:
            self._closed = True

    def _normalize_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        payload = record.get("response_anomaly_payload")
        if not isinstance(payload, dict):
            raise ValueError("response_anomaly_payload must be a dict")
        tokens = payload.get("tokens")
        topk = payload.get("topk_logprobs")
        if (
            not isinstance(tokens, list)
            or not isinstance(topk, list)
            or not tokens
            or len(tokens) != len(topk)
        ):
            raise ValueError(
                "tokens and topk_logprobs must be non-empty equal-length lists"
            )
        token_ids = []
        logprobs = []
        for item in topk:
            if not isinstance(item, dict) or not item:
                raise ValueError("each topk_logprobs item must be a non-empty dict")
            pairs = [(int(token_id), float(value)) for token_id, value in item.items()]
            token_ids.append([pair[0] for pair in pairs])
            logprobs.append([pair[1] for pair in pairs])
        return {
            "schema_version": SCHEMA_VERSION,
            "data_abbr": str(record.get("data_abbr", "")),
            "model_abbr": self.model_abbr,
            "id": str(record.get("id")),
            "uuid": str(record.get("uuid", "")),
            "token_count": len(tokens),
            "tokens": [int(token) for token in tokens],
            "topk_token_ids": token_ids,
            "topk_logprobs": logprobs,
        }

    def _flush_dataset(self, data_abbr: str) -> None:
        rows = self.buffers.get(data_abbr, [])
        while rows:
            self._ensure_writer(data_abbr)
            available = self.rows_per_shard - self.shard_rows[data_abbr]
            chunk = rows[:available]
            del rows[:available]
            pa, _ = _load_pyarrow()
            table = pa.Table.from_pylist(chunk, schema=self.schema)
            self.writers[data_abbr].write_table(table, row_group_size=len(chunk))
            self.shard_rows[data_abbr] += len(chunk)
            if self.shard_rows[data_abbr] >= self.rows_per_shard:
                self._close_shard(data_abbr)

    def _ensure_writer(self, data_abbr: str) -> None:
        if data_abbr in self.writers:
            return
        _, pq = _load_pyarrow()
        shard_index = self.shard_indexes.get(data_abbr, 0)
        directory = (
            self.work_dir
            / "response_anomaly"
            / self.model_abbr
            / "payload"
            / data_abbr
        )
        directory.mkdir(parents=True, exist_ok=True)
        name = (
            f"part-p{os.getpid()}-{self.session_id}-{shard_index:05d}.parquet"
        )
        inprogress = directory / f"{name}.inprogress"
        self.inprogress_paths[data_abbr] = inprogress
        self.writers[data_abbr] = pq.ParquetWriter(
            inprogress,
            self.schema,
            compression=self.compression,
            compression_level=self.compression_level,
            use_dictionary=True,
        )
        self.shard_rows[data_abbr] = 0

    def _close_shard(self, data_abbr: str) -> None:
        writer = self.writers.pop(data_abbr, None)
        if writer is None:
            return
        writer.close()
        inprogress = self.inprogress_paths.pop(data_abbr)
        final_path = inprogress.with_suffix("")
        _, pq = _load_pyarrow()
        parquet_file = pq.ParquetFile(inprogress)
        expected_rows = self.shard_rows[data_abbr]
        if parquet_file.metadata.num_rows != expected_rows:
            raise RuntimeError(
                f"Parquet row count mismatch for {inprogress}: "
                f"expected {expected_rows}, got {parquet_file.metadata.num_rows}"
            )
        os.replace(str(inprogress), str(final_path))
        self.shard_indexes[data_abbr] = self.shard_indexes.get(data_abbr, 0) + 1
        self.shard_rows.pop(data_abbr, None)

    def _buffered_rows(self) -> int:
        return sum(len(rows) for rows in self.buffers.values())


def iter_payload_records(
    payload_dir: Path, batch_size: int = 64
) -> Iterator[Dict[str, Any]]:
    """Yield payload records from completed shards without loading all rows."""
    _, pq = _load_pyarrow()
    for shard in sorted(payload_dir.glob("part-*.parquet")):
        parquet_file = pq.ParquetFile(shard)
        row_offset = 0
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            for row_index, row in enumerate(batch.to_pylist()):
                yield {
                    "data_abbr": row["data_abbr"],
                    "model_abbr": row["model_abbr"],
                    "id": row["id"],
                    "uuid": row["uuid"],
                    "payload_shard": shard.name,
                    "payload_row": row_offset + row_index,
                    "response_anomaly_payload": {
                        "tokens": row["tokens"],
                        "topk_logprobs": [
                            {
                                int(token_id): float(logprob)
                                for token_id, logprob in zip(token_ids, values)
                            }
                            for token_ids, values in zip(
                                row["topk_token_ids"], row["topk_logprobs"]
                            )
                        ],
                    },
                }
            row_offset += batch.num_rows


def build_payload_manifest(
    payload_dir: Path, storage_cfg: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Validate completed shards and atomically write their manifest."""
    _, pq = _load_pyarrow()
    shards = []
    total_rows = 0
    model_abbr = None
    data_abbr = payload_dir.name
    for shard in sorted(payload_dir.glob("part-*.parquet")):
        parquet_file = pq.ParquetFile(shard)
        if not parquet_file.schema_arrow.equals(payload_schema()):
            raise RuntimeError(f"Unexpected response anomaly schema in {shard}")
        rows = parquet_file.metadata.num_rows
        total_rows += rows
        digest = _sha256_file(shard)
        if rows and model_abbr is None:
            first = next(parquet_file.iter_batches(batch_size=1)).to_pylist()[0]
            model_abbr = first["model_abbr"]
        shards.append(
            {
                "file": shard.name,
                "rows": rows,
                "size_bytes": shard.stat().st_size,
                "sha256": f"sha256:{digest}",
            }
        )
    storage_cfg = storage_cfg or {}
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "format": "parquet",
        "compression": storage_cfg.get("compression", "zstd"),
        "compression_level": storage_cfg.get("compression_level", 3),
        "model_abbr": model_abbr,
        "data_abbr": data_abbr,
        "total_rows": total_rows,
        "shards": shards,
    }
    payload_dir.mkdir(parents=True, exist_ok=True)
    path = payload_dir / "payload_manifest.json"
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(str(tmp_path), str(path))
    return manifest


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
