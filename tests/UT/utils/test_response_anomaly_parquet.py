import json

import pyarrow.parquet as pq
import pytest

from ais_bench.benchmark.utils.response_anomaly_parquet import (
    ResponseAnomalyParquetWriter,
    build_payload_manifest,
    iter_payload_records,
)


def _record(case_id, data_abbr="ds"):
    return {
        "data_abbr": data_abbr,
        "id": case_id,
        "uuid": f"u{case_id}",
        "response_anomaly_payload": {
            "tokens": [case_id, case_id + 1],
            "topk_logprobs": [
                {str(case_id): -0.1, str(case_id + 10): -2.0},
                {str(case_id + 1): -0.2},
            ],
        },
    }


def _high_precision_record():
    return {
        "data_abbr": "ds",
        "id": 100,
        "uuid": "u100",
        "response_anomaly_payload": {
            "tokens": [100],
            "topk_logprobs": [
                {
                    "100": -0.123456789012345,
                    "101": -12.987654321098765,
                    "102": -0.000000123456789,
                }
            ],
        },
    }


def test_parquet_writer_uses_zstd_shards_and_round_trips(tmp_path):
    writer = ResponseAnomalyParquetWriter(
        {
            "work_dir": str(tmp_path),
            "model_abbr": "modelA",
            "compression": "zstd",
            "compression_level": 3,
            "write_batch_size": 2,
            "rows_per_shard": 3,
            "max_buffered_rows": 4,
        }
    )
    for case_id in range(7):
        writer.write(_record(case_id))
    writer.close()

    payload_dir = tmp_path / "response_anomaly" / "modelA" / "payload" / "ds"
    shards = sorted(payload_dir.glob("part-*.parquet"))
    assert len(shards) == 3
    assert not list(payload_dir.glob("*.inprogress"))
    assert pq.ParquetFile(shards[0]).metadata.row_group(0).column(0).compression == "ZSTD"

    records = list(iter_payload_records(payload_dir, batch_size=2))
    assert len(records) == 7
    assert records[0]["id"] == "0"
    assert records[0]["response_anomaly_payload"] == {
        "tokens": [0, 1],
        "topk_logprobs": [{0: pytest.approx(-0.1), 10: -2.0}, {1: pytest.approx(-0.2)}],
    }
    assert records[3]["payload_row"] == 0


def test_build_payload_manifest_contains_checksums(tmp_path):
    writer = ResponseAnomalyParquetWriter(
        {
            "work_dir": str(tmp_path),
            "model_abbr": "modelA",
            "write_batch_size": 1,
            "rows_per_shard": 2,
            "max_buffered_rows": 1,
        }
    )
    writer.write(_record(1))
    writer.write(_record(2))
    writer.close()
    payload_dir = tmp_path / "response_anomaly" / "modelA" / "payload" / "ds"

    manifest = build_payload_manifest(payload_dir)

    assert manifest["format"] == "parquet"
    assert manifest["compression"] == "zstd"
    assert manifest["total_rows"] == 2
    assert manifest["shards"][0]["sha256"].startswith("sha256:")
    assert json.loads(
        (payload_dir / "payload_manifest.json").read_text()
    ) == manifest


def test_parquet_writer_keeps_datasets_separate(tmp_path):
    writer = ResponseAnomalyParquetWriter(
        {
            "work_dir": str(tmp_path),
            "model_abbr": "modelA",
            "write_batch_size": 2,
            "rows_per_shard": 10,
            "max_buffered_rows": 2,
        }
    )
    writer.write(_record(1, "ds1"))
    writer.write(_record(2, "ds2"))
    writer.write(_record(3, "ds1"))
    writer.close()

    root = tmp_path / "response_anomaly" / "modelA" / "payload"
    assert len(list((root / "ds1").glob("part-*.parquet"))) == 1
    assert len(list((root / "ds2").glob("part-*.parquet"))) == 1


def test_parquet_round_trip_preserves_logprob_precision_and_order(tmp_path):
    writer = ResponseAnomalyParquetWriter(
        {
            "work_dir": str(tmp_path),
            "model_abbr": "modelA",
            "write_batch_size": 1,
            "rows_per_shard": 10,
            "max_buffered_rows": 1,
        }
    )
    original = _high_precision_record()
    writer.write(original)
    writer.close()

    payload_dir = tmp_path / "response_anomaly" / "modelA" / "payload" / "ds"
    payload = next(iter_payload_records(payload_dir))["response_anomaly_payload"]
    original_topk = original["response_anomaly_payload"]["topk_logprobs"][0]
    restored_topk = payload["topk_logprobs"][0]

    assert list(restored_topk) == [100, 101, 102]
    assert list(restored_topk.values()) == list(original_topk.values())
