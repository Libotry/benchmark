import gzip
import json
import sys
import types
from collections import Counter

import pytest

from ais_bench.benchmark.utils.response_anomaly import ResponseAnomalyCoordinator
from ais_bench.benchmark.utils.response_anomaly_online import (
    OnlineResponseAnomalyClient,
)


class FakeDetector:
    def __init__(self, result):
        self.result = result

    def run(self, topk_logprobs, tokens, model_configs):
        return [self.result]


class RoutingDetector:
    def run(self, topk_logprobs, tokens, model_configs):
        token = tokens[0][0]
        if token == 13:
            raise RuntimeError("detector failed")
        return [[token == 12, 2 if token == 12 else 0]]


def test_detect_case_calls_msprobe_detector():
    result = ResponseAnomalyCoordinator()._detect_case(
        {
            "id": 2,
            "uuid": "case-2",
            "response_anomaly_payload": {
                "tokens": [1, 2],
                "topk_logprobs": [{"1": -0.1}, {"2": -0.2}],
            },
        },
        {"model_name": "model"},
        FakeDetector([True, 3]),
        None,
    )

    assert result["is_anomaly"] is True
    assert result["anomaly_type"] == 3
    assert result["anomaly_type_name"] == "repetition"


def test_detect_case_skips_missing_token_payload():
    result = ResponseAnomalyCoordinator()._detect_case(
        {"id": 2, "uuid": "case-2"}, {}, None, None
    )

    assert result["detection_status"] == "skipped"
    assert result["is_anomaly"] is False


def test_detect_case_skips_inconsistent_payload():
    result = ResponseAnomalyCoordinator()._detect_case(
        {
            "id": 3,
            "uuid": "case-3",
            "response_anomaly_payload": {
                "tokens": [1, 2],
                "topk_logprobs": [{"1": -0.1}],
            },
        },
        {},
        None,
        None,
    )

    assert result["detection_status"] == "skipped"
    assert "equal length" in result["reason"]


def test_detect_case_reports_detector_init_error():
    result = ResponseAnomalyCoordinator()._detect_case(
        {
            "id": 4,
            "uuid": "case-4",
            "response_anomaly_payload": {
                "tokens": [1],
                "topk_logprobs": [{"1": -0.1}],
            },
        },
        {},
        None,
        ("unavailable", "mindstudio-probe is required"),
    )

    assert result["detection_status"] == "unavailable"
    assert result["anomaly_type_name"] == "unavailable"


def test_build_detector_reports_missing_msprobe(monkeypatch):
    monkeypatch.setitem(sys.modules, "msprobe", None)

    detector, init_error = ResponseAnomalyCoordinator._build_detector({})

    assert detector is None
    assert init_error[0] == "unavailable"
    assert "mindstudio-probe" in init_error[1]


def test_build_detector_reports_ill_detector_init_failure(monkeypatch):
    msprobe_pkg = types.ModuleType("msprobe")
    response_anomaly_pkg = types.ModuleType("msprobe.response_anomaly")
    response_anomaly_pkg.__file__ = (
        "/fake/msprobe/response_anomaly/__init__.py"
    )
    detector_module = types.ModuleType("msprobe.response_anomaly.detector")

    class FailingILLDetector:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("boom")

    detector_module.ILLDetector = FailingILLDetector
    monkeypatch.setitem(sys.modules, "msprobe", msprobe_pkg)
    monkeypatch.setitem(
        sys.modules, "msprobe.response_anomaly", response_anomaly_pkg
    )
    monkeypatch.setitem(
        sys.modules, "msprobe.response_anomaly.detector", detector_module
    )

    detector, init_error = ResponseAnomalyCoordinator._build_detector({})

    assert detector is None
    assert init_error[0] == "failed"
    assert "boom" in init_error[1]


def test_merge_model_anomaly_config_prefers_model_level():
    merged = ResponseAnomalyCoordinator._merge_model_anomaly_config(
        {
            "abbr": "qwen",
            "response_anomaly": {
                "model_name": "Qwen3-30B-A3B",
                "msprobe_mtype_path": "/custom/mtype.json",
            },
        },
        {
            "enabled": True,
            "model_name": "global-name",
            "top_logprobs": 20,
            "msprobe_mtype_path": None,
        },
    )

    assert merged["model_name"] == "Qwen3-30B-A3B"
    assert merged["msprobe_mtype_path"] == "/custom/mtype.json"
    assert merged["top_logprobs"] == 20


def test_prepare_model_config_auto_generates_when_paths_missing(
    tmp_path, monkeypatch
):
    generated = {
        "msprobe_config_path": str(tmp_path / "config.yaml"),
        "msprobe_mtype_path": str(tmp_path / "mtype.json"),
        "msprobe_token2category_dir": str(tmp_path / "tk2cat"),
    }
    monkeypatch.setattr(
        "ais_bench.tools.response_anomaly.gen_model_config.generate_model_config",
        lambda **kwargs: generated,
    )

    cfg = ResponseAnomalyCoordinator()._prepare_model_config(
        "qwen",
        {"model_path": "/models/qwen", "model_name": "Qwen3-30B-A3B"},
        str(tmp_path),
    )

    assert cfg["msprobe_mtype_path"] == str(tmp_path / "mtype.json")
    assert cfg["msprobe_token2category_dir"] == str(tmp_path / "tk2cat")
    assert cfg["model_name"] == "Qwen3-30B-A3B"


def test_prepare_model_config_overwrites_none_config_path(tmp_path, monkeypatch):
    """自动生成的 config.yaml 路径应覆盖 anomaly_cfg 中的 None 值。"""
    generated = {
        "msprobe_config_path": str(tmp_path / "config.yaml"),
        "msprobe_mtype_path": str(tmp_path / "mtype.json"),
        "msprobe_token2category_dir": str(tmp_path / "tk2cat"),
    }
    monkeypatch.setattr(
        "ais_bench.tools.response_anomaly.gen_model_config.generate_model_config",
        lambda **kwargs: generated,
    )

    cfg = ResponseAnomalyCoordinator()._prepare_model_config(
        "qwen",
        {
            "model_path": "/models/qwen",
            "model_name": "Qwen3-30B-A3B",
            "msprobe_config_path": None,
        },
        str(tmp_path),
    )

    assert cfg["msprobe_config_path"] == str(tmp_path / "config.yaml")


def test_prepare_model_config_requires_both_custom_paths(tmp_path):
    with pytest.raises(RuntimeError):
        ResponseAnomalyCoordinator()._prepare_model_config(
            "qwen",
            {
                "model_path": "/models/qwen",
                "msprobe_mtype_path": "/tmp/mtype.json",
            },
            str(tmp_path),
        )


def test_post_status_is_atomic(tmp_path):
    coordinator = ResponseAnomalyCoordinator()
    status_file = tmp_path / ResponseAnomalyCoordinator.STATUS_FILE_NAME

    coordinator._post_status(
        status_file,
        completed=1,
        total=2,
        counts=Counter({"normal": 1}),
        description="detecting",
    )

    data = json.loads(status_file.read_text(encoding="utf-8"))
    assert len(data) == 1
    assert data[0]["task_name"] == "ResponseAnomaly"
    assert data[0]["finish_count"] == 1
    assert data[0]["other_kwargs"] == {"normal": 1}
    assert not status_file.with_name(status_file.name + ".tmp").exists()


def test_read_jsonl_skips_broken_lines(tmp_path):
    path = tmp_path / "pred.jsonl"
    path.write_text('{"id": 1}\nnot-json\n{"id": 2}\n', encoding="utf-8")

    records = ResponseAnomalyCoordinator()._read_jsonl(path)

    assert [item["id"] for item in records] == [1, 2]


def test_load_inherited_results_only_keeps_completed(tmp_path):
    result_file = tmp_path / "result.jsonl"
    result_file.write_text(
        json.dumps({"id": 1, "uuid": "abc", "detection_status": "completed", "anomaly_type_name": "normal"})
        + "\n"
        + json.dumps({"id": 2, "uuid": "def", "detection_status": "skipped", "anomaly_type_name": "skipped"})
        + "\n",
        encoding="utf-8",
    )

    inherited = ResponseAnomalyCoordinator()._load_inherited_results(
        result_file, {"1:abc"}
    )

    assert set(inherited) == {"1:abc"}


def test_load_inherited_results_rejects_different_uuid(tmp_path):
    """同 id 不同 uuid 的旧结果不应被继承。"""
    result_file = tmp_path / "result.jsonl"
    result_file.write_text(
        json.dumps({"id": 1, "uuid": "old-uuid", "detection_status": "completed", "anomaly_type_name": "normal"})
        + "\n",
        encoding="utf-8",
    )

    inherited = ResponseAnomalyCoordinator()._load_inherited_results(
        result_file, {"1:new-uuid"}
    )

    assert len(inherited) == 0


def test_detect_runs_full_workflow(tmp_path, monkeypatch):
    """端到端驱动 _detect 主循环：读预测、逐条检测、写结果、收尾状态。"""
    work_dir = tmp_path
    prediction_file = work_dir / "predictions" / "modelA" / "ds.jsonl"
    prediction_file.parent.mkdir(parents=True)
    prediction_file.write_text(
        json.dumps(
            {
                "id": 1,
                "uuid": "u1",
                "response_anomaly_payload": {
                    "tokens": [1],
                    "topk_logprobs": [{"1": -0.1}],
                },
            }
        )
        + "\n"
        + json.dumps(
            {
                "id": 2,
                "uuid": "u2",
                "response_anomaly_payload": {
                    "tokens": [1, 2],
                    "topk_logprobs": [{"1": -0.1}, {"2": -0.2}],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = {
        "work_dir": str(work_dir),
        "models": [{"abbr": "modelA", "attr": "service"}],
        "datasets": [{"abbr": "ds"}],
        "response_anomaly": {},
    }

    coordinator = ResponseAnomalyCoordinator()
    monkeypatch.setattr(
        coordinator,
        "_build_detector",
        lambda cfg: (FakeDetector([False, 0]), None),
    )
    coordinator._detect(cfg)

    result_lines = (
        work_dir / "response_anomaly" / "modelA" / "ds.jsonl"
    ).read_text(encoding="utf-8").strip().splitlines()
    assert len(result_lines) == 2
    results = [json.loads(line) for line in result_lines]
    assert all(item["detection_status"] == "completed" for item in results)
    assert coordinator.summary == {"normal": 2}

    status = json.loads(
        (
            work_dir / "status_tmp" / ResponseAnomalyCoordinator.STATUS_FILE_NAME
        ).read_text(encoding="utf-8")
    )[0]
    assert status["status"] == "finish"
    assert status["finish_count"] == 2
    assert status["total_count"] == 2


def test_detect_warns_when_no_predictions_found(tmp_path, monkeypatch):
    """没有任何预测样本时应告警而不是静默完成。"""
    warnings = []
    coordinator = ResponseAnomalyCoordinator()
    monkeypatch.setattr(
        coordinator,
        "_build_detector",
        lambda cfg: (FakeDetector([False, 0]), None),
    )
    monkeypatch.setattr(
        coordinator.logger,
        "warning",
        lambda msg, *args: warnings.append(msg),
    )
    cfg = {
        "work_dir": str(tmp_path),
        "models": [{"abbr": "modelA", "attr": "service"}],
        "datasets": [{"abbr": "ds"}],
        "response_anomaly": {},
    }

    coordinator._detect(cfg)

    assert coordinator.summary == {}
    status = json.loads(
        (
            tmp_path / "status_tmp" / ResponseAnomalyCoordinator.STATUS_FILE_NAME
        ).read_text(encoding="utf-8")
    )[0]
    assert status["status"] == "finish"
    assert status["total_count"] == 0
    assert any("No predictions" in message for message in warnings)


def test_detect_reads_payload_from_staging_file(tmp_path, monkeypatch):
    prediction_file = tmp_path / "predictions" / "modelA" / "ds.jsonl"
    prediction_file.parent.mkdir(parents=True)
    prediction_file.write_text(
        json.dumps({"id": 1, "uuid": "u1", "prediction": "ok"}) + "\n",
        encoding="utf-8",
    )
    payload_file = (
        tmp_path / "response_anomaly_payload" / "modelA" / "ds.jsonl"
    )
    payload_file.parent.mkdir(parents=True)
    payload_file.write_text(
        json.dumps(
            {
                "id": 1,
                "uuid": "u1",
                "response_anomaly_payload": {
                    "tokens": [1],
                    "topk_logprobs": [{"1": -0.1}],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = {
        "work_dir": str(tmp_path),
        "models": [{"abbr": "modelA", "attr": "service"}],
        "datasets": [{"abbr": "ds"}],
        "response_anomaly": {},
    }
    coordinator = ResponseAnomalyCoordinator()
    monkeypatch.setattr(
        coordinator,
        "_build_detector",
        lambda cfg: (FakeDetector([False, 0]), None),
    )

    coordinator._detect(cfg)

    result = json.loads(
        (tmp_path / "response_anomaly" / "modelA" / "ds.jsonl")
        .read_text(encoding="utf-8")
        .strip()
    )
    assert result["detection_status"] == "completed"
    assert result["token_count"] == 1


def test_finalize_creates_compact_predictions_and_diagnostics(tmp_path):
    prediction_file = tmp_path / "predictions" / "modelA" / "ds.jsonl"
    result_file = tmp_path / "response_anomaly" / "modelA" / "ds.jsonl"
    payload_file = (
        tmp_path / "response_anomaly_payload" / "modelA" / "ds.jsonl"
    )
    prediction_file.parent.mkdir(parents=True)
    result_file.parent.mkdir(parents=True)
    payload_file.parent.mkdir(parents=True)

    predictions = []
    results = []
    payloads = []
    for case_id in range(14):
        predictions.append(
            {
                "data_abbr": "ds",
                "id": case_id,
                "uuid": f"u{case_id}",
                "prediction": "ok",
            }
        )
        is_anomaly = case_id == 12
        status = "failed" if case_id == 13 else "completed"
        results.append(
            {
                "id": case_id,
                "uuid": f"u{case_id}",
                "detection_status": status,
                "is_anomaly": is_anomaly,
                "anomaly_type": 2 if is_anomaly else 0,
                "anomaly_type_name": (
                    "garbled" if is_anomaly else status if status == "failed" else "normal"
                ),
                "token_count": 1,
            }
        )
        payloads.append(
            {
                "id": case_id,
                "uuid": f"u{case_id}",
                "response_anomaly_payload": {
                    "tokens": [case_id],
                    "topk_logprobs": [{str(case_id): -0.1}],
                },
            }
        )

    prediction_file.write_text(
        "".join(json.dumps(item) + "\n" for item in predictions),
        encoding="utf-8",
    )
    result_file.write_text(
        "".join(json.dumps(item) + "\n" for item in results),
        encoding="utf-8",
    )
    payload_file.write_text(
        "".join(json.dumps(item) + "\n" for item in payloads),
        encoding="utf-8",
    )
    cfg = {
        "work_dir": str(tmp_path),
        "models": [{"abbr": "modelA", "attr": "service"}],
        "datasets": [{"abbr": "ds"}],
        "response_anomaly": {
            "enabled": True,
            "normal_sample_rate": 0.001,
            "normal_sample_min": 10,
            "normal_sample_max": 50,
            "normal_sample_seed": 7,
        },
    }
    coordinator = ResponseAnomalyCoordinator()
    coordinator._summary = {"normal": 12, "garbled": 1, "failed": 1}
    coordinator._model_manifests = {
        "modelA": {"detector_version": "1.0", "config_digest": "sha256:test"}
    }

    coordinator.finalize(cfg)

    compact_predictions = [
        json.loads(line) for line in prediction_file.read_text().splitlines()
    ]
    assert all("response_anomaly_payload" not in item for item in compact_predictions)
    assert compact_predictions[12]["response_anomaly"]["anomaly_type_name"] == "garbled"
    assert not payload_file.exists()

    with gzip.open(
        tmp_path
        / "response_anomaly"
        / "modelA"
        / "ds_abnormal_and_failed.jsonl.gz",
        "rt",
        encoding="utf-8",
    ) as file:
        abnormal = [json.loads(line) for line in file]
    with gzip.open(
        tmp_path
        / "response_anomaly"
        / "modelA"
        / "ds_normal_samples.jsonl.gz",
        "rt",
        encoding="utf-8",
    ) as file:
        normal = [json.loads(line) for line in file]
    assert {item["id"] for item in abnormal} == {12, 13}
    assert len(normal) == 10

    manifest = json.loads(
        (tmp_path / "response_anomaly" / "detector_manifest.json").read_text()
    )
    assert manifest["normal_sampling"] == {
        "rate": 0.001,
        "minimum": 10,
        "maximum": 50,
        "method": "stable_bottom_k",
        "seed": 7,
    }


def test_online_detection_releases_normal_payloads_without_staging(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        ResponseAnomalyCoordinator,
        "_build_detector",
        staticmethod(lambda cfg: (RoutingDetector(), None)),
    )
    prediction_file = tmp_path / "predictions" / "modelA" / "ds.jsonl"
    prediction_file.parent.mkdir(parents=True)
    prediction_file.write_text(
        "".join(
            json.dumps(
                {
                    "data_abbr": "ds",
                    "id": case_id,
                    "uuid": f"u{case_id}",
                    "prediction": "ok",
                }
            )
            + "\n"
            for case_id in range(14)
        ),
        encoding="utf-8",
    )
    cfg = {
        "work_dir": str(tmp_path),
        "models": [
            {
                "abbr": "modelA",
                "attr": "service",
                "response_anomaly": {"model_name": "modelA"},
            }
        ],
        "datasets": [{"abbr": "ds"}],
        "response_anomaly": {
            "enabled": True,
            "detection_mode": "online",
            "top_logprobs": 1,
            "detector_queue_size": 2,
            "detector_enqueue_timeout": 5,
            "normal_sample_rate": 0.001,
            "normal_sample_min": 10,
            "normal_sample_max": 50,
            "normal_sample_seed": 7,
        },
    }
    coordinator = ResponseAnomalyCoordinator()

    runtime = coordinator.start_online(cfg)["modelA"]
    client = OnlineResponseAnomalyClient(runtime)
    for case_id in range(14):
        client.submit(
            {
                "data_abbr": "ds",
                "id": case_id,
                "uuid": f"u{case_id}",
                "response_anomaly_payload": {
                    "tokens": [case_id],
                    "topk_logprobs": [{str(case_id): -0.1}],
                },
            }
        )
    client.close()
    coordinator.finish_online_producers()
    coordinator.join()
    coordinator.finalize(cfg)

    assert not (tmp_path / "response_anomaly_payload").exists()
    compact = [
        json.loads(line) for line in prediction_file.read_text().splitlines()
    ]
    assert compact[12]["response_anomaly"]["anomaly_type_name"] == "garbled"
    assert compact[13]["response_anomaly"]["detection_status"] == "failed"

    model_dir = tmp_path / "response_anomaly" / "modelA"
    with gzip.open(
        model_dir / "ds_normal_samples.jsonl.gz", "rt", encoding="utf-8"
    ) as file:
        normal = [json.loads(line) for line in file]
    with gzip.open(
        model_dir / "ds_abnormal_and_failed.jsonl.gz",
        "rt",
        encoding="utf-8",
    ) as file:
        abnormal = [json.loads(line) for line in file]
    assert len(normal) == 10
    assert {item["id"] for item in abnormal} == {12, 13}

    manifest = json.loads(
        (tmp_path / "response_anomaly" / "detector_manifest.json").read_text()
    )
    assert manifest["detection_mode"] == "online"
    assert manifest["runtime_metrics"]["modelA"]["accepted"] == 14
    assert manifest["runtime_metrics"]["modelA"]["max_queue_depth"] <= 2


def test_online_detection_fails_before_inference_when_detector_cannot_start(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        ResponseAnomalyCoordinator,
        "_build_detector",
        staticmethod(lambda cfg: (None, ("failed", "detector init failed"))),
    )
    cfg = {
        "work_dir": str(tmp_path),
        "models": [{"abbr": "modelA", "attr": "service"}],
        "datasets": [{"abbr": "ds"}],
        "response_anomaly": {
            "enabled": True,
            "detection_mode": "online",
            "detector_queue_size": 2,
            "detector_enqueue_timeout": 5,
        },
    }
    coordinator = ResponseAnomalyCoordinator()

    with pytest.raises(RuntimeError, match="detector init failed"):
        coordinator.start_online(cfg)

    assert coordinator._online_processes == {}
    assert coordinator._online_runtimes == {}
