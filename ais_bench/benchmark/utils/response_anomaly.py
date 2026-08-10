"""Background msProbe response anomaly detection for completed AISBench predictions."""

import gzip
import hashlib
import importlib.metadata
import json
import math
import multiprocessing
import os
import tempfile
import threading
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ais_bench.benchmark.utils.logging import AISLogger
from ais_bench.benchmark.utils.results import safe_write


_ANOMALY_TYPE_NAMES = {
    0: "normal",
    1: "rare_character",
    2: "garbled",
    3: "repetition",
    4: "nan_value",
}


class ResponseAnomalyCoordinator:
    """Run response anomaly detection independently from the evaluation stage."""

    STATUS_TASK_NAME = "ResponseAnomaly"
    STATUS_FILE_NAME = "tmp_ResponseAnomaly.json"
    PAYLOAD_DIR_NAME = "response_anomaly_payload"
    DEFAULT_SAMPLE_RATE = 0.001
    DEFAULT_SAMPLE_MIN = 10
    DEFAULT_SAMPLE_MAX = 50
    DEFAULT_SAMPLE_SEED = 0

    def __init__(self) -> None:
        self.logger = AISLogger()
        self._thread: Optional[threading.Thread] = None
        self._summary: Dict[str, int] = {}
        self._model_manifests: Dict[str, Dict[str, Any]] = {}
        self._online_processes: Dict[str, multiprocessing.Process] = {}
        self._online_runtimes: Dict[str, Dict[str, Any]] = {}
        self._online_cfg: Optional[Dict[str, Any]] = None
        self._online_stop_sent = False
        self._monitor_stop = threading.Event()
        self._online_metrics: Dict[str, Dict[str, Any]] = {}

    @property
    def is_running(self) -> bool:
        return bool(
            (self._thread and self._thread.is_alive())
            or any(process.is_alive() for process in self._online_processes.values())
        )

    @property
    def summary(self) -> Dict[str, int]:
        return dict(self._summary)

    def start(self, cfg: Dict[str, Any]) -> None:
        if self.is_running:
            return
        self._summary = {}
        self._model_manifests = {}
        self._thread = threading.Thread(
            target=self._detect,
            args=(cfg,),
            name="response-anomaly",
            daemon=False,
        )
        self._thread.start()

    def join(self) -> None:
        if self._online_processes:
            self._join_online()
            return
        if self._thread:
            self._thread.join()

    def finalize(self, cfg: Dict[str, Any]) -> None:
        """Archive diagnostic payloads and add lightweight results to predictions."""
        if not cfg.get("response_anomaly", {}).get("enabled", False):
            return
        if cfg.get("response_anomaly", {}).get("detection_mode") == "online":
            self._finalize_online_outputs(cfg)
            return
        self._finalize_outputs(cfg)

    def start_online(self, cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Start one detector process per service model before inference."""
        if self._online_processes:
            return dict(self._online_runtimes)

        from ais_bench.benchmark.utils.response_anomaly_online import (
            run_online_detector,
        )

        self._summary = {}
        self._model_manifests = {}
        self._online_cfg = cfg
        self._online_stop_sent = False
        self._monitor_stop.clear()
        global_cfg = cfg["response_anomaly"]
        dataset_abbrs = [dataset["abbr"] for dataset in cfg.get("datasets", [])]
        started = []
        try:
            for model_cfg in cfg.get("models", []):
                if model_cfg.get("attr", "service") != "service":
                    continue
                model_abbr = model_cfg["abbr"]
                anomaly_cfg = self._merge_model_anomaly_config(
                    model_cfg, global_cfg
                )
                anomaly_cfg = self._prepare_model_config(
                    model_abbr, anomaly_cfg, cfg["work_dir"]
                )
                manifest = self._detector_manifest(anomaly_cfg)
                self._model_manifests[model_abbr] = manifest

                socket_name = "aisbench_ra_{}_{}_{}.sock".format(
                    hashlib.sha256(str(cfg["work_dir"]).encode()).hexdigest()[:8],
                    hashlib.sha256(model_abbr.encode()).hexdigest()[:8],
                    uuid.uuid4().hex[:8],
                )
                runtime = {
                    "work_dir": cfg["work_dir"],
                    "model_abbr": model_abbr,
                    "socket_path": str(Path(tempfile.gettempdir()) / socket_name),
                    "token": uuid.uuid4().hex,
                    "enqueue_timeout": global_cfg.get(
                        "detector_enqueue_timeout", 30
                    ),
                }
                server_cfg = {
                    **runtime,
                    "anomaly_cfg": anomaly_cfg,
                    "dataset_abbrs": dataset_abbrs,
                    "queue_size": global_cfg.get("detector_queue_size", 16),
                    "normal_sample_rate": global_cfg.get(
                        "normal_sample_rate", self.DEFAULT_SAMPLE_RATE
                    ),
                    "normal_sample_min": global_cfg.get(
                        "normal_sample_min", self.DEFAULT_SAMPLE_MIN
                    ),
                    "normal_sample_max": global_cfg.get(
                        "normal_sample_max", self.DEFAULT_SAMPLE_MAX
                    ),
                    "normal_sample_seed": global_cfg.get(
                        "normal_sample_seed", self.DEFAULT_SAMPLE_SEED
                    ),
                }
                ready_queue = multiprocessing.Queue(maxsize=2)
                process = multiprocessing.Process(
                    target=run_online_detector,
                    args=(server_cfg, ready_queue),
                    name=f"response-anomaly-{model_abbr}",
                    daemon=False,
                )
                process.start()
                started.append((model_abbr, process, runtime))
                ready = ready_queue.get(timeout=120)
                ready_queue.close()
                if not ready.get("ok"):
                    raise RuntimeError(
                        f"Failed to start response anomaly detector for "
                        f"'{model_abbr}': {ready.get('reason')}"
                    )
                self._online_processes[model_abbr] = process
                self._online_runtimes[model_abbr] = runtime
        except Exception:
            for _, process, runtime in started:
                if process.is_alive():
                    process.terminate()
                process.join()
                Path(runtime["socket_path"]).unlink(missing_ok=True)
            self._online_processes = {}
            self._online_runtimes = {}
            raise
        return dict(self._online_runtimes)

    def finish_online_producers(self) -> None:
        """Stop accepting new cases and let detector processes drain queues."""
        if not self._online_processes or self._online_stop_sent:
            return
        from ais_bench.benchmark.utils.response_anomaly_online import (
            request_detector_stop,
        )

        for model_abbr, runtime in self._online_runtimes.items():
            try:
                request_detector_stop(runtime)
            except Exception as exc:
                self.logger.error(
                    "Failed to stop response anomaly detector for %s: %s",
                    model_abbr,
                    exc,
                )
        self._online_stop_sent = True
        self._thread = threading.Thread(
            target=self._monitor_online,
            name="response-anomaly-monitor",
            daemon=False,
        )
        self._thread.start()

    def _join_online(self) -> None:
        if not self._online_stop_sent:
            self.finish_online_producers()
        for model_abbr, process in self._online_processes.items():
            process.join()
            if process.exitcode != 0:
                self.logger.error(
                    "Response anomaly detector for %s exited with code %s",
                    model_abbr,
                    process.exitcode,
                )
        self._ingest_undelivered()
        self._collect_online_metrics()
        self._summary = self._collect_result_summary()
        self._monitor_stop.set()
        if self._thread:
            self._thread.join()
        if self._online_cfg:
            status_file = (
                Path(self._online_cfg["work_dir"])
                / "status_tmp"
                / self.STATUS_FILE_NAME
            )
            total = sum(self._summary.values())
            self._post_status(
                status_file,
                total,
                total,
                Counter(self._summary),
                "response anomaly finished",
                "finish",
            )

    def _monitor_online(self) -> None:
        if not self._online_cfg:
            return
        status_file = (
            Path(self._online_cfg["work_dir"])
            / "status_tmp"
            / self.STATUS_FILE_NAME
        )
        while not self._monitor_stop.is_set():
            metrics = self._collect_online_metrics()
            total = sum(int(item.get("accepted", 0)) for item in metrics.values())
            completed = sum(
                int(item.get("completed", 0)) for item in metrics.values()
            )
            counts: Counter[str] = Counter()
            for item in metrics.values():
                counts.update(item.get("counts", {}))
            self._post_status(
                status_file,
                completed,
                total,
                counts,
                "response anomaly detecting",
            )
            self._monitor_stop.wait(0.3)

    def _collect_online_metrics(self) -> Dict[str, Dict[str, Any]]:
        if not self._online_cfg:
            return {}
        runtime_dir = (
            Path(self._online_cfg["work_dir"])
            / "response_anomaly"
            / ".runtime"
        )
        metrics = {}
        for model_abbr in self._online_processes:
            path = runtime_dir / f"{model_abbr}.json"
            try:
                metrics[model_abbr] = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                metrics[model_abbr] = {}
        self._online_metrics = metrics
        return metrics

    def _collect_result_summary(self) -> Dict[str, int]:
        if not self._online_cfg:
            return {}
        counts: Counter[str] = Counter()
        work_dir = Path(self._online_cfg["work_dir"])
        for model_cfg in self._online_cfg.get("models", []):
            if model_cfg.get("attr", "service") != "service":
                continue
            model_abbr = model_cfg["abbr"]
            for dataset_cfg in self._online_cfg.get("datasets", []):
                prediction_file = (
                    work_dir
                    / "predictions"
                    / model_abbr
                    / f"{dataset_cfg['abbr']}.jsonl"
                )
                prediction_keys = {
                    self._case_key(item)
                    for item in self._read_jsonl(prediction_file)
                }
                result_file = (
                    work_dir
                    / "response_anomaly"
                    / model_abbr
                    / f"{dataset_cfg['abbr']}.jsonl"
                )
                latest = {
                    self._case_key(item): item
                    for item in self._read_jsonl(result_file)
                    if self._case_key(item) in prediction_keys
                }
                counts.update(
                    item.get("anomaly_type_name", "unknown")
                    for item in latest.values()
                )
        return dict(counts)

    def _ingest_undelivered(self) -> None:
        if not self._online_cfg:
            return
        work_dir = Path(self._online_cfg["work_dir"])
        root = work_dir / "response_anomaly" / ".undelivered"
        if not root.exists():
            return
        for path in root.glob("*/*.jsonl"):
            model_abbr = path.parent.name
            dataset_abbr = path.stem
            result_file = (
                work_dir
                / "response_anomaly"
                / model_abbr
                / f"{dataset_abbr}.jsonl"
            )
            existing = {
                self._case_key(item): item
                for item in self._read_jsonl(result_file)
            }
            manifest = self._model_manifests.get(model_abbr, {})
            for record in self._read_jsonl(path):
                key = self._case_key(record)
                if existing.get(key, {}).get("detection_status") == "completed":
                    continue
                payload = record.get("response_anomaly_payload")
                result = {
                    "id": record.get("id"),
                    "uuid": record.get("uuid"),
                    "detection_status": "failed",
                    "is_anomaly": False,
                    "anomaly_type": 0,
                    "anomaly_type_name": "failed",
                    "token_count": (
                        len(payload.get("tokens", []))
                        if isinstance(payload, dict)
                        else 0
                    ),
                    "reason": record.get("reason", "Detector IPC submission failed."),
                    "detector_version": manifest.get("detector_version", "unknown"),
                    "detector_config_digest": manifest.get("config_digest"),
                }
                safe_write({key: result}, result_file)
                if isinstance(payload, dict):
                    diagnostic = {
                        "data_abbr": dataset_abbr,
                        "id": record.get("id"),
                        "uuid": record.get("uuid"),
                        "detection_result": {
                            name: value
                            for name, value in result.items()
                            if name not in ("id", "uuid")
                        },
                        "response_anomaly_payload": payload,
                    }
                    self._append_gzip_record(
                        result_file.parent
                        / f"{dataset_abbr}_abnormal_and_failed.jsonl.gz",
                        diagnostic,
                    )
            path.unlink(missing_ok=True)
        for directory in sorted(root.glob("*"), reverse=True):
            if directory.is_dir():
                try:
                    directory.rmdir()
                except OSError:
                    pass
        try:
            root.rmdir()
        except OSError:
            pass

    def _finalize_online_outputs(self, cfg: Dict[str, Any]) -> None:
        work_dir = Path(cfg["work_dir"])
        for model_cfg in cfg.get("models", []):
            if model_cfg.get("attr", "service") != "service":
                continue
            model_abbr = model_cfg["abbr"]
            for dataset_cfg in cfg.get("datasets", []):
                dataset_abbr = dataset_cfg["abbr"]
                prediction_file = (
                    work_dir / "predictions" / model_abbr / f"{dataset_abbr}.jsonl"
                )
                predictions = self._read_jsonl(prediction_file)
                if not predictions:
                    continue
                result_file = (
                    work_dir
                    / "response_anomaly"
                    / model_abbr
                    / f"{dataset_abbr}.jsonl"
                )
                results = {
                    self._case_key(item): item
                    for item in self._read_jsonl(result_file)
                }
                for prediction in predictions:
                    result = results.get(self._case_key(prediction))
                    if result is None:
                        result = {
                            "detection_status": "interrupted",
                            "is_anomaly": False,
                            "anomaly_type": 0,
                            "anomaly_type_name": "interrupted",
                            "token_count": 0,
                            "reason": "No online response anomaly result was produced.",
                        }
                    prediction.pop("response_anomaly_payload", None)
                    prediction["response_anomaly"] = {
                        name: value
                        for name, value in result.items()
                        if name not in ("id", "uuid")
                    }
                self._atomic_write_jsonl(prediction_file, predictions)

        anomaly_cfg = cfg.get("response_anomaly", {})
        manifest = {
            "detector_name": "msprobe",
            "detection_mode": "online",
            "models": self._model_manifests,
            "normal_sampling": {
                "rate": anomaly_cfg.get(
                    "normal_sample_rate", self.DEFAULT_SAMPLE_RATE
                ),
                "minimum": anomaly_cfg.get(
                    "normal_sample_min", self.DEFAULT_SAMPLE_MIN
                ),
                "maximum": anomaly_cfg.get(
                    "normal_sample_max", self.DEFAULT_SAMPLE_MAX
                ),
                "method": "stable_bottom_k",
                "seed": anomaly_cfg.get(
                    "normal_sample_seed", self.DEFAULT_SAMPLE_SEED
                ),
            },
            "runtime_metrics": self._online_metrics,
            "summary": self.summary,
        }
        self._atomic_write_json(
            work_dir / "response_anomaly" / "detector_manifest.json", manifest
        )
        runtime_dir = work_dir / "response_anomaly" / ".runtime"
        if runtime_dir.exists():
            for path in runtime_dir.glob("*.json"):
                path.unlink(missing_ok=True)
            try:
                runtime_dir.rmdir()
            except OSError:
                pass

    @staticmethod
    def _append_gzip_record(path: Path, record: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
        with path.open("ab") as raw_file:
            with gzip.GzipFile(fileobj=raw_file, mode="wb") as file:
                file.write(encoded)

    def _detect(self, cfg: Dict[str, Any]) -> None:
        work_dir = cfg["work_dir"]
        status_dir = Path(work_dir) / "status_tmp"
        status_file = status_dir / self.STATUS_FILE_NAME
        total = 0
        completed = 0
        counts: Counter[str] = Counter()
        try:
            # The feature only supports service-model generation chains.
            pairs = [
                (model["abbr"], dataset["abbr"], model)
                for model in cfg.get("models", [])
                if model.get("attr", "service") == "service"
                for dataset in cfg.get("datasets", [])
            ]

            task_groups = []
            for model_abbr, dataset_abbr, model_cfg in pairs:
                prediction_file = (
                    Path(work_dir) / "predictions" / model_abbr / f"{dataset_abbr}.jsonl"
                )
                predictions = self._read_jsonl(prediction_file)
                payloads = self._load_payloads(work_dir, model_abbr, dataset_abbr)
                for prediction in predictions:
                    key = self._case_key(prediction)
                    if (
                        "response_anomaly_payload" not in prediction
                        and key in payloads
                    ):
                        prediction["response_anomaly_payload"] = payloads[key]
                task_groups.append(
                    (model_abbr, dataset_abbr, model_cfg, prediction_file, predictions)
                )
                total += len(predictions)

            # Predictions are produced by the inference stage; an empty set
            # means that stage produced nothing (or its output moved), so warn
            # instead of silently "finishing" with zero analyzed cases.
            for model_abbr, dataset_abbr, _, _, predictions in task_groups:
                if not predictions:
                    self.logger.warning(
                        "No predictions found for model '%s' dataset '%s'; "
                        "response anomaly detection will skip this group.",
                        model_abbr,
                        dataset_abbr,
                    )
            # With at least one configured group, the per-group warnings above
            # already cover every empty group; only warn here when no group
            # was configured at all, so the run does not silently "finish".
            if not task_groups:
                self.logger.warning(
                    "Response anomaly detection has no service model/dataset "
                    "groups to analyze under %s.",
                    Path(work_dir) / "predictions",
                )

            self._post_status(
                status_file, completed, total, counts, "response anomaly detecting"
            )

            model_name_warned = False
            # Cache per-model config and detector so that a model with multiple
            # datasets only generates its msProbe config and initializes the
            # ILLDetector once (token2category loading is expensive).
            detector_cache: Dict[str, tuple] = {}
            for model_abbr, dataset_abbr, model_cfg, prediction_file, predictions in task_groups:
                if model_abbr in detector_cache:
                    anomaly_cfg, detector, init_error, metadata = detector_cache[model_abbr]
                else:
                    anomaly_cfg = self._merge_model_anomaly_config(
                        model_cfg, cfg["response_anomaly"]
                    )
                    try:
                        anomaly_cfg = self._prepare_model_config(
                            model_abbr, anomaly_cfg, work_dir
                        )
                        detector, init_error = self._build_detector(anomaly_cfg)
                    except Exception as exc:
                        self.logger.error(
                            "Failed to prepare response anomaly detection for model "
                            "%s: %s",
                            model_abbr,
                            exc,
                        )
                        detector = None
                        init_error = (
                            "failed",
                            f"Failed to prepare msProbe configuration: {exc}",
                        )
                    manifest = self._detector_manifest(anomaly_cfg)
                    self._model_manifests[model_abbr] = manifest
                    metadata = {
                        "detector_version": manifest["detector_version"],
                        "detector_config_digest": manifest["config_digest"],
                    }
                    detector_cache[model_abbr] = (
                        anomaly_cfg,
                        detector,
                        init_error,
                        metadata,
                    )
                result_file = (
                    Path(work_dir)
                    / "response_anomaly"
                    / model_abbr
                    / f"{dataset_abbr}.jsonl"
                )
                prediction_keys = {
                    f"{item.get('id')}:{item.get('uuid')}" for item in predictions
                }
                inherited = self._load_inherited_results(result_file, prediction_keys)
                completed += len(inherited)
                counts.update(
                    item.get("anomaly_type_name", "unknown")
                    for item in inherited.values()
                )

                if not model_name_warned and not anomaly_cfg.get("model_name"):
                    self.logger.warning(
                        "response_anomaly.model_name is not set; falling back to model "
                        "abbr '%s'. msProbe model matching may be degraded.",
                        model_cfg.get("abbr"),
                    )
                    model_name_warned = True

                # Ensure the result directory exists once per group instead of
                # once per prediction (mkdir with exist_ok=True is idempotent).
                result_file.parent.mkdir(parents=True, exist_ok=True)
                for prediction in predictions:
                    case_id = str(prediction.get("id"))
                    case_key = f"{prediction.get('id')}:{prediction.get('uuid')}"
                    if case_key in inherited:
                        continue
                    result = self._detect_case(
                        prediction, anomaly_cfg, detector, init_error
                    )
                    result.update(metadata)
                    safe_write({case_id: result}, result_file)
                    completed += 1
                    counts[result["anomaly_type_name"]] += 1
                    self._post_status(
                        status_file,
                        completed,
                        total,
                        counts,
                        "response anomaly detecting",
                    )

            self._summary = dict(counts)
            self._post_status(
                status_file,
                completed,
                total,
                counts,
                "response anomaly finished",
                "finish",
            )
        except Exception as exc:
            self.logger.error("Response anomaly detection failed: %s", exc)
            self._summary = dict(counts)
            self._post_status(
                status_file,
                completed,
                total,
                counts,
                f"response anomaly failed: {exc}",
                "error",
            )

    def _load_inherited_results(
        self, result_file: Path, prediction_keys: Iterable[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Return previously completed results whose id+uuid still exist in predictions.

        Matching on id+uuid ensures that a re-inferred response (different
        uuid) is not incorrectly assigned a stale anomaly result.
        Non-final statuses (skipped/unavailable/failed) are intentionally not
        inherited so they can be retried on resume.
        """
        existing_by_key: Dict[str, Dict[str, Any]] = {}
        for item in self._read_jsonl(result_file):
            key = f"{item.get('id')}:{item.get('uuid')}"
            existing_by_key[key] = item
        return {
            key: item
            for key, item in existing_by_key.items()
            if key in prediction_keys and item.get("detection_status") == "completed"
        }

    @staticmethod
    def _merge_model_anomaly_config(
        model_cfg: Dict[str, Any], global_cfg: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge global response_anomaly config with the model-level overrides."""
        merged = dict(global_cfg)
        model_cfg_anomaly = dict(model_cfg.get("response_anomaly") or {})
        for key, value in model_cfg_anomaly.items():
            if value is not None:
                merged[key] = value
        return merged

    def _prepare_model_config(
        self,
        model_abbr: str,
        anomaly_cfg: Dict[str, Any],
        work_dir: str,
    ) -> Dict[str, Any]:
        """Auto-generate msProbe model files when a local model path is given."""
        model_path = anomaly_cfg.get("model_path")
        if not model_path:
            return anomaly_cfg

        has_mtype = bool(anomaly_cfg.get("msprobe_mtype_path"))
        has_tk2cat = bool(anomaly_cfg.get("msprobe_token2category_dir"))
        if has_mtype and has_tk2cat:
            return anomaly_cfg
        if has_mtype != has_tk2cat:
            raise RuntimeError(
                "response_anomaly.msprobe_mtype_path and "
                "response_anomaly.msprobe_token2category_dir must be configured "
                "together; either provide both or rely on model_path "
                "auto-generation."
            )

        from ais_bench.tools.response_anomaly.gen_model_config import (
            generate_model_config,
        )

        output_dir = Path(work_dir) / "response_anomaly_config" / model_abbr
        generated = generate_model_config(
            model_path=str(model_path),
            model_name=anomaly_cfg.get("model_name"),
            output_dir=str(output_dir),
        )
        merged = dict(anomaly_cfg)
        for key, value in generated.items():
            # Overwrite None/empty values (e.g. msprobe_config_path set to
            # None by ConfigManager) so auto-generated paths take effect.
            if not merged.get(key):
                merged[key] = value
        return merged

    @staticmethod
    def _build_detector(anomaly_cfg: Dict[str, Any]):
        """Create one msProbe ILLDetector with the configured file paths."""
        try:
            import msprobe.response_anomaly as response_anomaly_pkg
            from msprobe.response_anomaly.detector import ILLDetector
        except ImportError:
            return None, (
                "unavailable",
                "mindstudio-probe is required for response anomaly detection. "
                "Install the AISBench response_anomaly extra.",
            )

        base = Path(response_anomaly_pkg.__file__).resolve().parent
        config_path = anomaly_cfg.get("msprobe_config_path") or str(
            base / "configs" / "config.yaml"
        )
        mtype_path = anomaly_cfg.get("msprobe_mtype_path") or str(
            base / "configs" / "mtype_config.json"
        )
        tk2cat_path = anomaly_cfg.get("msprobe_token2category_dir") or str(
            base / "token2category"
        )
        try:
            detector = ILLDetector(config_path, mtype_path, tk2cat_path)
        except Exception as exc:
            return None, (
                "failed",
                f"Failed to initialize msProbe detector: {exc}",
            )
        return detector, None

    def _detect_case(
        self,
        prediction: Dict[str, Any],
        anomaly_cfg: Dict[str, Any],
        detector=None,
        init_error=None,
    ) -> Dict[str, Any]:
        result = {
            "id": prediction.get("id"),
            "uuid": prediction.get("uuid"),
            "is_anomaly": False,
            "anomaly_type": 0,
            "anomaly_type_name": "normal",
            "token_count": 0,
        }
        payload = prediction.get("response_anomaly_payload")
        if not isinstance(payload, dict):
            result["detection_status"] = "skipped"
            result["reason"] = "Response does not contain token ids and top-k logprobs."
            result["anomaly_type_name"] = "skipped"
            return result

        tokens = payload.get("tokens")
        topk_logprobs = payload.get("topk_logprobs")
        if isinstance(tokens, list):
            result["token_count"] = len(tokens)
        if (
            not isinstance(tokens, list)
            or not isinstance(topk_logprobs, list)
            or len(tokens) == 0
            or len(tokens) != len(topk_logprobs)
            or any(not isinstance(item, dict) or not item for item in topk_logprobs)
        ):
            result["detection_status"] = "skipped"
            result["reason"] = (
                "tokens and topk_logprobs must be non-empty lists of equal length "
                "with non-empty per-token logprob maps."
            )
            result["anomaly_type_name"] = "skipped"
            return result

        if init_error is not None:
            status, reason = init_error
            result.update(
                detection_status=status,
                reason=reason,
                anomaly_type_name=status,
            )
            return result

        try:
            topk_logprobs = self._normalize_logprobs(topk_logprobs)
            tokens = [int(token) for token in tokens]
            model_name = anomaly_cfg.get("model_name")
            is_anomaly, anomaly_type = detector.run(
                [topk_logprobs], [tokens], [model_name]
            )[0]
            anomaly_type = int(anomaly_type)
            result.update(
                is_anomaly=bool(is_anomaly),
                anomaly_type=anomaly_type,
                anomaly_type_name=_ANOMALY_TYPE_NAMES.get(anomaly_type, "unknown"),
                detection_status="completed",
            )
        except Exception as exc:
            result.update(
                detection_status="failed",
                reason=f"{type(exc).__name__}: {exc}",
                anomaly_type_name="failed",
            )
        return result

    @staticmethod
    def _normalize_logprobs(items: Iterable[Dict[Any, Any]]) -> list[Dict[int, float]]:
        return [
            {int(token_id): float(logprob) for token_id, logprob in item.items()}
            for item in items
        ]

    @staticmethod
    def _case_key(item: Dict[str, Any]) -> str:
        return f"{item.get('id')}:{item.get('uuid')}"

    def _load_payloads(
        self, work_dir: str, model_abbr: str, dataset_abbr: str
    ) -> Dict[str, Dict[str, Any]]:
        """Load staging payloads and retained diagnostics for resume support."""
        root = Path(work_dir)
        paths = [
            root
            / "response_anomaly"
            / model_abbr
            / f"{dataset_abbr}_abnormal_and_failed.jsonl.gz",
            root
            / "response_anomaly"
            / model_abbr
            / f"{dataset_abbr}_normal_samples.jsonl.gz",
            root
            / self.PAYLOAD_DIR_NAME
            / model_abbr
            / f"{dataset_abbr}.jsonl",
        ]
        payloads: Dict[str, Dict[str, Any]] = {}
        for path in paths:
            records = (
                self._read_gzip_jsonl(path)
                if path.suffix == ".gz"
                else self._read_jsonl(path)
            )
            for record in records:
                payload = record.get("response_anomaly_payload")
                if isinstance(payload, dict):
                    payloads[self._case_key(record)] = payload
        return payloads

    def _finalize_outputs(self, cfg: Dict[str, Any]) -> None:
        """Create compact predictions and bounded compressed diagnostics."""
        work_dir = Path(cfg["work_dir"])
        sampling_cfg = cfg.get("response_anomaly", {})
        rate = sampling_cfg.get("normal_sample_rate", self.DEFAULT_SAMPLE_RATE)
        minimum = sampling_cfg.get("normal_sample_min", self.DEFAULT_SAMPLE_MIN)
        maximum = sampling_cfg.get("normal_sample_max", self.DEFAULT_SAMPLE_MAX)
        seed = sampling_cfg.get("normal_sample_seed", self.DEFAULT_SAMPLE_SEED)

        for model_cfg in cfg.get("models", []):
            if model_cfg.get("attr", "service") != "service":
                continue
            model_abbr = model_cfg["abbr"]
            for dataset_cfg in cfg.get("datasets", []):
                dataset_abbr = dataset_cfg["abbr"]
                prediction_file = (
                    work_dir / "predictions" / model_abbr / f"{dataset_abbr}.jsonl"
                )
                predictions = self._read_jsonl(prediction_file)
                if not predictions:
                    continue

                result_file = (
                    work_dir
                    / "response_anomaly"
                    / model_abbr
                    / f"{dataset_abbr}.jsonl"
                )
                results = {
                    self._case_key(item): item
                    for item in self._read_jsonl(result_file)
                }
                payloads = self._load_payloads(
                    str(work_dir), model_abbr, dataset_abbr
                )
                abnormal_records = []
                normal_candidates = []
                normal_count = 0

                for prediction in predictions:
                    key = self._case_key(prediction)
                    payload = prediction.pop("response_anomaly_payload", None)
                    if not isinstance(payload, dict):
                        payload = payloads.get(key)
                    result = results.get(key)
                    if result is None:
                        result = {
                            "id": prediction.get("id"),
                            "uuid": prediction.get("uuid"),
                            "detection_status": "failed",
                            "is_anomaly": False,
                            "anomaly_type": 0,
                            "anomaly_type_name": "failed",
                            "token_count": (
                                len(payload.get("tokens", [])) if payload else 0
                            ),
                            "reason": "No response anomaly result was produced.",
                        }
                    summary = {
                        key_name: value
                        for key_name, value in result.items()
                        if key_name not in ("id", "uuid")
                    }
                    prediction["response_anomaly"] = summary

                    status = result.get("detection_status")
                    is_normal = status == "completed" and not result.get("is_anomaly")
                    if is_normal:
                        normal_count += 1
                    if not isinstance(payload, dict):
                        continue
                    diagnostic = {
                        "data_abbr": prediction.get("data_abbr", dataset_abbr),
                        "id": prediction.get("id"),
                        "uuid": prediction.get("uuid"),
                        "detection_result": summary,
                        "response_anomaly_payload": payload,
                    }
                    if is_normal:
                        rank = self._sample_rank(
                            seed, model_abbr, dataset_abbr, prediction
                        )
                        normal_candidates.append((rank, diagnostic))
                    elif result.get("is_anomaly") or status != "completed":
                        abnormal_records.append(diagnostic)

                sample_count = min(
                    normal_count,
                    maximum,
                    max(minimum, math.ceil(normal_count * rate)),
                )
                normal_records = [
                    item[1]
                    for item in sorted(normal_candidates, key=lambda item: item[0])[
                        :sample_count
                    ]
                ]
                diagnostic_dir = work_dir / "response_anomaly" / model_abbr
                self._write_gzip_jsonl(
                    diagnostic_dir
                    / f"{dataset_abbr}_abnormal_and_failed.jsonl.gz",
                    abnormal_records,
                )
                self._write_gzip_jsonl(
                    diagnostic_dir / f"{dataset_abbr}_normal_samples.jsonl.gz",
                    normal_records,
                )
                self._atomic_write_jsonl(prediction_file, predictions)

                staging_file = (
                    work_dir
                    / self.PAYLOAD_DIR_NAME
                    / model_abbr
                    / f"{dataset_abbr}.jsonl"
                )
                staging_file.unlink(missing_ok=True)
                for empty_dir in (staging_file.parent, staging_file.parent.parent):
                    try:
                        empty_dir.rmdir()
                    except OSError:
                        pass

        manifest = {
            "detector_name": "msprobe",
            "models": self._model_manifests,
            "normal_sampling": {
                "rate": rate,
                "minimum": minimum,
                "maximum": maximum,
                "method": "stable_bottom_k",
                "seed": seed,
            },
            "summary": self.summary,
        }
        self._atomic_write_json(
            work_dir / "response_anomaly" / "detector_manifest.json", manifest
        )

    @staticmethod
    def _sample_rank(
        seed: Any,
        model_abbr: str,
        dataset_abbr: str,
        prediction: Dict[str, Any],
    ) -> str:
        value = ":".join(
            (
                str(seed),
                model_abbr,
                dataset_abbr,
                str(prediction.get("id")),
                str(prediction.get("uuid")),
            )
        )
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def _detector_manifest(self, anomaly_cfg: Dict[str, Any]) -> Dict[str, Any]:
        try:
            version = importlib.metadata.version("mindstudio-probe")
        except importlib.metadata.PackageNotFoundError:
            version = "unknown"

        configured_paths = {
            "config": anomaly_cfg.get("msprobe_config_path"),
            "mtype_config": anomaly_cfg.get("msprobe_mtype_path"),
            "token2category": anomaly_cfg.get("msprobe_token2category_dir"),
        }
        try:
            import msprobe.response_anomaly as response_anomaly_pkg

            base = Path(response_anomaly_pkg.__file__).resolve().parent
            configured_paths["config"] = configured_paths["config"] or str(
                base / "configs" / "config.yaml"
            )
            configured_paths["mtype_config"] = configured_paths[
                "mtype_config"
            ] or str(base / "configs" / "mtype_config.json")
            configured_paths["token2category"] = configured_paths[
                "token2category"
            ] or str(base / "token2category")
        except ImportError:
            pass

        digests = {
            name: self._digest_path(path) for name, path in configured_paths.items()
        }
        digest_input = {
            "model_name": anomaly_cfg.get("model_name"),
            "top_logprobs": anomaly_cfg.get("top_logprobs"),
            "file_digests": digests,
        }
        config_digest = "sha256:" + hashlib.sha256(
            json.dumps(digest_input, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return {
            "detector_version": version,
            "model_name": anomaly_cfg.get("model_name"),
            "top_logprobs": anomaly_cfg.get("top_logprobs"),
            "paths": configured_paths,
            "file_digests": digests,
            "config_digest": config_digest,
        }

    @staticmethod
    def _digest_path(path_value: Any) -> Optional[str]:
        if not path_value:
            return None
        path = Path(path_value)
        if not path.exists():
            return None
        digest = hashlib.sha256()
        try:
            files = [path] if path.is_file() else sorted(
                item for item in path.rglob("*") if item.is_file()
            )
            for item in files:
                if path.is_dir():
                    digest.update(str(item.relative_to(path)).encode("utf-8"))
                with item.open("rb") as file:
                    for chunk in iter(lambda: file.read(1024 * 1024), b""):
                        digest.update(chunk)
        except OSError:
            return None
        return "sha256:" + digest.hexdigest()

    @staticmethod
    def _atomic_write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_file = path.with_name(path.name + ".tmp")
        with tmp_file.open("w", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
        os.replace(str(tmp_file), str(path))

    @staticmethod
    def _atomic_write_json(path: Path, value: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_file = path.with_name(path.name + ".tmp")
        tmp_file.write_text(
            json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(str(tmp_file), str(path))

    @staticmethod
    def _write_gzip_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_file = path.with_name(path.name + ".tmp")
        with gzip.open(tmp_file, "wt", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
        os.replace(str(tmp_file), str(path))

    def _read_gzip_jsonl(self, path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            return []
        records = []
        try:
            with gzip.open(path, "rt", encoding="utf-8") as file:
                for line_no, line in enumerate(file, 1):
                    if not line.strip():
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        self.logger.warning(
                            "Skip malformed line %s:%s: %s", path, line_no, exc
                        )
        except OSError as exc:
            self.logger.warning("Unable to read diagnostic file %s: %s", path, exc)
        return records

    def _post_status(
        self,
        status_file: Path,
        completed: int,
        total: int,
        counts: Counter[str],
        description: str,
        status: str = "response anomaly",
    ) -> None:
        """Atomically write the latest status.

        The status file is replaced instead of appended, so the coordinator
        writer and TasksMonitor readers never observe partial JSON.
        """
        status_file.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {
                "task_name": self.STATUS_TASK_NAME,
                "process_id": os.getpid(),
                "finish_count": completed,
                "total_count": total,
                "progress_description": description,
                "status": status,
                "other_kwargs": dict(counts),
            }
        ]
        tmp_file = status_file.with_name(status_file.name + ".tmp")
        tmp_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(str(tmp_file), str(status_file))

    def _read_jsonl(self, path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            return []
        records = []
        with path.open(encoding="utf-8") as file:
            for line_no, line in enumerate(file, 1):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    self.logger.warning(
                        "Skip malformed line %s:%s: %s", path, line_no, exc
                    )
        return records
