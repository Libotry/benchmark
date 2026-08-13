"""Background msProbe response anomaly detection for completed AISBench predictions."""

import json
import hashlib
import importlib.metadata
import os
import threading
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

    def __init__(self) -> None:
        self.logger = AISLogger()
        self._thread: Optional[threading.Thread] = None
        self._summary: Dict[str, int] = {}

    @property
    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    @property
    def summary(self) -> Dict[str, int]:
        return dict(self._summary)

    def start(self, cfg: Dict[str, Any]) -> None:
        if self.is_running:
            return
        self._summary = {}
        self._thread = threading.Thread(
            target=self._detect,
            args=(cfg,),
            name="response-anomaly",
            daemon=False,
        )
        self._thread.start()

    def join(self) -> None:
        if self._thread:
            self._thread.join()

    def _detect(self, cfg: Dict[str, Any]) -> None:
        work_dir = cfg["work_dir"]
        status_dir = Path(work_dir) / "status_tmp"
        status_file = status_dir / self.STATUS_FILE_NAME
        total = 0
        completed = 0
        counts: Counter[str] = Counter()
        detector_manifests: Dict[str, Dict[str, Any]] = {}
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
                payload_dir = (
                    Path(work_dir)
                    / "response_anomaly"
                    / model_abbr
                    / "payload"
                    / dataset_abbr
                )
                task_groups.append(
                    (
                        model_abbr,
                        dataset_abbr,
                        model_cfg,
                        prediction_file,
                        predictions,
                        payload_dir,
                    )
                )
                total += len(predictions)

            # Predictions are produced by the inference stage; an empty set
            # means that stage produced nothing (or its output moved), so warn
            # instead of silently "finishing" with zero analyzed cases.
            for model_abbr, dataset_abbr, _, _, predictions, _ in task_groups:
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
            for (
                model_abbr,
                dataset_abbr,
                model_cfg,
                prediction_file,
                predictions,
                payload_dir,
            ) in task_groups:
                if model_abbr in detector_cache:
                    anomaly_cfg, detector, init_error, metadata = detector_cache[
                        model_abbr
                    ]
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
                    detector_manifest = self._detector_manifest(anomaly_cfg)
                    detector_manifests[model_abbr] = detector_manifest
                    metadata = {
                        "detector_version": detector_manifest["detector_version"],
                        "detector_config_digest": detector_manifest[
                            "config_digest"
                        ],
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
                prediction_by_key = {
                    self._case_key(item): item for item in predictions
                }
                detected_keys = set(inherited)
                payload_read_error = None
                try:
                    from ais_bench.benchmark.utils.response_anomaly_parquet import (
                        build_payload_manifest,
                        iter_payload_records,
                    )

                    build_payload_manifest(
                        payload_dir,
                        cfg["response_anomaly"].get("payload_storage", {}),
                    )
                    payload_records = iter_payload_records(
                        payload_dir,
                        batch_size=cfg["response_anomaly"].get(
                            "detector_read_batch_size", 64
                        ),
                    )
                except Exception as exc:
                    self.logger.error(
                        "Failed to prepare response anomaly payloads for "
                        "%s/%s: %s",
                        model_abbr,
                        dataset_abbr,
                        exc,
                    )
                    payload_records = []
                    payload_read_error = (
                        "Failed to read response anomaly Parquet payloads: "
                        f"{exc}"
                    )
                    if init_error is None:
                        init_error = ("failed", payload_read_error)

                for payload_record in payload_records:
                    case_key = self._case_key(payload_record)
                    if case_key not in prediction_by_key or case_key in detected_keys:
                        continue
                    result = self._detect_case(
                        payload_record, anomaly_cfg, detector, init_error
                    )
                    result.update(metadata)
                    result["payload_shard"] = payload_record["payload_shard"]
                    result["payload_row"] = payload_record["payload_row"]
                    safe_write({case_key: result}, result_file)
                    detected_keys.add(case_key)
                    completed += 1
                    counts[result["anomaly_type_name"]] += 1
                    self._post_status(
                        status_file,
                        completed,
                        total,
                        counts,
                        "response anomaly detecting",
                    )

                for case_key, prediction in prediction_by_key.items():
                    if case_key in detected_keys:
                        continue
                    case_id = str(prediction.get("id"))
                    if case_key in inherited:
                        continue
                    if (
                        payload_read_error is not None
                        and "response_anomaly_payload" not in prediction
                    ):
                        result = self._failed_result(
                            prediction, payload_read_error
                        )
                    else:
                        result = self._detect_case(
                            prediction, anomaly_cfg, detector, init_error
                        )
                    result.update(metadata)
                    safe_write({case_id: result}, result_file)
                    detected_keys.add(case_key)
                    completed += 1
                    counts[result["anomaly_type_name"]] += 1
                    self._post_status(
                        status_file,
                        completed,
                        total,
                        counts,
                        "response anomaly detecting",
                    )
                self._merge_results_into_predictions(prediction_file, result_file)

            self._summary = dict(counts)
            self._write_detector_manifest(
                Path(work_dir), cfg, detector_manifests, self._summary
            )
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
    def _case_key(item: Dict[str, Any]) -> str:
        return f"{item.get('id')}:{item.get('uuid')}"

    def _merge_results_into_predictions(
        self, prediction_file: Path, result_file: Path
    ) -> None:
        if not prediction_file.exists():
            return
        predictions = self._read_jsonl(prediction_file)
        results = {
            self._case_key(item): item for item in self._read_jsonl(result_file)
        }
        for prediction in predictions:
            prediction.pop("response_anomaly_payload", None)
            result = results.get(self._case_key(prediction))
            if result is not None:
                prediction["response_anomaly"] = {
                    key: value
                    for key, value in result.items()
                    if key not in ("id", "uuid")
                }
        tmp_file = prediction_file.with_name(prediction_file.name + ".tmp")
        with tmp_file.open("w", encoding="utf-8") as file:
            for prediction in predictions:
                file.write(json.dumps(prediction, ensure_ascii=False) + "\n")
        os.replace(str(tmp_file), str(prediction_file))

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

    @staticmethod
    def _failed_result(prediction: Dict[str, Any], reason: str) -> Dict[str, Any]:
        return {
            "id": prediction.get("id"),
            "uuid": prediction.get("uuid"),
            "is_anomaly": False,
            "anomaly_type": 0,
            "anomaly_type_name": "failed",
            "token_count": 0,
            "detection_status": "failed",
            "reason": reason,
        }

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

        topk_widths = [len(item) for item in topk_logprobs]
        result["topk_min"] = min(topk_widths)
        result["topk_max"] = max(topk_widths)

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
            # msProbe caches the minimum top-k width from its first request in
            # detector.topk. Parquet shards can be read in a different order
            # from prediction JSONL, so retaining that value makes detection
            # depend on which request happens to be first. Recompute it for
            # every response while still reusing the expensive detector data.
            if hasattr(detector, "topk"):
                detector.topk = None
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

    def _detector_manifest(self, anomaly_cfg: Dict[str, Any]) -> Dict[str, Any]:
        try:
            version = importlib.metadata.version("mindstudio-probe")
        except importlib.metadata.PackageNotFoundError:
            version = "unknown"
        paths = {
            "config": anomaly_cfg.get("msprobe_config_path"),
            "mtype_config": anomaly_cfg.get("msprobe_mtype_path"),
            "token2category": anomaly_cfg.get("msprobe_token2category_dir"),
        }
        digests = {
            name: self._digest_path(value) for name, value in paths.items()
        }
        digest_value = {
            "model_name": anomaly_cfg.get("model_name"),
            "top_logprobs": anomaly_cfg.get("top_logprobs"),
            "file_digests": digests,
        }
        config_digest = "sha256:" + hashlib.sha256(
            json.dumps(digest_value, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return {
            "detector_version": version,
            "model_name": anomaly_cfg.get("model_name"),
            "top_logprobs": anomaly_cfg.get("top_logprobs"),
            "paths": paths,
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
    def _write_detector_manifest(
        work_dir: Path,
        cfg: Dict[str, Any],
        models: Dict[str, Dict[str, Any]],
        summary: Dict[str, int],
    ) -> None:
        anomaly_cfg = cfg.get("response_anomaly", {})
        manifest = {
            "detector_name": "msprobe",
            "detection_mode": "post_inference",
            "models": models,
            "payload_storage": anomaly_cfg.get("payload_storage", {}),
            "detector_read_batch_size": anomaly_cfg.get(
                "detector_read_batch_size", 64
            ),
            "summary": summary,
        }
        path = work_dir / "response_anomaly" / "detector_manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_file = path.with_name(path.name + ".tmp")
        tmp_file.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(str(tmp_file), str(path))

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
