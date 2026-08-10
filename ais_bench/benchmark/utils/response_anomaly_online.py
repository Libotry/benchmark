"""Online response-anomaly transport and detector process."""

import gzip
import hashlib
import json
import math
import os
import pickle
import queue
import socket
import struct
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

from ais_bench.benchmark.utils.results import safe_write


_HEADER = struct.Struct("!Q")
_MAX_MESSAGE_SIZE = 1024 * 1024 * 1024


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise EOFError("response anomaly socket closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _send_message(sock: socket.socket, value: Any) -> None:
    payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(_HEADER.pack(len(payload)) + payload)


def _recv_message(sock: socket.socket) -> Any:
    size = _HEADER.unpack(_recv_exact(sock, _HEADER.size))[0]
    if size > _MAX_MESSAGE_SIZE:
        raise ValueError(f"response anomaly IPC message is too large: {size}")
    return pickle.loads(_recv_exact(sock, size))


def write_undelivered(
    runtime: Dict[str, Any], record: Dict[str, Any], reason: str
) -> None:
    """Persist a payload only when it could not reach the detector process."""
    work_dir = Path(runtime["work_dir"])
    model_abbr = runtime["model_abbr"]
    dataset_abbr = record.get("data_abbr", "unknown")
    path = (
        work_dir
        / "response_anomaly"
        / ".undelivered"
        / model_abbr
        / f"{dataset_abbr}.jsonl"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    item = dict(record)
    item["reason"] = reason
    safe_write({f"{item.get('id')}:{item.get('uuid')}": item}, path)


class OnlineResponseAnomalyClient:
    """Persistent client used by one inference output-consumer thread."""

    def __init__(self, runtime: Dict[str, Any]) -> None:
        self.runtime = dict(runtime)
        self._socket: Optional[socket.socket] = None

    def _connect(self) -> socket.socket:
        if self._socket is not None:
            return self._socket
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(float(self.runtime.get("enqueue_timeout", 30)))
        sock.connect(self.runtime["socket_path"])
        self._socket = sock
        return sock

    def submit(self, record: Dict[str, Any]) -> None:
        sock = self._connect()
        try:
            _send_message(
                sock,
                {
                    "type": "case",
                    "token": self.runtime["token"],
                    "record": record,
                },
            )
            response = _recv_message(sock)
        except Exception:
            self.close()
            raise
        if not response.get("accepted"):
            raise RuntimeError(response.get("reason", "detector rejected payload"))

    def close(self) -> None:
        sock, self._socket = self._socket, None
        if sock is None:
            return
        try:
            _send_message(
                sock,
                {"type": "close", "token": self.runtime["token"]},
            )
        except OSError:
            pass
        finally:
            sock.close()


def request_detector_stop(runtime: Dict[str, Any]) -> None:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(float(runtime.get("enqueue_timeout", 30)))
    try:
        sock.connect(runtime["socket_path"])
        _send_message(
            sock,
            {"type": "stop", "token": runtime["token"]},
        )
        _recv_message(sock)
    finally:
        sock.close()


class _OnlineDetectorServer:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        from ais_bench.benchmark.utils.response_anomaly import (
            ResponseAnomalyCoordinator,
        )

        self.cfg = cfg
        self.helper = ResponseAnomalyCoordinator()
        self.work_dir = Path(cfg["work_dir"])
        self.model_abbr = cfg["model_abbr"]
        self.anomaly_cfg = cfg["anomaly_cfg"]
        self.socket_path = Path(cfg["socket_path"])
        self.token = cfg["token"]
        self.enqueue_timeout = float(cfg.get("enqueue_timeout", 30))
        self.queue = queue.Queue(maxsize=int(cfg.get("queue_size", 16)))
        self.stop_event = threading.Event()
        self.client_threads = []
        self.client_threads_lock = threading.Lock()
        self.write_lock = threading.Lock()
        self.stats_lock = threading.Lock()
        self.accepted = 0
        self.completed = 0
        self.max_queue_depth = 0
        self.enqueue_blocked_time = 0.0
        self.counts: Counter[str] = Counter()
        self.processed = set()
        self.normal_counts = defaultdict(int)
        self.normal_candidates = defaultdict(list)
        self.detector = None
        self.metadata: Dict[str, Any] = {}

    def initialize(self) -> None:
        self.detector, init_error = self.helper._build_detector(self.anomaly_cfg)
        if init_error is not None:
            raise RuntimeError(init_error[1])
        manifest = self.helper._detector_manifest(self.anomaly_cfg)
        self.metadata = {
            "detector_version": manifest["detector_version"],
            "detector_config_digest": manifest["config_digest"],
        }
        self._load_existing_state()

    def _load_existing_state(self) -> None:
        model_dir = self.work_dir / "response_anomaly" / self.model_abbr
        for dataset_abbr in self.cfg.get("dataset_abbrs", []):
            prediction_file = (
                self.work_dir
                / "predictions"
                / self.model_abbr
                / f"{dataset_abbr}.jsonl"
            )
            prediction_keys = {
                self.helper._case_key(item)
                for item in self.helper._read_jsonl(prediction_file)
            }
            result_file = model_dir / f"{dataset_abbr}.jsonl"
            latest = {
                self.helper._case_key(item): item
                for item in self.helper._read_jsonl(result_file)
                if self.helper._case_key(item) in prediction_keys
            }
            for key, result in latest.items():
                if result.get("detection_status") != "completed":
                    continue
                self.processed.add(key)
                name = result.get("anomaly_type_name", "unknown")
                self.counts[name] += 1
                if not result.get("is_anomaly"):
                    self.normal_counts[dataset_abbr] += 1

            sample_file = model_dir / f"{dataset_abbr}_normal_samples.jsonl.gz"
            for record in self.helper._read_gzip_jsonl(sample_file):
                if self.helper._case_key(record) not in prediction_keys:
                    continue
                rank = self._sample_rank(dataset_abbr, record)
                self.normal_candidates[dataset_abbr].append((rank, record))
            self.normal_candidates[dataset_abbr].sort(key=lambda item: item[0])
            del self.normal_candidates[dataset_abbr][
                int(self.cfg.get("normal_sample_max", 50)) :
            ]

    def run(self, ready_queue) -> None:
        listener = None
        try:
            self.initialize()
            self.socket_path.parent.mkdir(parents=True, exist_ok=True)
            self.socket_path.unlink(missing_ok=True)
            listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            listener.bind(str(self.socket_path))
            os.chmod(self.socket_path, 0o600)
            listener.listen()
            listener.settimeout(0.2)
            detector_thread = threading.Thread(
                target=self._detector_loop,
                name=f"response-anomaly-{self.model_abbr}",
                daemon=False,
            )
            detector_thread.start()
            ready_queue.put({"ok": True})

            while not self.stop_event.is_set():
                try:
                    connection, _ = listener.accept()
                except socket.timeout:
                    continue
                thread = threading.Thread(
                    target=self._handle_client,
                    args=(connection,),
                    daemon=False,
                )
                with self.client_threads_lock:
                    self.client_threads.append(thread)
                thread.start()

            listener.close()
            listener = None
            with self.client_threads_lock:
                threads = list(self.client_threads)
            for thread in threads:
                thread.join()
            self.queue.put(None)
            detector_thread.join()
            self._write_normal_samples()
            self._write_status("finish")
        except Exception as exc:
            ready_queue.put({"ok": False, "reason": f"{type(exc).__name__}: {exc}"})
            raise
        finally:
            if listener is not None:
                listener.close()
            self.socket_path.unlink(missing_ok=True)

    def _handle_client(self, connection: socket.socket) -> None:
        with connection:
            while True:
                try:
                    message = _recv_message(connection)
                except (EOFError, OSError):
                    return
                if message.get("token") != self.token:
                    _send_message(connection, {"accepted": False, "reason": "invalid token"})
                    return
                message_type = message.get("type")
                if message_type == "close":
                    return
                if message_type == "stop":
                    self.stop_event.set()
                    _send_message(connection, {"accepted": True})
                    return
                if message_type != "case":
                    _send_message(connection, {"accepted": False, "reason": "invalid message"})
                    continue

                started = time.monotonic()
                try:
                    self.queue.put(message["record"], timeout=self.enqueue_timeout)
                except queue.Full:
                    _send_message(
                        connection,
                        {"accepted": False, "reason": "detector queue is full"},
                    )
                    continue
                elapsed = time.monotonic() - started
                with self.stats_lock:
                    self.accepted += 1
                    self.enqueue_blocked_time += elapsed
                    self.max_queue_depth = max(
                        self.max_queue_depth, self.queue.qsize()
                    )
                _send_message(connection, {"accepted": True})

    def _detector_loop(self) -> None:
        while True:
            record = self.queue.get()
            if record is None:
                return
            try:
                self._process_case(record)
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
                try:
                    write_undelivered(self.cfg, record, reason)
                except Exception:
                    self.stop_event.set()
                finally:
                    with self.stats_lock:
                        self.completed += 1
                        self.counts["failed"] += 1
                    self._write_status("detecting")

    def _process_case(self, record: Dict[str, Any]) -> None:
        key = self.helper._case_key(record)
        if key in self.processed:
            with self.stats_lock:
                self.completed += 1
            self._write_status("detecting")
            return

        prediction = {
            "id": record.get("id"),
            "uuid": record.get("uuid"),
            "response_anomaly_payload": record.get("response_anomaly_payload"),
        }
        result = self.helper._detect_case(
            prediction, self.anomaly_cfg, self.detector, None
        )
        result.update(self.metadata)
        dataset_abbr = record.get("data_abbr", "unknown")
        result_file = (
            self.work_dir
            / "response_anomaly"
            / self.model_abbr
            / f"{dataset_abbr}.jsonl"
        )
        result_file.parent.mkdir(parents=True, exist_ok=True)
        safe_write({key: result}, result_file)

        status = result.get("detection_status")
        is_normal = status == "completed" and not result.get("is_anomaly")
        payload = record.get("response_anomaly_payload")
        summary = {
            name: value
            for name, value in result.items()
            if name not in ("id", "uuid")
        }
        diagnostic = {
            "data_abbr": dataset_abbr,
            "id": record.get("id"),
            "uuid": record.get("uuid"),
            "detection_result": summary,
            "response_anomaly_payload": payload,
        }
        if is_normal:
            self.normal_counts[dataset_abbr] += 1
            if isinstance(payload, dict):
                self._retain_normal_candidate(dataset_abbr, diagnostic)
        elif isinstance(payload, dict) and (
            result.get("is_anomaly") or status != "completed"
        ):
            path = (
                result_file.parent
                / f"{dataset_abbr}_abnormal_and_failed.jsonl.gz"
            )
            self._append_gzip_record(path, diagnostic)

        self.processed.add(key)
        with self.stats_lock:
            self.completed += 1
            self.counts[result.get("anomaly_type_name", "unknown")] += 1
        self._write_status("detecting")

    def _retain_normal_candidate(
        self, dataset_abbr: str, diagnostic: Dict[str, Any]
    ) -> None:
        candidates = self.normal_candidates[dataset_abbr]
        candidates.append((self._sample_rank(dataset_abbr, diagnostic), diagnostic))
        candidates.sort(key=lambda item: item[0])
        maximum = int(self.cfg.get("normal_sample_max", 50))
        if len(candidates) > maximum:
            candidates.pop()

    def _sample_rank(self, dataset_abbr: str, record: Dict[str, Any]) -> str:
        value = ":".join(
            (
                str(self.cfg.get("normal_sample_seed", 0)),
                self.model_abbr,
                dataset_abbr,
                str(record.get("id")),
                str(record.get("uuid")),
            )
        )
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def _write_normal_samples(self) -> None:
        model_dir = self.work_dir / "response_anomaly" / self.model_abbr
        rate = float(self.cfg.get("normal_sample_rate", 0.001))
        minimum = int(self.cfg.get("normal_sample_min", 10))
        maximum = int(self.cfg.get("normal_sample_max", 50))
        for dataset_abbr in self.cfg.get("dataset_abbrs", []):
            normal_count = self.normal_counts[dataset_abbr]
            sample_count = min(
                normal_count,
                maximum,
                max(minimum, math.ceil(normal_count * rate)),
            )
            records = [
                item[1]
                for item in self.normal_candidates[dataset_abbr][:sample_count]
            ]
            self.helper._write_gzip_jsonl(
                model_dir / f"{dataset_abbr}_normal_samples.jsonl.gz",
                records,
            )

    def _append_gzip_record(self, path: Path, record: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
        with self.write_lock:
            with path.open("ab") as raw_file:
                with gzip.GzipFile(fileobj=raw_file, mode="wb") as file:
                    file.write(encoded)

    def _write_status(self, status: str) -> None:
        with self.stats_lock:
            value = {
                "accepted": self.accepted,
                "completed": self.completed,
                "max_queue_depth": self.max_queue_depth,
                "enqueue_blocked_time": self.enqueue_blocked_time,
                "counts": dict(self.counts),
                "status": status,
            }
        path = (
            self.work_dir
            / "response_anomaly"
            / ".runtime"
            / f"{self.model_abbr}.json"
        )
        self.helper._atomic_write_json(path, value)


def run_online_detector(cfg: Dict[str, Any], ready_queue) -> None:
    _OnlineDetectorServer(cfg).run(ready_queue)
