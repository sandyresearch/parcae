from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import requests
import torch


class SGLangModel:
    def __init__(self, base_url: str, timeout: float, batch_size: int):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.batch_size = batch_size
        self.sessions = threading.local()

    def forward_tokens(self, input_ids: torch.Tensor, prefix_lengths=None):
        if not hasattr(self.sessions, "session"):
            self.sessions.session = requests.Session()
        for attempt in range(3):
            try:
                response = self.sessions.session.post(
                    f"{self.base_url}/generate",
                    json={
                        "input_ids": input_ids.tolist(),
                        "sampling_params": {
                            "temperature": 0,
                            "max_new_tokens": 0,
                            "ignore_eos": True,
                        },
                        "return_logprob": True,
                        "logprob_start_len": 0,
                        "top_logprobs_num": 1,
                    },
                    timeout=self.timeout,
                )
                break
            except requests.ConnectionError:
                if attempt == 2:
                    raise
                self.sessions.session = requests.Session()
        if not response.ok:
            raise RuntimeError(
                f"SGLang returned HTTP {response.status_code}: {response.text}"
            )
        outputs = response.json()
        if not isinstance(outputs, list):
            outputs = [outputs]
        if len(outputs) != input_ids.shape[0]:
            raise RuntimeError(
                f"SGLang returned {len(outputs)} outputs for batch {input_ids.shape[0]}"
            )

        losses = torch.full(input_ids.shape, float("nan"), dtype=torch.float32)
        predictions = torch.zeros_like(input_ids)
        for row, output in enumerate(outputs):
            meta = output["meta_info"]
            token_logprobs = meta["input_token_logprobs"]
            top_logprobs = meta["input_top_logprobs"]
            for index in range(1, min(input_ids.shape[1], len(token_logprobs))):
                logprob = token_logprobs[index][0]
                if logprob is not None:
                    losses[row, index - 1] = -logprob
                top = top_logprobs[index]
                if top:
                    predictions[row, index - 1] = top[0][1]
        return losses, predictions


def _open(host: str, port: int):
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def _wait(process, base_url: str, log_path: Path, timeout: float):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            lines = log_path.read_text(errors="replace").splitlines()[-40:]
            raise RuntimeError("\n".join(lines))
        try:
            if requests.post(f"{base_url}/flush_cache", timeout=2).ok:
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    raise TimeoutError(f"SGLang startup timed out; see {log_path}")


def _interrupt(*_):
    raise KeyboardInterrupt


def tokenizer_path(settings):
    value = settings.sglang.tokenizer_path or settings.tokenizer_path
    if not value:
        raise ValueError("tokenizer_path is required")
    path = Path(value).expanduser()
    if not path.is_absolute() and not path.exists():
        candidate = Path(__file__).resolve().parents[2] / path
        if candidate.exists():
            path = candidate
    return str(path if path.exists() else value)


def _shim(settings, strategy: str):
    run_dir = Path(settings.out_dir).resolve()
    root = (
        Path(settings.sglang.shim_root).expanduser().resolve()
        if settings.sglang.shim_root
        else run_dir / "eval" / "sglang_shims"
    )
    output = root / strategy
    recurrence = (
        settings.override_mean_recurrence
        or getattr(settings.model_config, "mean_recurrence", None)
        or 8
    )
    command = [
        sys.executable,
        "-m",
        "sglang_parcae.build_model_dir",
        "--run-dir",
        str(run_dir),
        "--tokenizer",
        tokenizer_path(settings),
        "--strategy",
        strategy,
        "--state-init",
        settings.sglang.state_init,
        "--t-max",
        str(settings.sglang.t_max or max(8, recurrence)),
        "--t-default",
        str(recurrence),
        "--dtype",
        {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}[
            settings.precision
        ],
        "--out",
        str(output),
    ]
    if settings.checkpoint_path:
        command.extend(["--checkpoint-path", str(settings.checkpoint_path)])
    subprocess.run(command, check=True)
    return output


@contextmanager
def managed_sglang_model(settings, strategy: str):
    config = settings.sglang
    shim = _shim(settings, strategy)
    if _open(config.host, config.port):
        raise RuntimeError(f"{config.host}:{config.port} is already in use")

    log_path = Path(settings.out_dir) / "eval" / f"sglang_{strategy}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "sglang_parcae.register",
        "--model-path",
        str(shim),
        "--host",
        config.host,
        "--port",
        str(config.port),
        "--tp-size",
        str(config.tp_size),
        "--dp-size",
        str(config.dp_size),
        "--dtype",
        "bfloat16",
        "--kv-cache-dtype",
        "bfloat16",
        "--context-length",
        str(config.context_length),
        "--cuda-graph-backend-decode",
        "full",
        "--cuda-graph-max-bs-decode",
        "64",
        "--disable-radix-cache",
        "--disable-overlap-schedule",
        "--enable-deterministic-inference",
        "--decode-attention-backend",
        "fa3" if config.mode == "exact" else "flashinfer",
    ]
    if config.mode == "turbo":
        command.extend(["--enable-torch-compile", "--torch-compile-max-bs", "56"])

    env = os.environ.copy()
    visible = env.get("CUDA_VISIBLE_DEVICES")
    world_size = config.tp_size * config.dp_size
    devices = visible.split(",") if visible else list(map(str, range(world_size)))
    if len(devices) < world_size:
        raise ValueError(
            f"tp_size * dp_size={world_size}, but only {len(devices)} GPUs are visible"
        )
    env["CUDA_VISIBLE_DEVICES"] = ",".join(devices[:world_size])
    if config.mode == "exact" and torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        if (major, minor) < (9, 0):
            print(
                f"WARNING: --sglang.mode exact uses FA3 decode attention, which needs sm_90+ "
                f"(this GPU is sm_{major}{minor}); the server will likely fail to start. "
                "Use --sglang.mode turbo on this GPU.",
                flush=True,
            )
    if config.tp_size > 1:
        env["SGLANG_PARCAE_ALLOW_UNVERIFIED_TP"] = "1"
        env["NCCL_CUMEM_ENABLE"] = "1"
    env["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"
    log = log_path.open("w")
    try:
        previous_sigterm = signal.signal(signal.SIGTERM, _interrupt)
    except ValueError:
        previous_sigterm = None
    process = None
    try:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        base_url = f"http://{config.host}:{config.port}"
        _wait(process, base_url, log_path, config.server_timeout_s)
        yield SGLangModel(
            base_url, config.request_timeout_s, settings.device_batch_size
        )
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        log.close()
        if previous_sigterm is not None:
            signal.signal(signal.SIGTERM, previous_sigterm)
