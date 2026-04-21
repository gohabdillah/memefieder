from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import requests


DEFAULT_FEATURE_DIM = 333


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a simple concurrent load test against cloud /infer."
    )
    parser.add_argument(
        "--url",
        default="http://localhost:5000",
        help="Cloud base URL (default: http://localhost:5000).",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=200,
        help="Total number of requests to send (default: 200).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=20,
        help="Number of concurrent worker threads (default: 20).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=2.0,
        help="Request timeout in seconds (default: 2.0).",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Optional path to write JSON summary report.",
    )
    parser.add_argument(
        "--device-prefix",
        default="load-client",
        help="Device ID prefix used in payloads (default: load-client).",
    )
    return parser.parse_args()


def get_feature_dim(base_url: str, timeout: float) -> int:
    try:
        response = requests.get(f"{base_url}/health", timeout=timeout)
        response.raise_for_status()
        data = response.json()
        dim = int(data.get("expected_feature_dim", DEFAULT_FEATURE_DIM))
        return max(1, dim)
    except (requests.RequestException, ValueError, TypeError):
        return DEFAULT_FEATURE_DIM


def single_request(
    base_url: str,
    timeout: float,
    keypoint_dim: int,
    device_prefix: str,
    index: int,
) -> dict[str, Any]:
    payload = {
        "keypoints": np.random.random(keypoint_dim).astype(float).tolist(),
        "device_id": f"{device_prefix}-{index % 50}",
    }

    start = time.perf_counter()
    try:
        response = requests.post(f"{base_url}/infer", json=payload, timeout=timeout)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        status_ok = response.status_code == 200

        data: dict[str, Any] = {}
        try:
            data = response.json() if response.content else {}
        except ValueError:
            data = {}

        has_prediction = isinstance(data.get("meme"), str)
        return {
            "ok": bool(status_ok and has_prediction),
            "status_code": response.status_code,
            "latency_ms": elapsed_ms,
            "error": "" if status_ok else str(data.get("error", "request failed")),
        }
    except requests.RequestException as error:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return {
            "ok": False,
            "status_code": 0,
            "latency_ms": elapsed_ms,
            "error": str(error),
        }


def summarize(results: list[dict[str, Any]], total_seconds: float) -> dict[str, Any]:
    latencies = np.array([r["latency_ms"] for r in results], dtype=float)
    success_count = sum(1 for r in results if r["ok"])
    total = len(results)

    summary = {
        "total_requests": total,
        "success_count": success_count,
        "success_rate": round((success_count / total) * 100.0, 2) if total else 0.0,
        "duration_seconds": round(total_seconds, 3),
        "throughput_rps": round(total / total_seconds, 2) if total_seconds > 0 else 0.0,
        "latency_ms": {
            "min": round(float(np.min(latencies)), 2) if total else 0.0,
            "p50": round(float(np.percentile(latencies, 50)), 2) if total else 0.0,
            "p95": round(float(np.percentile(latencies, 95)), 2) if total else 0.0,
            "p99": round(float(np.percentile(latencies, 99)), 2) if total else 0.0,
            "max": round(float(np.max(latencies)), 2) if total else 0.0,
        },
        "error_samples": [r["error"] for r in results if not r["ok"]][:5],
    }
    return summary


def main() -> None:
    args = parse_args()
    base_url = args.url.rstrip("/")

    total_requests = max(1, args.requests)
    workers = max(1, min(args.workers, total_requests))
    timeout = max(0.1, args.timeout)

    keypoint_dim = get_feature_dim(base_url, timeout)

    print(f"Target: {base_url}/infer")
    print(f"Requests: {total_requests}, workers: {workers}, timeout: {timeout:.2f}s")
    print(f"Using keypoint dim: {keypoint_dim}")

    start = time.perf_counter()
    results: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                single_request,
                base_url,
                timeout,
                keypoint_dim,
                args.device_prefix,
                idx,
            )
            for idx in range(total_requests)
        ]
        for future in as_completed(futures):
            results.append(future.result())

    duration = time.perf_counter() - start
    summary = summarize(results, duration)

    print("\nLoad Test Summary")
    print("-----------------")
    print(f"Success rate: {summary['success_rate']}% ({summary['success_count']}/{summary['total_requests']})")
    print(f"Throughput: {summary['throughput_rps']} req/s")
    print(
        "Latency ms: "
        f"min={summary['latency_ms']['min']}, "
        f"p50={summary['latency_ms']['p50']}, "
        f"p95={summary['latency_ms']['p95']}, "
        f"p99={summary['latency_ms']['p99']}, "
        f"max={summary['latency_ms']['max']}"
    )

    if summary["error_samples"]:
        print("Sample errors:")
        for err in summary["error_samples"]:
            print(f"- {err}")

    if args.report_path is not None:
        args.report_path.parent.mkdir(parents=True, exist_ok=True)
        with args.report_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        print(f"Saved report to {args.report_path}")


if __name__ == "__main__":
    main()
