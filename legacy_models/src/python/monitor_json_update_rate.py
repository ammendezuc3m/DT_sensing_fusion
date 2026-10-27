#!/usr/bin/env python3
import argparse
import json
import os
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="results/online/live_inference_state.json")
    parser.add_argument("--poll", type=float, default=0.02)
    parser.add_argument("--max-lines", type=int, default=0)
    args = parser.parse_args()

    path = Path(args.file)

    last_mtime_ns = None
    last_wall = None
    last_json_ts = None
    last_sample = None
    count = 0

    print(f"Monitoring: {path}")
    print("Press Ctrl+C to stop.")
    print()
    print(
        "update | sample | label | dt_wall_ms | dt_json_ms | sample_delta | "
        "json_rate_hz | file_lag_ms | total_ms | infer_ms"
    )

    while True:
        try:
            st = path.stat()
        except FileNotFoundError:
            time.sleep(args.poll)
            continue

        if last_mtime_ns is None:
            last_mtime_ns = st.st_mtime_ns

        if st.st_mtime_ns != last_mtime_ns:
            now = time.time()
            last_mtime_ns = st.st_mtime_ns

            try:
                data = json.loads(path.read_text())
            except Exception:
                time.sleep(args.poll)
                continue

            sample = int(data.get("sample_idx", -1))
            label = data.get("inference", {}).get("stable_label", "NA")
            json_ts = float(data.get("timestamp_unix", 0.0))

            total_ms = data.get("timing", {}).get("total_ms", None)
            infer_ms = data.get("timing", {}).get("inference_ms", None)

            if last_wall is None:
                dt_wall_ms = 0.0
            else:
                dt_wall_ms = (now - last_wall) * 1000.0

            if last_json_ts is None or json_ts == 0:
                dt_json_ms = 0.0
                json_rate_hz = 0.0
            else:
                dt_json_ms = (json_ts - last_json_ts) * 1000.0
                json_rate_hz = 1000.0 / dt_json_ms if dt_json_ms > 0 else 0.0

            if last_sample is None:
                sample_delta = 0
            else:
                sample_delta = sample - last_sample

            file_lag_ms = (now - json_ts) * 1000.0 if json_ts > 0 else 0.0

            count += 1

            print(
                f"{count:06d} | "
                f"{sample:06d} | "
                f"{label:<5s} | "
                f"{dt_wall_ms:9.1f} | "
                f"{dt_json_ms:8.1f} | "
                f"{sample_delta:12d} | "
                f"{json_rate_hz:10.2f} | "
                f"{file_lag_ms:10.1f} | "
                f"{float(total_ms or 0):8.1f} | "
                f"{float(infer_ms or 0):8.1f}"
            )

            last_wall = now
            last_json_ts = json_ts
            last_sample = sample

            if args.max_lines > 0 and count >= args.max_lines:
                break

        time.sleep(args.poll)


if __name__ == "__main__":
    main()
