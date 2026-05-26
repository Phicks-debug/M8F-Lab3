#!/usr/bin/env python3
"""
Lab 3 - Thermal + FPS Logger for RPi 5 + Hailo AI HAT+

Samples CPU temperature (via `vcgencmd measure_temp`), Hailo NPU temperature
(via the HailoRT Python API: `Device.control.get_chip_temperature()`), and the
pipeline's current FPS (read from a JSON status file written by
pipeline_template.py), then writes one row per sample to a CSV.

Use alongside the pipeline in a second terminal:

    # Terminal 1 - continuous inference
    python3 pipeline_template.py --model yolo26n.hef --source picamera \\
            --no-display --duration 300

    # Terminal 2 - thermal + FPS logging
    python3 thermal_logger.py --duration 300 --interval 10 \\
            --output thermal_log.csv

CSV columns:
    timestamp_iso, elapsed_s, cpu_temp_c, hailo_temp_c, fps

At the end the script prints the summary metrics that fill in Table 2 of the
lab: peak FPS (first 30 s), sustained FPS (last 60 s), FPS drop %, and the
max CPU / Hailo temperatures observed.
"""

import argparse
import csv
import json
import os
import re
import subprocess
import time
from datetime import datetime
from typing import Optional


DEFAULT_STATUS_FILE = "/tmp/pipeline_status.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Thermal + FPS logger for RPi 5 + Hailo AI HAT+")
    p.add_argument("--duration", type=float, required=True,
                   help="Total logging duration in seconds")
    p.add_argument("--interval", type=float, default=10.0,
                   help="Sampling interval in seconds (default 10)")
    p.add_argument("--output", default="thermal_log.csv",
                   help="Output CSV path")
    p.add_argument("--status-file", default=DEFAULT_STATUS_FILE,
                   help="JSON status file written by pipeline_template.py "
                        "(default %(default)s)")
    p.add_argument("--stale-after", type=float, default=15.0,
                   help="Treat FPS as missing if the status file hasn't been "
                        "updated within this many seconds (default 15)")
    return p.parse_args()


def _run(cmd: list, timeout: float = 3.0) -> Optional[str]:
    try:
        out = subprocess.check_output(
            cmd, stderr=subprocess.STDOUT, timeout=timeout)
        return out.decode("utf-8", errors="replace")
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def read_cpu_temp_c() -> Optional[float]:
    out = _run(["vcgencmd", "measure_temp"], timeout=2.0)
    if out:
        m = re.search(r"temp=([-\d.]+)", out)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
    # Sysfs fallback: works on any Linux including RPi.
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return float(f.read().strip()) / 1000.0
    except (OSError, ValueError):
        return None


class HailoTempReader:
    """Reads Hailo NPU die temperature via the HailoRT Python API.

    The Hailo-10H exposes two on-die sensors, TS0 and TS1, through
    `Device.control.get_chip_temperature()`. We report the hotter of the two
    (the relevant figure for thermal headroom).

    A short-lived control handle is opened per sample and released immediately
    (via the `with Device(...)` context manager), so this can run in a second
    terminal alongside a benchmark/pipeline that holds the device for
    inference -- provided your HailoRT/firmware permits a concurrent control
    read. If the device can't be reached, the sample is recorded as missing
    and `last_reason` is set to explain why.

    Note: the older `hailortcli fw-control identify` text-parsing approach does
    NOT carry temperature on current firmware; the Python API is the correct
    source. `hailortcli run --measure-temp` also fails on the 10H (it needs
    `run2`), which is why we go through the API directly.
    """

    def __init__(self) -> None:
        self._device_cls = None
        self._device_info = None
        self._import_failed = False
        self.last_reason: Optional[str] = None

    def _load_device_cls(self):
        if self._device_cls is None and not self._import_failed:
            try:
                from hailo_platform import Device
                self._device_cls = Device
            except Exception as e:  # ImportError / runtime/arch errors
                self._import_failed = True
                self.last_reason = f"hailo_platform import failed: {e}"
        return self._device_cls

    def read_c(self) -> Optional[float]:
        Device = self._load_device_cls()
        if Device is None:
            return None
        try:
            if self._device_info is None:
                infos = Device.scan()
                if not infos:
                    self.last_reason = "Device.scan() found no Hailo device"
                    return None
                self._device_info = infos[0]
            with Device(self._device_info) as dev:
                t = dev.control.get_chip_temperature()
            return max(float(t.ts0_temperature), float(t.ts1_temperature))
        except Exception as e:
            # The cached device id may be stale (e.g. the device was briefly
            # busy). Clear it so the next sample re-scans instead of giving up.
            self.last_reason = f"{type(e).__name__}: {e}"
            self._device_info = None
            return None


def read_pipeline_fps(status_file: str,
                      stale_after: float) -> Optional[float]:
    try:
        st = os.stat(status_file)
    except OSError:
        return None
    if time.time() - st.st_mtime > stale_after:
        return None
    try:
        with open(status_file) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    fps = data.get("fps")
    if fps is None:
        return None
    try:
        return float(fps)
    except (TypeError, ValueError):
        return None


def _fmt(v: Optional[float], digits: int) -> str:
    return f"{v:.{digits}f}" if v is not None else ""


def summarize(rows: list) -> dict:
    """Compute the Table 2 metrics from the collected samples."""
    def window(t_lo: float, t_hi: float) -> list:
        return [r["fps"] for r in rows
                if r["fps"] is not None and t_lo <= r["elapsed"] <= t_hi]

    last_t = max((r["elapsed"] for r in rows), default=0.0)
    first_30 = window(0.0, 30.0)
    last_60 = window(max(0.0, last_t - 60.0), last_t)

    peak = max(first_30) if first_30 else None
    sustained = (sum(last_60) / len(last_60)) if last_60 else None
    drop_pct = (
        (peak - sustained) / peak * 100.0
        if (peak and sustained and peak > 0) else None
    )

    cpu_max = max(
        (r["cpu_temp"] for r in rows if r["cpu_temp"] is not None),
        default=None)
    hailo_max = max(
        (r["hailo_temp"] for r in rows if r["hailo_temp"] is not None),
        default=None)

    # Time to thermal steady-state: the first elapsed time at which CPU
    # temperature reaches within 2 C of its eventual maximum (a defensible
    # proxy for "temperature has stopped rising meaningfully").
    time_to_ss = None
    if cpu_max is not None:
        threshold = cpu_max - 2.0
        for r in rows:
            if r["cpu_temp"] is not None and r["cpu_temp"] >= threshold:
                time_to_ss = r["elapsed"]
                break

    return {
        "peak_fps_first_30s": peak,
        "sustained_fps_last_60s": sustained,
        "fps_drop_pct": drop_pct,
        "max_cpu_temp_c": cpu_max,
        "max_hailo_temp_c": hailo_max,
        "time_to_steady_state_s": time_to_ss,
    }


def main() -> None:
    args = parse_args()

    print(f"[INFO] Output         : {args.output}")
    print(f"[INFO] Duration       : {args.duration}s")
    print(f"[INFO] Interval       : {args.interval}s")
    print(f"[INFO] FPS source     : {args.status_file}")

    # Probe the NPU temperature source once up front so the operator gets
    # immediate feedback on whether a concurrent control read works while a
    # benchmark/pipeline holds the device.
    temp_reader = HailoTempReader()
    probe = temp_reader.read_c()
    if probe is None:
        print(f"[WARN] Hailo NPU temp  : unavailable ({temp_reader.last_reason})")
        print("[WARN]                   hailo(C) will be blank; "
              "CPU temp + FPS still logged.")
    else:
        print(f"[INFO] Hailo NPU temp  : HailoRT API, max(TS0,TS1); "
              f"probe={probe:.1f}C")
    print()
    print(f"{'time(s)':>8} {'cpu(C)':>8} {'hailo(C)':>10} {'fps':>7}")

    rows: list = []
    samples = 0

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_iso", "elapsed_s",
                         "cpu_temp_c", "hailo_temp_c", "fps"])

        start = time.perf_counter()
        next_sample = start
        try:
            while True:
                now = time.perf_counter()
                elapsed = now - start
                if elapsed > args.duration:
                    break
                if now < next_sample:
                    time.sleep(min(0.2, next_sample - now))
                    continue

                cpu_t = read_cpu_temp_c()
                hailo_t = temp_reader.read_c()
                fps = read_pipeline_fps(args.status_file, args.stale_after)

                writer.writerow([
                    datetime.now().isoformat(timespec="seconds"),
                    f"{elapsed:.1f}",
                    _fmt(cpu_t, 1),
                    _fmt(hailo_t, 1),
                    _fmt(fps, 2),
                ])
                f.flush()
                rows.append({
                    "elapsed": elapsed,
                    "cpu_temp": cpu_t,
                    "hailo_temp": hailo_t,
                    "fps": fps,
                })

                print(f"{elapsed:8.1f} {_fmt(cpu_t, 1):>8} "
                      f"{_fmt(hailo_t, 1):>10} {_fmt(fps, 2):>7}")

                samples += 1
                next_sample = start + samples * args.interval
        except KeyboardInterrupt:
            print("\n[INFO] Interrupted; finalizing CSV...")

    print()
    print(f"[DONE] Wrote {samples} samples to {args.output}")

    s = summarize(rows)
    print()
    print("=== Summary (Table 2 inputs) ===")
    print(f"  Peak FPS (first 30s)       : "
          f"{_fmt(s['peak_fps_first_30s'], 2) or 'n/a'}")
    print(f"  Sustained FPS (last 60s)   : "
          f"{_fmt(s['sustained_fps_last_60s'], 2) or 'n/a'}")
    print(f"  FPS drop (%)               : "
          f"{_fmt(s['fps_drop_pct'], 1) or 'n/a'}")
    print(f"  Max CPU temperature (C)    : "
          f"{_fmt(s['max_cpu_temp_c'], 1) or 'n/a'}")
    print(f"  Max Hailo temperature (C)  : "
          f"{_fmt(s['max_hailo_temp_c'], 1) or 'n/a'}")
    print(f"  Time to thermal steady-state (s): "
          f"{_fmt(s['time_to_steady_state_s'], 1) or 'n/a'}")


if __name__ == "__main__":
    main()
