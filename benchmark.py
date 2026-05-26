#!/usr/bin/env python3
"""
Lab 3 - Per-stage benchmark for Hailo AI HAT+ on RPi 5.

Measures the latency of each pipeline stage with high-precision counters
(time.perf_counter), reports p50/p95/p99 percentiles, and produces the
inputs for both:

- Part B / B1 - per-stage latency breakdown table
- Part C / C1 - multi-model FPS + p50 / p95 table

Examples
--------
    # B1 - full per-stage profiling with the Pi Camera
    python3 benchmark.py --model yolo26n.hef --source picamera \\
            --iterations 1000 --warmup 100 --profile

    # B1 - reproducibility check
    python3 benchmark.py --model yolo26n.hef --source picamera \\
            --iterations 500 --warmup 100 --profile

    # B1 - no warm-up comparison
    python3 benchmark.py --model yolo26n.hef --source picamera \\
            --no-warmup --iterations 200 --profile

    # C1 - multi-model comparison (synthetic input for reproducibility)
    python3 benchmark.py --model mobilenetv2.hef     --iterations 1000 --warmup 100
    python3 benchmark.py --model efficientnet_b0.hef --iterations 1000 --warmup 100
    python3 benchmark.py --model yolo26n.hef         --iterations 1000 --warmup 100
    python3 benchmark.py --model mobileclip_s2.hef   --iterations 1000 --warmup 100

Pass --output-csv c1_summary.csv to all four C1 runs to get one
aggregated table you can paste straight into the lab.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from hailo_platform import (
    HEF,
    FormatType,
    VDevice,
)

# NOTE (Hailo-10H): the legacy synchronous pipeline
# (ConfigureParams + VDevice.configure() + InferVStreams) raises
# HAILO_NOT_IMPLEMENTED on Hailo-10H. We use the new InferModel async
# API, which is supported on both Hailo-8 and Hailo-10H. The --interface
# CLI flag is therefore informational only and no longer passed to the SDK.

try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Per-stage Hailo benchmark for the Lab 3 profiling tables")
    p.add_argument("--model", required=True, help="Path to .hef file")
    p.add_argument("--source", default="synthetic",
                   choices=["synthetic", "picamera"],
                   help="Input source. 'synthetic' = pseudo-random uint8 frames "
                        "(reproducible, no camera); 'picamera' = real Pi Camera "
                        "ISP capture (needed for the Camera-capture row of B1).")
    p.add_argument("--iterations", type=int, default=1000,
                   help="Number of timed iterations after warm-up")
    p.add_argument("--warmup", type=int, default=100,
                   help="Number of warm-up iterations")
    p.add_argument("--no-warmup", dest="warmup", action="store_const", const=0,
                   help="Skip warm-up entirely (B1 cold-start comparison)")
    p.add_argument("--profile", action="store_true",
                   help="Print the full B1 per-stage breakdown (otherwise just "
                        "the C1 summary line)")
    p.add_argument("--conf-threshold", type=float, default=0.4,
                   help="Confidence threshold used during the NMS post-process "
                        "timing step")
    p.add_argument("--output-csv", default=None,
                   help="Append the C1 summary row to this CSV "
                        "(creates header on first run)")
    p.add_argument("--capture-size", type=int, nargs=2,
                   default=(1280, 720), metavar=("W", "H"),
                   help="Pi Camera capture resolution (only used with "
                        "--source picamera)")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for synthetic frames")
    p.add_argument("--interface", default="auto",
                   choices=["auto", "pcie", "integrated", "eth"],
                   help="HailoRT stream interface. 'auto' lets the SDK pick "
                        "(recommended for Hailo-10H). Use 'pcie' for Hailo-8 "
                        "M.2 modules, 'integrated' for HAT+ form-factor devices.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Frame sources
# ---------------------------------------------------------------------------

def open_picamera(width: int, height: int):
    if not PICAMERA_AVAILABLE:
        sys.exit("picamera2 is not installed "
                 "(sudo apt install python3-picamera2)")
    cam = Picamera2()
    cfg = cam.create_video_configuration(
        main={"size": (width, height), "format": "RGB888"})
    cam.configure(cfg)
    cam.start()
    time.sleep(1.0)
    return cam


# ---------------------------------------------------------------------------
# Generic post-process (NMS-like) for timing
# ---------------------------------------------------------------------------

def postprocess_timing(raw, conf_threshold: float) -> int:
    """Touch every element of the raw output and apply a confidence filter.

    For Hailo YOLO with NMS-on-NPU the output is already decoded; for
    classification models it's a flat logit vector. Either way we want a
    realistic NMS / argmax-style cost in the timing, not a no-op.
    """
    if isinstance(raw, dict):
        if not raw:
            return 0
        raw = next(iter(raw.values()))

    arr = np.asarray(raw)
    if arr.size == 0:
        return 0

    # Hailo NMS layout: per-class arrays of (n_det, 5).
    if arr.dtype == object or (arr.ndim >= 2 and arr.shape[-1] == 5):
        if arr.ndim == 3:
            it = arr[0]
        elif arr.ndim == 2 and arr.shape[-1] == 5:
            it = [arr]
        else:
            it = arr
        kept = 0
        for dets in it:
            if dets is None:
                continue
            d = np.asarray(dets)
            if d.size == 0 or d.ndim < 2:
                continue
            kept += int((d[:, 4] >= conf_threshold).sum())
        return kept

    # Classification fallback: argmax + softmax-style normalization.
    flat = arr.reshape(arr.shape[0], -1) if arr.ndim > 1 else arr.reshape(1, -1)
    exp = np.exp(flat - flat.max(axis=1, keepdims=True))
    prob = exp / exp.sum(axis=1, keepdims=True)
    return int(np.argmax(prob, axis=1)[0])


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def stats(values: list) -> dict:
    if not values:
        return {"p50": float("nan"), "p95": float("nan"),
                "p99": float("nan"), "mean": float("nan")}
    a = np.asarray(values)
    return {
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "mean": float(a.mean()),
    }


def fmt_row(name: str, s, on: str, width: int = 28) -> str:
    if s is None:
        return f"{name:<{width}} {'—':>10} {'—':>10} {'—':>10}  {on}"
    return (f"{name:<{width}} {s['p50']:>10.3f} {s['p95']:>10.3f} "
            f"{s['p99']:>10.3f}  {on}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    model_path = args.model
    model_name = Path(model_path).name
    hef_mb = os.path.getsize(model_path) / (1024 * 1024)

    print(f"[INFO] Model       : {model_name}  ({hef_mb:.2f} MB)")
    print(f"[INFO] Source      : {args.source}")
    print(f"[INFO] Interface   : {args.interface}")
    print(f"[INFO] Iterations  : {args.iterations}  (warm-up={args.warmup})")

    hef = HEF(model_path)
    input_info = hef.get_input_vstream_infos()[0]
    in_h, in_w, _ = input_info.shape
    print(f"[INFO] Input shape : {in_h}x{in_w}")

    cam = None
    if args.source == "picamera":
        cam = open_picamera(*args.capture_size)
        cap_h, cap_w = args.capture_size[1], args.capture_size[0]
    else:
        cap_h, cap_w = max(in_h, 720), max(in_w, 1280)
    rng = np.random.default_rng(args.seed)
    synthetic_frame = rng.integers(0, 256, (cap_h, cap_w, 3), dtype=np.uint8)

    def grab_frame_with_time():
        """Returns (bgr_frame, capture_seconds)."""
        if cam is not None:
            t0 = time.perf_counter()
            f = cam.capture_array()  # BGR per picamera2 "RGB888" quirk
            return f, time.perf_counter() - t0
        # Synthetic: pretend-capture (just a memcpy). We still time it so the
        # B1 capture row has a real number, but it isn't representative ISP
        # latency — that's why --source picamera is the lab default for B1.
        t0 = time.perf_counter()
        f = synthetic_frame.copy()
        return f, time.perf_counter() - t0

    t_capture: list = []
    t_preprocess: list = []
    t_infer: list = []
    t_postprocess: list = []
    t_total: list = []

    # Per-iteration timeout for run() in milliseconds. Big enough to never
    # be the bottleneck (HailoRT default is also ~10 s).
    INFER_TIMEOUT_MS = 10_000

    try:
        with VDevice() as vdevice:
            # --- New InferModel API (Hailo-10H compatible) ----------------
            infer_model = vdevice.create_infer_model(model_path)
            infer_model.set_batch_size(1)
            infer_model.input().set_format_type(FormatType.UINT8)
            for out_name in infer_model.output_names:
                infer_model.output(out_name).set_format_type(FormatType.FLOAT32)

            with infer_model.configure() as configured:
                # Pre-allocate buffers once and reuse them every iteration.
                input_name = infer_model.input_names[0]
                model_in_shape = infer_model.input(input_name).shape  # (H,W,C)
                input_buf = np.empty(model_in_shape, dtype=np.uint8)

                output_bufs = {}
                for out_name in infer_model.output_names:
                    out_shape = infer_model.output(out_name).shape
                    output_bufs[out_name] = np.empty(out_shape, dtype=np.float32)

                bindings = configured.create_bindings()
                bindings.input(input_name).set_buffer(input_buf)
                for out_name, buf in output_bufs.items():
                    bindings.output(out_name).set_buffer(buf)

                def run_once() -> dict:
                    configured.run([bindings], INFER_TIMEOUT_MS)
                    return output_bufs

                if args.warmup > 0:
                    print(f"[INFO] Warm-up   : {args.warmup} iterations")
                    for _ in range(args.warmup):
                        f, _ = grab_frame_with_time()
                        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                        np.copyto(input_buf, cv2.resize(rgb, (in_w, in_h)))
                        run_once()

                print(f"[INFO] Timed run : {args.iterations} iterations")
                report_every = max(1, args.iterations // 10)
                for i in range(args.iterations):
                    iter_start = time.perf_counter()

                    f, cap_dt = grab_frame_with_time()
                    t_capture.append(cap_dt * 1000.0)

                    t0 = time.perf_counter()
                    rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                    resized = cv2.resize(rgb, (in_w, in_h))
                    np.copyto(input_buf, resized)
                    t_preprocess.append((time.perf_counter() - t0) * 1000.0)

                    t0 = time.perf_counter()
                    raw = run_once()
                    t_infer.append((time.perf_counter() - t0) * 1000.0)

                    t0 = time.perf_counter()
                    _ = postprocess_timing(raw, args.conf_threshold)
                    t_postprocess.append((time.perf_counter() - t0) * 1000.0)

                    t_total.append((time.perf_counter() - iter_start) * 1000.0)

                    if (i + 1) % report_every == 0:
                        running_fps = (i + 1) / (sum(t_total) / 1000.0)
                        print(f"[INFO] {i + 1:5d}/{args.iterations}  "
                              f"running fps: {running_fps:.2f}")
    finally:
        if cam is not None:
            cam.stop()

    s_capture = stats(t_capture)
    s_preprocess = stats(t_preprocess)
    s_infer = stats(t_infer)
    s_postprocess = stats(t_postprocess)
    s_total = stats(t_total)

    fps_p50 = 1000.0 / s_total["p50"] if s_total["p50"] > 0 else 0.0
    fps_mean = 1000.0 / s_total["mean"] if s_total["mean"] > 0 else 0.0

    # -----------------------------------------------------------------
    # Part B / B1: per-stage breakdown
    # -----------------------------------------------------------------
    if args.profile:
        print()
        print(f"=== B1 per-stage breakdown ({model_name}) ===")
        print(f"{'Pipeline Stage':<28} {'p50(ms)':>10} {'p95(ms)':>10} "
              f"{'p99(ms)':>10}  Runs On")
        # Only show the capture row when it's a real ISP measurement.
        capture_display = s_capture if args.source == "picamera" else None
        print(fmt_row("Camera capture", capture_display, "ISP"))
        print(fmt_row("Resize + normalize", s_preprocess, "CPU"))
        print(fmt_row("Host->NPU transfer", None, "PCIe"))
        print(fmt_row("NPU inference", s_infer, "Hailo"))
        print(fmt_row("NPU->Host transfer", None, "PCIe"))
        print(fmt_row("NMS post-processing", s_postprocess, "CPU"))
        print(f"{'Total':<28} {s_total['p50']:>10.3f} "
              f"{s_total['p95']:>10.3f} {s_total['p99']:>10.3f}")
        print(f"{'Effective FPS':<28} {fps_p50:>10.2f}")
        print()
        print("NOTE: HailoRT's high-level InferVStreams API exposes Host->NPU,")
        print("      NPU compute, and NPU->Host as one blocking infer() call,")
        print("      so the 'NPU inference' row is the combined wall-clock for")
        print("      all three. The 'Host->NPU transfer' and 'NPU->Host transfer'")
        print("      rows are reported as '—'. To separate them, enable Hailo")
        print("      hardware-latency measurement and subtract from the total.")

    # -----------------------------------------------------------------
    # Part C / C1: one-line summary
    # -----------------------------------------------------------------
    print()
    print(f"=== C1 summary ({model_name}) ===")
    print(f"  .hef size (MB)         : {hef_mb:.2f}")
    print(f"  Total p50 (ms)         : {s_total['p50']:.3f}")
    print(f"  Total p95 (ms)         : {s_total['p95']:.3f}")
    print(f"  Total p99 (ms)         : {s_total['p99']:.3f}")
    print(f"  Mean latency (ms)      : {s_total['mean']:.3f}")
    print(f"  FPS (1000 / p50)       : {fps_p50:.2f}")
    print(f"  FPS (1000 / mean)      : {fps_mean:.2f}")

    if args.output_csv:
        write_header = not os.path.exists(args.output_csv)
        with open(args.output_csv, "a") as f:
            if write_header:
                f.write("model,hef_mb,iterations,warmup,source,"
                        "p50_ms,p95_ms,p99_ms,mean_ms,fps_p50,fps_mean\n")
            f.write(f"{model_name},{hef_mb:.2f},{args.iterations},"
                    f"{args.warmup},{args.source},"
                    f"{s_total['p50']:.3f},{s_total['p95']:.3f},"
                    f"{s_total['p99']:.3f},{s_total['mean']:.3f},"
                    f"{fps_p50:.2f},{fps_mean:.2f}\n")
        print(f"[INFO] Appended summary row to {args.output_csv}")


if __name__ == "__main__":
    main()
