#!/usr/bin/env python3
"""
Capture a real-image calibration dataset on the Raspberry Pi 5 + Hailo HAT+.

The Hailo Dataflow Compiler picks INT8 quantization scales from the
distribution of the calibration set. When that set is random noise (the
default in convert_hef_file.py), the class-logit range collapses and every
detection score is crushed near zero. Capturing ~128 real frames from the
same camera+scene fixes that.

Output
------
A single .npy file holding an NHWC float32 array, normalized to [0, 1],
ready to be passed to:

    python3 convert_hef_file.py --onnx <model>.onnx --calib-npy calib.npy ...

Usage
-----
    # Default: 128 frames at 640x640, save to calib_real.npy
    python3 capture_calib_dataset.py

    # Pick number of frames, resolution, output path, delay between captures
    python3 capture_calib_dataset.py \
        --num-frames 256 --imgsz 640 --output calib_scene.npy --delay 0.2

    # Capture from a video file instead of the Pi Camera (useful on the
    # workstation, before going to the Pi)
    python3 capture_calib_dataset.py --source video --input clip.mp4 \
        --num-frames 128 --output calib_clip.npy

Notes
-----
* Move the camera through the scene while it captures so the calibration
  set covers the brightness / texture range you expect at inference time.
* For best results, capture in the same lighting and at the same camera
  settings (resolution, exposure) as your deployment.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture a calibration .npy for Hailo HEF compilation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", default="picamera",
                        choices=["picamera", "video"],
                        help="Frame source.")
    parser.add_argument("--input", default=None,
                        help="Path to input video (only when --source=video).")
    parser.add_argument("--num-frames", type=int, default=128,
                        help="Number of frames to capture.")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Square resize size to match model input.")
    parser.add_argument("--capture-size", type=int, nargs=2,
                        default=(1280, 720), metavar=("W", "H"),
                        help="Pi Camera capture resolution.")
    parser.add_argument("--delay", type=float, default=0.1,
                        help="Seconds to wait between captures.")
    parser.add_argument("--output", type=Path, default=Path("calib_real.npy"),
                        help="Output .npy path (NHWC float32 in [0,1]).")
    parser.add_argument("--preview", action="store_true",
                        help="Show a live preview while capturing.")
    return parser.parse_args()


def open_picamera(width: int, height: int):
    if not PICAMERA_AVAILABLE:
        sys.exit("picamera2 is not installed. "
                 "Install with: sudo apt install python3-picamera2")
    cam = Picamera2()
    cam.configure(cam.create_video_configuration(
        main={"size": (width, height), "format": "RGB888"}))
    cam.start()
    time.sleep(1.0)  # let auto-exposure settle
    return cam


def make_reader(args: argparse.Namespace):
    """Return (read_fn, close_fn). read_fn returns a BGR uint8 ndarray."""
    if args.source == "picamera":
        cam = open_picamera(*args.capture_size)
        return cam.capture_array, cam.stop
    if args.source == "video":
        if not args.input:
            sys.exit("--input is required when --source=video")
        cap = cv2.VideoCapture(args.input)
        if not cap.isOpened():
            sys.exit(f"Failed to open video: {args.input}")

        def read():
            ok, bgr = cap.read()
            return bgr if ok else None

        return read, cap.release
    sys.exit(f"Unsupported source: {args.source}")


def main() -> int:
    args = parse_args()
    read_frame, close_source = make_reader(args)

    frames = []
    target_size = (args.imgsz, args.imgsz)
    print(f"[INFO] Capturing {args.num_frames} frames at "
          f"{args.imgsz}x{args.imgsz} -> {args.output}")
    print("[INFO] Move the camera through the scene for diverse calibration.")

    try:
        start = time.perf_counter()
        while len(frames) < args.num_frames:
            bgr = read_frame()
            if bgr is None:
                print("[INFO] Source ended before reaching target frame count.")
                break

            # picamera2 "RGB888" gives BGR-order ndarray; cv2.VideoCapture too
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, target_size, interpolation=cv2.INTER_LINEAR)
            frames.append(resized.astype(np.float32) / 255.0)

            if args.preview:
                cv2.imshow("calib capture (q to quit)", bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[INFO] Aborted by user (q).")
                    break

            if len(frames) % 16 == 0:
                elapsed = time.perf_counter() - start
                print(f"  captured {len(frames)}/{args.num_frames} "
                      f"({len(frames)/elapsed:.1f} fps)")

            if args.delay > 0:
                time.sleep(args.delay)
    finally:
        try:
            close_source()
        except Exception:
            pass
        if args.preview:
            cv2.destroyAllWindows()

    if not frames:
        sys.exit("No frames captured; aborting.")

    arr = np.stack(frames, axis=0)  # (N, H, W, 3) float32 in [0,1]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, arr)

    print()
    print(f"[OK] Saved {arr.shape} {arr.dtype} -> {args.output}")
    print(f"     min={float(arr.min()):.3f} "
          f"max={float(arr.max()):.3f} "
          f"mean={float(arr.mean()):.3f}")
    print("[NEXT] Transfer this file to the workstation and run "
          "convert_hef_file.py with --calib-npy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
