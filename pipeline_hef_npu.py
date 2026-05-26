#!/usr/bin/env python3
"""
Lab 3 - Vision Pipeline Template
Runs a Hailo-compiled YOLO model (.hef) on a Raspberry Pi 5 + Hailo AI HAT+
using the Pi Camera Module 3 as input.

Usage examples
--------------
    # A1 - live view with detections
    python3 pipeline_template.py --model yolo26n.hef --source picamera --display

    # A2 - headless FPS run (no GUI overhead)
    python3 pipeline_template.py --model yolo26n.hef --source picamera \
            --no-display --iterations 200

    # B3 - continuous run for thermal logging
    python3 pipeline_template.py --model yolo26n.hef --source picamera \
            --no-display --duration 300

Requires (on the RPi 5):
    sudo apt install hailo-all python3-picamera2
    pip install opencv-python numpy
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

from hailo_platform import (
    HEF,
    FormatType,
    VDevice,
)

try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False


COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLO vision pipeline on Hailo AI HAT+ (RPi 5)")
    parser.add_argument("--model", required=True,
                        help="Path to the compiled .hef model file")
    parser.add_argument("--source", default="picamera",
                        choices=["picamera", "video"],
                        help="Input source")
    parser.add_argument("--input", default=None,
                        help="Path to input video (used when --source=video)")
    parser.add_argument("--display", dest="display", action="store_true",
                        help="Show detections in an OpenCV window")
    parser.add_argument("--no-display", dest="display", action="store_false",
                        help="Run headless (no GUI window)")
    parser.set_defaults(display=False)
    parser.add_argument("--iterations", type=int, default=None,
                        help="Stop after N inference iterations")
    parser.add_argument("--duration", type=float, default=None,
                        help="Stop after N seconds of wall-clock time")
    parser.add_argument("--conf-threshold", type=float, default=0.4,
                        help="Confidence threshold for detections")
    parser.add_argument("--iou-threshold", type=float, default=0.45,
                        help="IoU threshold for Non-Maximum Suppression")
    parser.add_argument("--num-classes", type=int, default=80,
                        help="Number of classes in the YOLO head (COCO=80)")
    parser.add_argument("--reg-max", type=int, default=16,
                        help="DFL reg_max in the YOLO head (Ultralytics=16)")
    parser.add_argument("--capture-size", type=int, nargs=2,
                        default=(1280, 720), metavar=("W", "H"),
                        help="Pi Camera capture resolution (default 1280x720)")
    parser.add_argument("--status-file", default="/tmp/pipeline_status.json",
                        help="JSON file to write running FPS to, so external "
                             "tools (e.g. thermal_logger.py) can read it. "
                             "Pass an empty string to disable.")
    parser.add_argument("--status-interval", type=float, default=0.5,
                        help="Minimum seconds between status-file updates")
    return parser.parse_args()


def open_picamera(width: int, height: int) -> "Picamera2":
    """Start the Pi Camera in "RGB888" mode.

    Picamera2's "RGB888" format actually delivers a numpy array with channels
    in BGR order (this is documented in the picamera2 manual; the naming is
    libcamera's MSB-first convention vs. numpy's LSB-first read). The upside
    is the buffer matches OpenCV's BGR convention directly, so we can feed it
    to cv2.imshow without conversion. We only flip to true RGB just before
    the Hailo inference call, since YOLO weights expect RGB.
    """
    if not PICAMERA_AVAILABLE:
        sys.exit("picamera2 is not installed. "
                 "Install with: sudo apt install python3-picamera2")
    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={"size": (width, height), "format": "RGB888"})
    picam2.configure(config)
    picam2.start()
    time.sleep(1.0)  # let auto-exposure settle
    return picam2


def preprocess(frame_rgb: np.ndarray, target_hw: tuple) -> np.ndarray:
    """Resize to model input size and shape as (1, H, W, C) uint8."""
    h, w = target_hw
    resized = cv2.resize(frame_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
    if resized.dtype != np.uint8:
        resized = resized.astype(np.uint8)
    return np.expand_dims(resized, axis=0)


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


# Cache decoded grids so we don't rebuild them every frame
_GRID_CACHE: dict = {}


def _make_grid(h: int, w: int, stride: int):
    key = (h, w, stride)
    g = _GRID_CACHE.get(key)
    if g is not None:
        return g
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    # +0.5 → centre of each grid cell, then scale to input pixels
    cx = (xs.astype(np.float32) + 0.5) * stride
    cy = (ys.astype(np.float32) + 0.5) * stride
    g = (cx.reshape(-1), cy.reshape(-1))
    _GRID_CACHE[key] = g
    return g


def _decode_yolov10_head(raw_output: dict, num_classes: int,
                         reg_max: int, input_hw: tuple):
    """Decode the 6 raw conv outputs of a cut YOLOv10/YOLO26 head.

    Expected: 3 scales × {box-reg (C=4*reg_max), classification (C=num_classes)}
    Each tensor is shape (1, H, W, C) in NHWC (Hailo default) or (1, C, H, W)
    in NCHW. We auto-detect the channel axis by matching C to either
    4*reg_max or num_classes.

    Returns three arrays in input-pixel coords:
        xyxy   (M, 4)
        scores (M, num_classes)   (post-sigmoid)
    """
    in_h, in_w = input_hw
    box_ch = 4 * reg_max
    # NoDFL heads (e.g. YOLO26n exported by Ultralytics) emit only 4 raw
    # distance channels per cell instead of 4*reg_max DFL logits. Accept both.
    box_channels = {box_ch, 4}

    # Bucket tensors by (h, w) and by channel role.
    # Hailo FLOAT32 outputs commonly come back as (H, W, C) NHWC *without*
    # a batch axis. We also accept (1, H, W, C), (C, H, W), (1, C, H, W),
    # and even flat (H*W, C) / (C, H*W) for robustness.
    by_scale: dict = {}
    for name, t in raw_output.items():
        a = np.asarray(t)
        if a.size == 0:
            continue

        # Drop a leading batch axis of size 1 if present
        if a.ndim >= 1 and a.shape[0] == 1 and a.ndim > 1:
            # only strip if it's truly the batch axis (heuristic: ndim>=3)
            if a.ndim >= 4:
                a = a[0]

        arr = None
        valid_ch = box_channels | {num_classes}
        # 3-D layouts
        if a.ndim == 3:
            if a.shape[-1] in valid_ch:                       # (H, W, C) NHWC
                c = a.shape[-1]
                h, w = a.shape[0], a.shape[1]
                arr = a
            elif a.shape[0] in valid_ch:                      # (C, H, W) NCHW
                c = a.shape[0]
                h, w = a.shape[1], a.shape[2]
                arr = np.transpose(a, (1, 2, 0))              # → (H, W, C)
        # 2-D fallback (already flattened spatially)
        elif a.ndim == 2:
            if a.shape[-1] in valid_ch:                       # (HW, C)
                c = a.shape[-1]
                hw = a.shape[0]
                # Derive (h, w) assuming square grids (true for 640x640 input)
                side = int(round(hw ** 0.5))
                if side * side != hw:
                    continue
                h = w = side
                arr = a.reshape(h, w, c)
            elif a.shape[0] in valid_ch:                      # (C, HW)
                c = a.shape[0]
                hw = a.shape[1]
                side = int(round(hw ** 0.5))
                if side * side != hw:
                    continue
                h = w = side
                arr = a.T.reshape(h, w, c)

        if arr is None:
            continue

        slot = by_scale.setdefault((h, w), {})
        if c in box_channels:
            slot["box"] = arr
        elif c == num_classes:
            slot["cls"] = arr

    # Order scales by stride (largest grid = smallest stride)
    scales = sorted(by_scale.keys(), key=lambda hw: -hw[0] * hw[1])
    # Stride per scale: input_h / grid_h (input is square in our export)
    boxes_all = []
    scores_all = []

    # DFL projection vector [0, 1, …, reg_max-1]
    proj = np.arange(reg_max, dtype=np.float32)

    for (h, w) in scales:
        pair = by_scale[(h, w)]
        if "box" not in pair or "cls" not in pair:
            continue
        stride = int(round(in_h / h))
        box_raw = pair["box"]
        if box_raw.shape[-1] == 4:
            # NoDFL head: 4 channels are already (l, t, r, b) distances in
            # stride units. No softmax / projection needed.
            dist = box_raw.reshape(h * w, 4).astype(np.float32)
        else:
            box = box_raw.reshape(h * w, 4, reg_max)       # (HW, 4, reg_max)
            # DFL: softmax over bins, weighted sum → continuous distance
            dist = (_softmax(box, axis=-1) * proj).sum(axis=-1)   # (HW, 4)

        cx, cy = _make_grid(h, w, stride)                  # both (HW,)
        # dist = [left, top, right, bottom] in stride units
        l = dist[:, 0] * stride
        t = dist[:, 1] * stride
        r = dist[:, 2] * stride
        b = dist[:, 3] * stride
        x1 = cx - l
        y1 = cy - t
        x2 = cx + r
        y2 = cy + b
        boxes_all.append(np.stack([x1, y1, x2, y2], axis=1))

        cls_logits = pair["cls"].reshape(h * w, num_classes)
        cls_scores = 1.0 / (1.0 + np.exp(-cls_logits))     # sigmoid
        scores_all.append(cls_scores)

    if not boxes_all:
        return np.zeros((0, 4), np.float32), np.zeros((0, num_classes), np.float32)

    return (np.concatenate(boxes_all, axis=0),
            np.concatenate(scores_all, axis=0))

def postprocess(raw_output, conf_threshold: float,
                input_hw: tuple, orig_hw: tuple,
                iou_threshold: float = 0.45,
                num_classes: int = 80, reg_max: int = 16) -> list:
    """
    Decode the 6 raw conv outputs of the Hailo HEF (YOLO26/YOLOv10-style head,
    cut before TopK/GatherElements), apply sigmoid + DFL + grid decode,
    confidence filtering, and class-aware NMS.

    Input:
      raw_output : dict{name -> ndarray} of 6 tensors
                   (3 box-regression of 4*reg_max channels, 3 classification
                    of num_classes channels), in NHWC or NCHW.
    Output:
      list of detection dicts with bbox in *original frame* pixel coords.
    """
    detections: list = []

    if isinstance(raw_output, np.ndarray):
        # Single-tensor fallback (e.g. NMS baked in)
        raw_output = {"out": raw_output}

    if not isinstance(raw_output, dict) or len(raw_output) == 0:
        return detections

    boxes, scores = _decode_yolov10_head(
        raw_output, num_classes=num_classes,
        reg_max=reg_max, input_hw=input_hw)

    if boxes.shape[0] == 0:
        # One-shot warning so the user knows decoding produced nothing —
        # almost always a num_classes / reg_max mismatch or unexpected layout.
        if not getattr(postprocess, "_warned_empty", False):
            shapes = {n: tuple(np.asarray(t).shape) for n, t in raw_output.items()}
            print("[WARN] YOLOv10 decoder produced 0 candidates. "
                  f"Got outputs={shapes}. "
                  f"Expected 3 box tensors with C=4 (NoDFL) or C={4*reg_max} "
                  f"(DFL) and 3 cls tensors with C={num_classes}. "
                  "Override with --num-classes / --reg-max if needed.")
            postprocess._warned_empty = True
        return detections

    # Per-anchor best class
    class_ids = np.argmax(scores, axis=1)
    confidences = scores[np.arange(scores.shape[0]), class_ids]

    mask = confidences >= conf_threshold
    if not np.any(mask):
        return detections
    boxes = boxes[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]

    # Class-aware NMS via cv2.dnn.NMSBoxesBatched (fallback: per-class loop)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    boxes_xywh = np.stack(
        [x1, y1, x2 - x1, y2 - y1], axis=1).astype(np.float32).tolist()
    try:
        indices = cv2.dnn.NMSBoxesBatched(
            boxes_xywh, confidences.tolist(), class_ids.tolist(),
            conf_threshold, iou_threshold)
    except (AttributeError, cv2.error):
        # Older OpenCV: do per-class NMS
        indices = []
        for c in np.unique(class_ids):
            sel = np.where(class_ids == c)[0]
            sub_boxes = [boxes_xywh[i] for i in sel]
            sub_scores = confidences[sel].tolist()
            keep = cv2.dnn.NMSBoxes(
                sub_boxes, sub_scores, conf_threshold, iou_threshold)
            if keep is None or len(keep) == 0:
                continue
            indices.extend(int(sel[int(i)]) for i in np.asarray(keep).flatten())

    indices = np.asarray(indices).flatten()
    if indices.size == 0:
        return detections

    in_h, in_w = input_hw
    oh, ow = orig_hw
    sx, sy = ow / in_w, oh / in_h

    for i in indices:
        bx1 = int(np.clip(x1[i] * sx, 0, ow - 1))
        by1 = int(np.clip(y1[i] * sy, 0, oh - 1))
        bx2 = int(np.clip(x2[i] * sx, 0, ow - 1))
        by2 = int(np.clip(y2[i] * sy, 0, oh - 1))
        cid = int(class_ids[i])
        detections.append({
            "bbox": (bx1, by1, bx2, by2),
            "confidence": float(confidences[i]),
            "class_id": cid,
            "class_name": (COCO_CLASSES[cid]
                           if 0 <= cid < len(COCO_CLASSES) else str(cid)),
        })
    return detections


def draw_detections(frame_bgr: np.ndarray, detections: list,
                    fps: float) -> np.ndarray:
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = f"{det['class_name']} {det['confidence']:.2f}"
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame_bgr, label, (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(frame_bgr, f"FPS: {fps:.1f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return frame_bgr


def make_frame_reader(args: argparse.Namespace, capture_wh: tuple):
    """Return (read_fn, close_fn).

    read_fn() returns a uint8 ndarray of shape (H, W, 3) in **BGR** order
    (or None when the source ends). Both picamera2's "RGB888" format and
    cv2.VideoCapture yield BGR, which matches OpenCV's display convention —
    we convert to RGB once, just before the model.
    """
    if args.source == "picamera":
        cam = open_picamera(*capture_wh)
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


def write_status(path: str, fps: float, frames: int, elapsed: float) -> None:
    """Atomically write the current pipeline FPS so external tools
    (e.g. thermal_logger.py running in another terminal) can read it.

    Atomic rename avoids the reader catching a half-written file."""
    if not path:
        return
    try:
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({
                "fps": fps,
                "frames": frames,
                "elapsed_s": elapsed,
                "timestamp": time.time(),
            }, f)
        os.replace(tmp, path)
    except OSError:
        pass  # best-effort; never let status writes crash the pipeline


def should_stop(args: argparse.Namespace, frame_idx: int,
                start_time: float) -> bool:
    if args.iterations is not None and frame_idx >= args.iterations:
        return True
    if args.duration is not None and (time.perf_counter() - start_time) >= args.duration:
        return True
    return False


def main() -> None:
    args = parse_args()

    print(f"[INFO] Loading HEF: {args.model}")
    hef = HEF(args.model)

    input_vstream_info = hef.get_input_vstream_infos()[0]
    in_h, in_w, _ = input_vstream_info.shape
    print(f"[INFO] Model input: {in_w}x{in_h} (name={input_vstream_info.name})")

    read_frame, close_source = make_frame_reader(args, tuple(args.capture_size))

    frame_count = 0
    detections_total = 0
    start = time.perf_counter()
    last_log = start
    last_status_write = start

    try:
        with VDevice() as target:
            # Hailo-10H only supports the new Infer Model (async) API.
            # The legacy VDevice.configure(...) + InferVStreams path raises
            # HAILO_NOT_IMPLEMENTED on this device.
            infer_model = target.create_infer_model(args.model)
            infer_model.input().set_format_type(FormatType.UINT8)
            # YOLO HEFs typically expose multiple output tensors, so we must
            # configure each one by name (calling .output() with no arg only
            # works for single-output models).
            output_names = list(infer_model.output_names)
            for name in output_names:
                infer_model.output(name).set_format_type(FormatType.FLOAT32)

            with infer_model.configure() as configured_infer_model:
                input_name = infer_model.input().name

                while not should_stop(args, frame_count, start):
                    frame_bgr = read_frame()
                    if frame_bgr is None:
                        print("[INFO] Source ended")
                        break

                    orig_hw = frame_bgr.shape[:2]
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    input_tensor = preprocess(frame_rgb, (in_h, in_w))

                    bindings = configured_infer_model.create_bindings()
                    bindings.input(input_name).set_buffer(input_tensor)
                    output_buffers = {}
                    for name in output_names:
                        buf = np.empty(
                            infer_model.output(name).shape,
                            dtype=np.float32,
                        )
                        bindings.output(name).set_buffer(buf)
                        output_buffers[name] = buf

                    configured_infer_model.run([bindings], timeout=10_000)

                    raw_output = {name: output_buffers[name]
                                  for name in output_names}

                    if frame_count == 0:
                        print("[DEBUG] Raw HEF outputs (first frame):")
                        for name in output_names:
                            buf = raw_output[name]
                            print(f"  {name}: shape={tuple(buf.shape)} "
                                  f"dtype={buf.dtype} "
                                  f"min={float(buf.min()):.3f} "
                                  f"max={float(buf.max()):.3f} "
                                  f"mean={float(buf.mean()):.3f}")

                    detections = postprocess(
                        raw_output, args.conf_threshold,
                        (in_h, in_w), orig_hw,
                        iou_threshold=args.iou_threshold,
                        num_classes=args.num_classes,
                        reg_max=args.reg_max)

                    frame_count += 1
                    detections_total += len(detections)

                    elapsed = time.perf_counter() - start
                    running_fps = frame_count / elapsed if elapsed > 0 else 0.0

                    if args.display:
                        annotated = draw_detections(frame_bgr.copy(),
                                                    detections, running_fps)
                        cv2.imshow("YOLO @ Hailo", annotated)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break

                    now = time.perf_counter()
                    if (args.status_file
                            and now - last_status_write >= args.status_interval):
                        write_status(args.status_file, running_fps,
                                     frame_count, elapsed)
                        last_status_write = now

                    if now - last_log >= 2.0:
                        print(f"[INFO] frames={frame_count} "
                              f"fps={running_fps:.2f} "
                              f"dets_in_frame={len(detections)}")
                        last_log = now
    finally:
        elapsed = time.perf_counter() - start
        fps = frame_count / elapsed if elapsed > 0 else 0.0
        print()
        print(f"[RESULT] Frames processed : {frame_count}")
        print(f"[RESULT] Elapsed time     : {elapsed:.2f} s")
        print(f"[RESULT] Average FPS      : {fps:.2f}")
        print(f"[RESULT] Total detections : {detections_total}")

        try:
            close_source()
        except Exception:
            pass
        if args.display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
