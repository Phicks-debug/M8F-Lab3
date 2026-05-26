#!/usr/bin/env python3
"""
Lab 3 - Vision Pipeline (ONNX / CPU)
Runs a YOLO model exported to ONNX on the Raspberry Pi CPU.
Drop-in replacement for pipeline_template.py — same CLI, no Hailo required.

Supported ONNX export formats
------------------------------
  YOLOv8 / v9  : output shape (1, 4+num_classes, num_anchors)   — no objectness
  YOLOv5 / v7  : output shape (1, num_anchors, 5+num_classes)   — with objectness

Usage examples
--------------
    # live view with detections
    python3 pipeline_onnx_cpu.py --model yolov8n.onnx --source picamera --display

    # headless FPS run
    python3 pipeline_onnx_cpu.py --model yolov8n.onnx --source picamera \
            --no-display --iterations 200

    # continuous run for thermal logging
    python3 pipeline_onnx_cpu.py --model yolov8n.onnx --source picamera \
            --no-display --duration 300

    # run on a video file
    python3 pipeline_onnx_cpu.py --model yolov8n.onnx --source video \
            --input sample.mp4 --display

Requires (on the RPi 5):
    pip install onnxruntime opencv-python numpy
    sudo apt install python3-picamera2   # for --source picamera
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import onnxruntime as ort

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
        description="YOLO vision pipeline on CPU via ONNX Runtime (RPi 5)")
    parser.add_argument("--model", required=True,
                        help="Path to the exported .onnx model file")
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
    parser.add_argument("--capture-size", type=int, nargs=2,
                        default=(1280, 720), metavar=("W", "H"),
                        help="Pi Camera capture resolution (default 1280x720)")
    parser.add_argument("--num-threads", type=int, default=4,
                        help="Number of CPU threads for ONNX Runtime (default 4)")
    parser.add_argument("--status-file", default="/tmp/pipeline_status.json",
                        help="JSON file to write running FPS to, so external "
                             "tools (e.g. thermal_logger.py) can read it. "
                             "Pass an empty string to disable.")
    parser.add_argument("--status-interval", type=float, default=0.5,
                        help="Minimum seconds between status-file updates")
    return parser.parse_args()


def build_session(model_path: str, num_threads: int) -> ort.InferenceSession:
    """Create a CPU-only ONNX Runtime session."""
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = num_threads
    opts.inter_op_num_threads = num_threads
    # SPEED_OPTIMIZED gives best latency on single-frame inference
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        model_path,
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )
    return session


def open_picamera(width: int, height: int) -> "Picamera2":
    """Start the Pi Camera in "RGB888" mode.

    Picamera2's "RGB888" format delivers a numpy array in BGR order
    (libcamera MSB-first vs. numpy LSB-first convention).  The buffer
    matches OpenCV's BGR convention so we can pass it to cv2.imshow
    directly; we only flip to RGB just before the ONNX inference call.
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


def preprocess(frame_bgr: np.ndarray, target_hw: tuple) -> np.ndarray:
    """Prepare a BGR frame for ONNX YOLO inference.

    Returns float32 NCHW tensor normalised to [0, 1] — the standard
    format expected by YOLOv5 / v8 ONNX exports.
    """
    h, w = target_hw
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
    blob = resized.astype(np.float32) / 255.0          # HWC [0,1]
    blob = np.transpose(blob, (2, 0, 1))               # CHW
    blob = np.expand_dims(blob, axis=0)                # NCHW
    return blob


def _xywh2xyxy(cx: np.ndarray, cy: np.ndarray,
               w: np.ndarray, h: np.ndarray):
    """Convert centre-format boxes to corner format (vectorised)."""
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return x1, y1, x2, y2


def postprocess(raw_output: np.ndarray, conf_threshold: float,
                iou_threshold: float, input_hw: tuple,
                orig_hw: tuple) -> list:
    """
    Decode ONNX YOLO output and apply NMS.

    Handles three layouts (auto-detected by shape):
      YOLOv10 / v26 end-to-end — (1, N, 6) : [x1,y1,x2,y2,score,cls], NMS-free
      YOLOv8 / v9              — (1, 4+C, N) : no explicit objectness score
      YOLOv5 / v7              — (1, N, 5+C) : includes objectness score
    All box coordinates are in pixels relative to the model input size.
    """
    detections: list = []

    # Take first (and usually only) output tensor
    if isinstance(raw_output, (list, tuple)):
        arr = np.asarray(raw_output[0])
    else:
        arr = np.asarray(raw_output)

    if arr.size == 0:
        return detections

    # Remove batch dimension → shape becomes (4+C, N), (N, 5+C) or (N, 6)
    arr = arr[0]

    in_h, in_w = input_hw
    oh, ow = orig_hw
    scale_x = ow / in_w
    scale_y = oh / in_h

    if arr.ndim == 2 and arr.shape[1] == 6:
        # ---- YOLOv10 / YOLO26 end-to-end layout: (N, 6) ----
        # Columns are [x1, y1, x2, y2, score, class_id] already decoded
        # and top-k filtered by the model. NMS-free → no need to run NMS.
        confidences = arr[:, 4]
        sel = np.where(confidences >= conf_threshold)[0]
        if sel.size == 0:
            return detections
        x1 = arr[sel, 0]
        y1 = arr[sel, 1]
        x2 = arr[sel, 2]
        y2 = arr[sel, 3]
        confidences = arr[sel, 4]
        class_ids = arr[sel, 5].astype(np.int32)

        for i in range(sel.size):
            bx1 = int(x1[i] * scale_x)
            by1 = int(y1[i] * scale_y)
            bx2 = int(x2[i] * scale_x)
            by2 = int(y2[i] * scale_y)
            cid = int(class_ids[i])
            detections.append({
                "bbox": (bx1, by1, bx2, by2),
                "confidence": float(confidences[i]),
                "class_id": cid,
                "class_name": (COCO_CLASSES[cid]
                               if 0 <= cid < len(COCO_CLASSES) else str(cid)),
            })
        return detections

    if arr.ndim == 2 and arr.shape[0] < arr.shape[1]:
        # ---- YOLOv8 / v9 layout: (4+C, N) ----
        arr = arr.T                                 # → (N, 4+C)
        cx, cy, bw, bh = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
        class_scores = arr[:, 4:]                  # (N, C)
        class_ids = np.argmax(class_scores, axis=1)
        confidences = class_scores[np.arange(len(class_ids)), class_ids]
    else:
        # ---- YOLOv5 / v7 layout: (N, 5+C) ----
        cx, cy, bw, bh = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
        obj_scores = arr[:, 4]
        class_scores = arr[:, 5:]                  # (N, C)
        class_ids = np.argmax(class_scores, axis=1)
        confidences = obj_scores * class_scores[np.arange(len(class_ids)), class_ids]

    # Pre-filter by confidence to reduce NMS work
    mask = confidences >= conf_threshold
    if not np.any(mask):
        return detections

    cx, cy, bw, bh = cx[mask], cy[mask], bw[mask], bh[mask]
    class_ids = class_ids[mask]
    confidences = confidences[mask]

    x1, y1, x2, y2 = _xywh2xyxy(cx, cy, bw, bh)

    # cv2.dnn.NMSBoxes expects integer/float lists, (x, y, w, h) format
    boxes_xywh = np.stack(
        [x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
    scores_list = confidences.tolist()

    indices = cv2.dnn.NMSBoxes(
        boxes_xywh, scores_list, conf_threshold, iou_threshold)

    if indices is None or len(indices) == 0:
        return detections

    # cv2 ≥ 4.7 returns a flat array; older versions return [[i], [j], …]
    indices = np.asarray(indices).flatten()

    for i in indices:
        bx1 = int(x1[i] * scale_x)
        by1 = int(y1[i] * scale_y)
        bx2 = int(x2[i] * scale_x)
        by2 = int(y2[i] * scale_y)
        cid = int(class_ids[i])
        detections.append({
            "bbox": (bx1, by1, bx2, by2),
            "confidence": float(confidences[i]),
            "class_id": cid,
            "class_name": (COCO_CLASSES[cid]
                           if cid < len(COCO_CLASSES) else str(cid)),
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

    read_fn() returns a uint8 ndarray of shape (H, W, 3) in BGR order,
    or None when the source ends.
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
    """Atomically write the current pipeline FPS for external readers."""
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
        pass


def should_stop(args: argparse.Namespace, frame_idx: int,
                start_time: float) -> bool:
    if args.iterations is not None and frame_idx >= args.iterations:
        return True
    if args.duration is not None and (time.perf_counter() - start_time) >= args.duration:
        return True
    return False


def main() -> None:
    args = parse_args()

    print(f"[INFO] Loading ONNX model: {args.model}")
    session = build_session(args.model, args.num_threads)

    input_meta = session.get_inputs()[0]
    input_name = input_meta.name
    # Shape is typically [1, 3, H, W]; fall back to 640 if dynamic (-1)
    _, _, in_h, in_w = input_meta.shape
    in_h = in_h if isinstance(in_h, int) and in_h > 0 else 640
    in_w = in_w if isinstance(in_w, int) and in_w > 0 else 640
    print(f"[INFO] Model input : {input_name}  {in_w}x{in_h} (NCHW float32)")
    print(f"[INFO] CPU threads : {args.num_threads}")

    output_names = [o.name for o in session.get_outputs()]
    print(f"[INFO] Output names: {output_names}")

    read_frame, close_source = make_frame_reader(args, tuple(args.capture_size))

    frame_count = 0
    detections_total = 0
    start = time.perf_counter()
    last_log = start
    last_status_write = start

    try:
        while not should_stop(args, frame_count, start):
            frame_bgr = read_frame()
            if frame_bgr is None:
                print("[INFO] Source ended")
                break

            orig_hw = frame_bgr.shape[:2]
            input_tensor = preprocess(frame_bgr, (in_h, in_w))

            raw_outputs = session.run(output_names, {input_name: input_tensor})

            detections = postprocess(
                raw_outputs, args.conf_threshold, args.iou_threshold,
                (in_h, in_w), orig_hw)

            frame_count += 1
            detections_total += len(detections)

            elapsed = time.perf_counter() - start
            running_fps = frame_count / elapsed if elapsed > 0 else 0.0

            if args.display:
                annotated = draw_detections(frame_bgr.copy(),
                                            detections, running_fps)
                cv2.imshow("YOLO @ CPU (ONNX)", annotated)
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
