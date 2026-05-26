#!/usr/bin/env python3
"""
Lab 3 - Part C1 accuracy evaluation for Hailo HEF models on Raspberry Pi 5
+ Hailo-10H HAT+.

Loads a `.hef` model and a pre-built evaluation dataset (NHWC uint8 images
+ integer class labels), runs inference image-by-image on the NPU, and
reports top-1 / top-5 classification accuracy.

The companion teacher notebook (`lab3b_flowers102_multimodel_teacher.ipynb`)
generates `flowers102_test.npz` -- a single file holding the Oxford
Flowers-102 *test* split resized to the model's input size as uint8 RGB
NHWC, together with the matching integer labels. Copy that .npz next to
your `.hef` files on the Pi, then run this script.

Examples
--------
    # Evaluate one model on the Flowers-102 test set
    python3 evaluate_accuracy.py \
        --model hef_models/mobilenet_v2_flower102.hef \
        --test-data flowers102_test.npz

    # Quick sanity check on the first 200 images
    python3 evaluate_accuracy.py \
        --model hef_models/resnet18_flowers102.hef \
        --test-data flowers102_test.npz \
        --max-samples 200

    # Evaluate all four models and append rows into one CSV (paste into
    # the Part C1 results table)
    for m in mobilenet_v2_flower102 efficientnet_b0_flowers102 \
             resnet18_flowers102 yolo26n_cls_flowers102; do
        python3 evaluate_accuracy.py \
            --model hef_models/${m}.hef \
            --test-data flowers102_test.npz \
            --output-csv c1_accuracy.csv
    done

NPZ schema (built by the teacher notebook)
------------------------------------------
    images  : (N, H, W, 3) uint8, RGB, in [0, 255]
    labels  : (N,)         int64, class indices in [0, num_classes-1]
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from hailo_platform import FormatType, HEF, VDevice


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Accuracy evaluation for a Hailo .hef classifier")
    p.add_argument("--model", required=True,
                   help="Path to the .hef file to evaluate.")
    p.add_argument("--test-data", required=True,
                   help="Path to an .npz file with 'images' (uint8 NHWC) "
                        "and 'labels' (int).")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Evaluate at most this many samples (default: all).")
    p.add_argument("--top-k", type=int, default=5,
                   help="Also report top-K accuracy (default: 5).")
    p.add_argument("--output-csv", default=None,
                   help="Append one summary row to this CSV.")
    p.add_argument("--report-every", type=int, default=200,
                   help="Print running accuracy every N samples.")
    p.add_argument("--infer-timeout-ms", type=int, default=10_000,
                   help="Per-iteration HailoRT timeout.")
    return p.parse_args()


def load_dataset(path: str, in_h: int, in_w: int):
    """Load (images, labels) from an .npz, resizing if needed."""
    if not os.path.isfile(path):
        sys.exit(f"--test-data not found: {path}")
    data = np.load(path)
    if "images" not in data or "labels" not in data:
        sys.exit(f"{path} must contain arrays 'images' and 'labels'.")
    images = data["images"]
    labels = data["labels"].astype(np.int64).reshape(-1)

    if images.dtype != np.uint8:
        sys.exit(f"'images' must be uint8 NHWC, got dtype {images.dtype}.")
    if images.ndim != 4 or images.shape[-1] != 3:
        sys.exit(f"'images' must be (N,H,W,3), got shape {images.shape}.")
    if images.shape[0] != labels.shape[0]:
        sys.exit(f"images / labels length mismatch: "
                 f"{images.shape[0]} vs {labels.shape[0]}.")

    # Resize on the fly if the npz was built for a different input size.
    if images.shape[1] != in_h or images.shape[2] != in_w:
        print(f"[INFO] Resizing dataset {images.shape[1]}x{images.shape[2]} "
              f"-> {in_h}x{in_w}")
        resized = np.empty((images.shape[0], in_h, in_w, 3), dtype=np.uint8)
        for i, img in enumerate(images):
            resized[i] = cv2.resize(img, (in_w, in_h),
                                    interpolation=cv2.INTER_LINEAR)
        images = resized

    return images, labels


def logits_from_output(buf: np.ndarray) -> np.ndarray:
    """Squeeze a Hailo output buffer into a 1-D logits vector."""
    arr = np.asarray(buf).astype(np.float32, copy=False)
    return arr.reshape(-1)


def main() -> None:
    args = parse_args()

    model_path = args.model
    model_name = Path(model_path).name
    hef_mb = os.path.getsize(model_path) / (1024 * 1024)

    print(f"[INFO] Model       : {model_name}  ({hef_mb:.2f} MB)")
    print(f"[INFO] Test data   : {args.test_data}")

    hef = HEF(model_path)
    input_info = hef.get_input_vstream_infos()[0]
    in_h, in_w, _ = input_info.shape
    print(f"[INFO] Input shape : {in_h}x{in_w}")

    images, labels = load_dataset(args.test_data, in_h, in_w)
    n_total = images.shape[0]
    if args.max_samples is not None:
        n_total = min(n_total, args.max_samples)
        images = images[:n_total]
        labels = labels[:n_total]
    print(f"[INFO] Samples     : {n_total}")

    top_k = max(1, args.top_k)
    correct_top1 = 0
    correct_topk = 0
    t_start = time.perf_counter()

    with VDevice() as vdevice:
        infer_model = vdevice.create_infer_model(model_path)
        infer_model.set_batch_size(1)
        infer_model.input().set_format_type(FormatType.UINT8)
        for out_name in infer_model.output_names:
            infer_model.output(out_name).set_format_type(FormatType.FLOAT32)

        with infer_model.configure() as configured:
            input_name = infer_model.input_names[0]
            in_shape = infer_model.input(input_name).shape  # (H,W,C)
            input_buf = np.empty(in_shape, dtype=np.uint8)

            output_bufs = {}
            for out_name in infer_model.output_names:
                out_shape = infer_model.output(out_name).shape
                output_bufs[out_name] = np.empty(out_shape, dtype=np.float32)

            bindings = configured.create_bindings()
            bindings.input(input_name).set_buffer(input_buf)
            for out_name, buf in output_bufs.items():
                bindings.output(out_name).set_buffer(buf)

            # Pick the largest output as the classifier logits (the only
            # vstream for a single-head classifier; for multi-head models
            # we conservatively fall back to the longest 1-D output).
            logit_name = max(
                output_bufs,
                key=lambda k: int(np.prod(output_bufs[k].shape)),
            )

            for i in range(n_total):
                np.copyto(input_buf, images[i])
                configured.run([bindings], args.infer_timeout_ms)
                logits = logits_from_output(output_bufs[logit_name])

                # top-K via argpartition (cheap; we don't need them sorted
                # to test membership).
                k = min(top_k, logits.shape[0])
                topk_idx = np.argpartition(-logits, k - 1)[:k]
                top1 = int(topk_idx[np.argmax(logits[topk_idx])])
                label = int(labels[i])

                if top1 == label:
                    correct_top1 += 1
                if label in topk_idx:
                    correct_topk += 1

                if (i + 1) % args.report_every == 0:
                    acc = 100.0 * correct_top1 / (i + 1)
                    print(f"[INFO] {i + 1:5d}/{n_total}  "
                          f"running top-1: {acc:.2f}%")

    elapsed = time.perf_counter() - t_start
    top1_acc = 100.0 * correct_top1 / n_total
    topk_acc = 100.0 * correct_topk / n_total
    fps = n_total / elapsed if elapsed > 0 else 0.0

    print()
    print(f"=== C1 accuracy ({model_name}) ===")
    print(f"  Samples            : {n_total}")
    print(f"  Top-1 accuracy (%) : {top1_acc:.2f}")
    print(f"  Top-{top_k} accuracy (%) : {topk_acc:.2f}")
    print(f"  Elapsed (s)        : {elapsed:.2f}")
    print(f"  Throughput (img/s) : {fps:.2f}")

    if args.output_csv:
        write_header = not os.path.exists(args.output_csv)
        with open(args.output_csv, "a") as f:
            if write_header:
                f.write("model,hef_mb,samples,top1_acc,topk,topk_acc,"
                        "elapsed_s,img_per_s\n")
            f.write(f"{model_name},{hef_mb:.2f},{n_total},"
                    f"{top1_acc:.2f},{top_k},{topk_acc:.2f},"
                    f"{elapsed:.2f},{fps:.2f}\n")
        print(f"[INFO] Appended row to {args.output_csv}")


if __name__ == "__main__":
    main()
