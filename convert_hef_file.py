#!/usr/bin/env python3
"""
Convert an ONNX model into a Hailo HEF binary.

Wraps the Hailo Dataflow Compiler flow (translate -> optimize/quantize ->
compile) behind a small CLI so the calibration shape, ONNX path and output
HEF path can all be passed as arguments instead of being hard-coded.

Examples
--------
    # MobileNetV2 / CIFAR-10, default 100x32x32x3 calibration set
    python3 convert_hef_file.py --onnx mobilenetv2_cifar10.onnx

    # Explicit output path and a 224x224 RGB calibration set
    python3 convert_hef_file.py \\
            --onnx model.onnx \\
            --hef out/model.hef \\
            --calib-shape 64 224 224 3

    # Use a real calibration dataset saved as NHWC float32 .npy
    python3 convert_hef_file.py --onnx model.onnx --calib-npy calib.npy

    # Compile for a different Hailo target and let TF use the GPU
    python3 convert_hef_file.py --onnx model.onnx --hw-arch hailo8 --gpu

Note on layout
--------------
ONNX models exported from PyTorch are NCHW (N, C, H, W), but Hailo's parser
transposes inputs to NHWC internally. Calibration data MUST therefore be
provided as NHWC: (N, H, W, C). --calib-shape takes values in that order.
"""

import argparse
import os
import sys
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Convert an ONNX model into a Hailo HEF binary.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--onnx",
        required=True,
        type=Path,
        help="Path to the input .onnx file.",
    )
    parser.add_argument(
        "--hef",
        "--output",
        dest="hef",
        type=Path,
        default=None,
        help="Path to the output .hef file "
        "(default: <onnx name>.hef next to the ONNX file).",
    )
    parser.add_argument(
        "--calib-shape",
        nargs=4,
        type=int,
        metavar=("N", "H", "W", "C"),
        default=[100, 224, 224, 3],
        help="Calibration dataset shape in NHWC order. "
        "Ignored when --calib-npy is given.",
    )
    parser.add_argument(
        "--calib-npy",
        type=Path,
        default=None,
        help="Optional .npy file holding a real NHWC float32 calibration set. "
        "Overrides --calib-shape (random data is used otherwise).",
    )
    parser.add_argument(
        "--hw-arch",
        default="hailo10h",
        help="Target Hailo hardware architecture (e.g. hailo10h, hailo8, hailo8l).",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Internal model name (default: ONNX file stem).",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Allow Hailo's internal TensorFlow to use the GPU. "
        "By default TF is forced onto the CPU to avoid JIT/XLA failures "
        "in conda envs with mismatched CUDA/cuDNN.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for the random calibration set (reproducibility).",
    )
    parser.add_argument(
        "--end-node",
        dest="end_nodes",
        action="append",
        default=None,
        help="ONNX node name where the Hailo graph should be cut. "
        "Use this to drop unsupported decode ops (e.g. YOLO26's TopK / "
        "GatherElements / Mod head). Can be passed multiple times. "
        "For YOLO26 the Hailo parser recommends /model.23/Transpose.",
    )
    parser.add_argument(
        "--start-node",
        dest="start_nodes",
        action="append",
        default=None,
        help="Optional ONNX node name(s) to use as graph input(s). "
        "Usually not needed; pass through only if Hailo asks for it.",
    )
    parser.add_argument(
        "--alls",
        type=Path,
        default=None,
        help="Path to a Hailo model script (.alls) loaded before optimize(). "
        "Use for pre-quantization tweaks (e.g. promoting a layer to 16-bit, "
        "splitting a global avgpool, etc.).",
    )
    parser.add_argument(
        "--alls-cmd",
        dest="alls_cmds",
        action="append",
        default=None,
        help="Inline model-script command appended after --alls. Repeatable.",
    )
    parser.add_argument(
        "--avgpool-a16",
        dest="avgpool_a16",
        default=None,
        help="Shortcut: promote the given avgpool layer to 16-bit activations "
        "(fixes 'Shift delta ... larger than 2' on EfficientNet-style GAPs). "
        "Example: --avgpool-a16 efficientnet_b0_flowers102/avgpool1",
    )
    return parser.parse_args(argv)


def load_calib_dataset(args, np):
    """Return an NHWC float32 calibration array."""
    if args.calib_npy is not None:
        data = np.load(args.calib_npy).astype(np.float32)
        if data.ndim != 4:
            sys.exit(
                f"Calibration data must be 4-D NHWC, got shape {data.shape}."
            )
        print(f"-> Loaded calibration set {data.shape} from {args.calib_npy}")
        return data
    
    if args.seed is not None:
        np.random.seed(args.seed)
    shape = tuple(args.calib_shape)
    # WARNING: random calibration data yields poor prediction accuracy but is
    # perfectly valid for FPS / benchmarking purposes.
    print(f"-> Generating random calibration set {shape} (NHWC)")
    return np.random.rand(*shape).astype(np.float32)


def main(argv=None):
    args = parse_args(argv)

    if not args.onnx.is_file():
        sys.exit(f"ONNX file not found: {args.onnx}")

    model_name = args.model_name or args.onnx.stem
    hef_path = args.hef or args.onnx.with_suffix(".hef")
    hef_path.parent.mkdir(parents=True, exist_ok=True)

    # These env vars must be set BEFORE TensorFlow / hailo_sdk_client import,
    # so we configure them here and defer the heavy import until after.
    if not args.gpu:
        # Force Hailo's internal TensorFlow onto the CPU. The GPU JIT/XLA path
        # often fails inside conda envs with mismatched CUDA/cuDNN versions,
        # raising "JIT compilation failed. [Op:Log]" during optimize().
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"

    import numpy as np
    from hailo_sdk_client import ClientRunner

    runner = ClientRunner(hw_arch=args.hw_arch)

    print(f"-> Translating ONNX ({args.onnx}) for {args.hw_arch}...")
    translate_kwargs = {}
    if args.end_nodes:
        translate_kwargs["end_node_names"] = args.end_nodes
        print(f"   end_node_names = {args.end_nodes}")
    if args.start_nodes:
        translate_kwargs["start_node_names"] = args.start_nodes
        print(f"   start_node_names = {args.start_nodes}")
    runner.translate_onnx_model(str(args.onnx), model_name, **translate_kwargs)

    calib_dataset = load_calib_dataset(args, np)

    alls_chunks = []
    if args.alls is not None:
        if not args.alls.is_file():
            sys.exit(f"--alls file not found: {args.alls}")
        alls_chunks.append(args.alls.read_text())
        print(f"-> Loaded model script from {args.alls}")
    if args.avgpool_a16:
        # AvgPool has no weights -> must use a16_w16 (a16_w8 is rejected).
        alls_chunks.append(
            f"quantization_param({args.avgpool_a16}, precision_mode=a16_w16)\n"
        )
        print(f"-> Promoting {args.avgpool_a16} to a16_w16")
    if args.alls_cmds:
        for cmd in args.alls_cmds:
            line = cmd if cmd.endswith("\n") else cmd + "\n"
            alls_chunks.append(line)
            print(f"-> Appending alls cmd: {cmd}")
    if alls_chunks:
        script = "\n".join(alls_chunks)
        runner.load_model_script(script)

    print("-> Quantizing / optimizing (consumes a lot of RAM and time)...")
    runner.optimize(calib_dataset)

    print("-> Compiling to HEF...")
    hef_buffer = runner.compile()

    with open(hef_path, "wb") as f:
        f.write(hef_buffer)

    print(f"Done. HEF saved to: {hef_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
