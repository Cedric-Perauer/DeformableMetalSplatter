import argparse
import time
from typing import Literal

import numpy as np

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


def _parse_compute_units(s: str):
    import coremltools as ct

    m = {
        "all": ct.ComputeUnit.ALL,
        "cpuOnly": ct.ComputeUnit.CPU_ONLY,
        "cpuAndGPU": ct.ComputeUnit.CPU_AND_GPU,
        "cpuAndNeuralEngine": ct.ComputeUnit.CPU_AND_NE,
    }
    if s not in m:
        raise ValueError(f"Unknown compute units '{s}'. Choose from: {', '.join(m.keys())}")
    return m[s]


def export_deform_to_coreml(
    *,
    weights_path: str,
    out_path: str,
    batch_size: int,
    precision: Literal["fp16", "fp32"] = "fp16",
    quantize: Literal["none", "int8"] = "none",
):
    """
    Exports DeformNetwork(x[B,3], t[B,1]) -> (d_xyz[B,3], rotation[B,4], scaling[B,3]) to Core ML.
    """
    import torch
    import coremltools as ct

    from export_deform_weights import DeformNetwork

    def _nn_deployment_target():
        # Older "neuralnetwork" models are not supported for newer deployment targets.
        # coremltools raises if minimum_deployment_target is macOS12+/iOS15+ with convert_to="neuralnetwork".
        for name in ("macOS11", "macOS10_15", "macOS10_14"):
            if hasattr(ct.target, name):
                return getattr(ct.target, name)
        # As a last resort, don't set a minimum target (lets coremltools decide).
        return None

    model = DeformNetwork(D=8, W=256, verbose=False).cpu().eval()
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state)

    # Trace with fixed batch size (Core ML will compile for this shape).
    x = torch.rand(batch_size, 3, dtype=torch.float32)
    t = torch.rand(batch_size, 1, dtype=torch.float32)
    with torch.inference_mode():
        traced = torch.jit.trace(model, (x, t), strict=False)

    # Use ML Program backend by default (recommended on modern macOS).
    inputs = [
        ct.TensorType(name="x", shape=x.shape, dtype=np.float32),
        ct.TensorType(name="t", shape=t.shape, dtype=np.float32),
    ]

    if quantize == "int8":
        used_legacy_int8 = False
        # Prefer modern mlprogram + coremltools.optimize when available.
        try:
            import coremltools.optimize as cto  # type: ignore

            mlmodel = ct.convert(
                traced,
                convert_to="mlprogram",
                inputs=inputs,
                minimum_deployment_target=ct.target.macOS13,
                compute_precision=ct.precision.FLOAT16 if precision == "fp16" else ct.precision.FLOAT32,
            )
            op_config = cto.coreml.OpLinearQuantizerConfig(mode="linear_symmetric")
            config = cto.coreml.OptimizationConfig(global_config=op_config)
            mlmodel = cto.coreml.linear_quantize_weights(mlmodel, config=config)
        except Exception:
            # Legacy fallback for older coremltools: neuralnetwork + quantization_utils.
            # Note: this is weight compression for neuralnetwork models (not mlprogram).
            used_legacy_int8 = True
            nn_target = _nn_deployment_target()
            if nn_target is None:
                mlmodel = ct.convert(
                    traced,
                    convert_to="neuralnetwork",
                    inputs=inputs,
                )
            else:
                mlmodel = ct.convert(
                    traced,
                    convert_to="neuralnetwork",
                    inputs=inputs,
                    minimum_deployment_target=nn_target,
                )
            from coremltools.models.neural_network import quantization_utils

            mlmodel = quantization_utils.quantize_weights(mlmodel, nbits=8)

        if used_legacy_int8 and out_path.endswith(".mlpackage"):
            # Legacy neuralnetwork models are typically stored as .mlmodel.
            out_path = out_path[: -len(".mlpackage")] + ".mlmodel"
            print(
                f"[info] coremltools.optimize not found; using legacy neuralnetwork int8 quantization. "
                f"Saving as: {out_path}"
            )
        elif not used_legacy_int8:
            print("[info] Using mlprogram + coremltools.optimize int8 weight quantization.")
    else:
        # macOS 13+ is a safe baseline for ML Program.
        mlmodel = ct.convert(
            traced,
            convert_to="mlprogram",
            inputs=inputs,
            minimum_deployment_target=ct.target.macOS13,
            compute_precision=ct.precision.FLOAT16 if precision == "fp16" else ct.precision.FLOAT32,
        )

    mlmodel.save(out_path)
    return out_path


def benchmark_coreml(
    *,
    model_path: str,
    iters: int,
    batch_size: int,
    compute_units: str = "all",
):
    import coremltools as ct

    cu = _parse_compute_units(compute_units)
    mlmodel = ct.models.MLModel(model_path, compute_units=cu)

    x = np.random.rand(batch_size, 3).astype(np.float32)
    t = np.random.rand(batch_size, 1).astype(np.float32)

    # Warmup (Core ML has compilation and caching effects).
    for _ in range(min(5, iters)):
        _ = mlmodel.predict({"x": x, "t": t})

    per_iter_s: list[float] = []
    iterator = range(iters)
    if tqdm is not None:
        iterator = tqdm(iterator, total=iters, desc=f"CoreML B={batch_size} ({compute_units})")

    for _ in iterator:
        t0 = time.perf_counter()
        _ = mlmodel.predict({"x": x, "t": t})
        t1 = time.perf_counter()
        dt = t1 - t0
        per_iter_s.append(dt)

        if tqdm is not None:
            avg_ms = 1000.0 * (sum(per_iter_s) / len(per_iter_s))
            iterator.set_postfix_str(f"{dt*1000.0:.2f} ms/batch (avg {avg_ms:.2f})")

    avg_ms = 1000.0 * (sum(per_iter_s) / len(per_iter_s)) if per_iter_s else float("nan")
    print(f"Done: {iters} iters, batch={batch_size}, compute_units={compute_units}, avg {avg_ms:.2f} ms/batch")


def main():
    p = argparse.ArgumentParser(description="Export DeformNetwork to Core ML and benchmark runtime.")
    p.add_argument("--weights", default="deform.pth", help="Path to PyTorch weights (deform.pth).")
    p.add_argument("--out", default="DeformNetwork.mlpackage", help="Core ML output path (.mlpackage recommended).")
    p.add_argument("--batch-size", type=int, default=16384, help="Batch size used for export + benchmark.")
    p.add_argument("--iters", type=int, default=50, help="Benchmark iterations.")
    p.add_argument(
        "--compute-units",
        default="all",
        choices=["all", "cpuOnly", "cpuAndGPU", "cpuAndNeuralEngine"],
        help="Core ML compute units.",
    )
    p.add_argument("--precision", default="fp16", choices=["fp16", "fp32"], help="Core ML compute precision.")
    p.add_argument(
        "--quantize",
        default="none",
        choices=["none", "int8"],
        help="Optional post-training weight quantization.",
    )
    p.add_argument("--export-only", action="store_true", help="Only export the Core ML model, don't benchmark.")
    args = p.parse_args()

    export_deform_to_coreml(
        weights_path=args.weights,
        out_path=args.out,
        batch_size=args.batch_size,
        precision=args.precision,
        quantize=args.quantize,
    )
    print(f"Saved Core ML model to: {args.out}")

    if not args.export_only:
        benchmark_coreml(
            model_path=args.out,
            iters=args.iters,
            batch_size=args.batch_size,
            compute_units=args.compute_units,
        )


if __name__ == "__main__":
    main()

