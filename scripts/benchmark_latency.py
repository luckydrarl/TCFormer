"""
Latency benchmark for EEG models.
Tests both CPU and GPU (if available).
"""
import sys
sys.path.insert(0, 'D:/EEGnet/external/TCFormer')

import torch
import time
import argparse
import numpy as np
from pathlib import Path

from utils.get_model_cls import get_model_cls


def measure_latency(model, input_shape, device="cuda:0", warmup=50, repeat=200):
    """Measure mean and std forward-pass latency (ms) for a single sample."""
    dev = torch.device(device)
    model = model.to(dev).eval()
    dummy = torch.randn(*input_shape, device=dev)

    # Warm-up
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)
        if dev.type == "cuda":
            torch.cuda.synchronize()

        # Timed passes
        times = []
        for _ in range(repeat):
            start = time.perf_counter()
            _ = model(dummy)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)  # ms

    return np.mean(times), np.std(times)


def benchmark_model(model_name, input_shape=(1, 22, 1000),
                    warmup=50, repeat=200):
    """Benchmark a model on CPU and GPU (if available)."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {model_name}")
    print(f"Input shape: {input_shape}")
    print(f"Warmup: {warmup}, Repeat: {repeat}")
    print(f"{'='*60}")

    # Get model class and instantiate
    model_name_map = {
        "tcformer": "TCFormer", "atcnet": "ATCNet",
        "eegconformer": "EEGConformer", "eegtcnet": "EEGTCNet",
        "eegnet": "EEGNet", "shallownet": "ShallowNet",
        "basenet": "BaseNet", "ctnet": "CTNet", "mscformer": "MSCFormer",
    }
    cls_name = model_name_map.get(model_name.lower(), model_name)
    model_cls = get_model_cls(cls_name)

    # Some models need extra init args
    extra_kwargs = {}
    if cls_name in ("EEGNet", "ShallowNet", "BaseNet"):
        extra_kwargs["input_window_samples"] = 1000
    if cls_name == "EEGConformer":
        extra_kwargs["input_size_cls"] = 2440

    model = model_cls(n_channels=22, n_classes=4, **extra_kwargs)
    if hasattr(model, 'model'):
        model = model.model  # Unwrap LightningModule to get pure nn.Module

    results = {}

    # CPU benchmark
    print("\n--- CPU ---")
    try:
        latency_mean, latency_std = measure_latency(
            model, input_shape, device="cpu", warmup=warmup, repeat=repeat)
        print(f"Latency: {latency_mean:.2f} ± {latency_std:.2f} ms")
        results['cpu_latency_mean_ms'] = latency_mean
        results['cpu_latency_std_ms'] = latency_std
    except Exception as e:
        print(f"CPU benchmark failed: {e}")
        results['cpu_latency_mean_ms'] = None
        results['cpu_latency_std_ms'] = None

    # GPU benchmark
    if torch.cuda.is_available():
        print("\n--- GPU ---")
        try:
            latency_mean, latency_std = measure_latency(
                model, input_shape, device="cuda:0", warmup=warmup, repeat=repeat)
            print(f"Latency: {latency_mean:.2f} ± {latency_std:.2f} ms")
            results['gpu_latency_mean_ms'] = latency_mean
            results['gpu_latency_std_ms'] = latency_std
        except Exception as e:
            print(f"GPU benchmark failed: {e}")
            results['gpu_latency_mean_ms'] = None
            results['gpu_latency_std_ms'] = None
    else:
        print("\n--- GPU: Not available ---")
        results['gpu_latency_mean_ms'] = None
        results['gpu_latency_std_ms'] = None

    # Parameter count
    params = sum(p.numel() for p in model.parameters())
    results['params'] = params
    print(f"\nTotal params: {params:,}")

    return results


def main():
    parser = argparse.ArgumentParser(description="EEG Model Latency Benchmark")
    parser.add_argument("--model", type=str, default="tcformer",
                        choices=["tcformer", "atcnet", "eegconformer", "eegtcnet",
                                 "eegnet", "shallownet", "basenet", "ctnet", "mscformer"])
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeat", type=int, default=200)
    parser.add_argument("--output", type=str,
                        default="D:/EEGnet/results/tables/latency_benchmark.csv")
    parser.add_argument("--all", action="store_true",
                        help="Benchmark all models")
    args = parser.parse_args()

    models_to_benchmark = (["tcformer", "atcnet", "eegconformer", "eegtcnet",
                            "eegnet", "shallownet", "basenet"]
                           if args.all else [args.model])

    all_results = []
    input_shape = (1, 22, 1000)  # BCI2a format: (batch, channels, time)

    for model_name in models_to_benchmark:
        try:
            results = benchmark_model(model_name, input_shape,
                                      warmup=args.warmup, repeat=args.repeat)
            results['model'] = model_name
            all_results.append(results)
        except Exception as e:
            print(f"Skipping {model_name}: {e}")

    # Save results
    if all_results:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            header = "model,params,cpu_latency_mean_ms,cpu_latency_std_ms," \
                     "gpu_latency_mean_ms,gpu_latency_std_ms\n"
            f.write(header)
            for r in all_results:
                f.write(f"{r['model']},{r['params']},"
                        f"{r.get('cpu_latency_mean_ms', '')},{r.get('cpu_latency_std_ms', '')},"
                        f"{r.get('gpu_latency_mean_ms', '')},{r.get('gpu_latency_std_ms', '')}\n")

        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
