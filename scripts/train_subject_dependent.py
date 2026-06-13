"""
Subject-dependent training wrapper for TCFormer and other EEG models.
Uses the official TCFormer pipeline but saves results in unified CSV format.

Usage:
    python scripts/train_subject_dependent.py --model tcformer --seed 42
    python scripts/train_subject_dependent.py --all --gpu_id 0
"""
import sys
sys.path.insert(0, 'D:/EEGnet/external/TCFormer')

import os
import re
import csv
import argparse
import yaml
from pathlib import Path
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="tcformer")
    parser.add_argument("--dataset", type=str, default="bcic2a")
    parser.add_argument("--interaug", action="store_true", default=True)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--output_dir", type=str,
                        default="D:/EEGnet/results/tables")
    return parser.parse_args()


def run_official_pipeline(model_name, dataset, interaug, gpu_id, seed):
    """Run the official TCFormer train_pipeline.py and return the results dir."""
    import subprocess

    cmd = [
        "python", "D:/EEGnet/external/TCFormer/train_pipeline.py",
        "--model", model_name,
        "--dataset", dataset,
        "--gpu_id", str(gpu_id),
        "--seed", str(seed),
    ]
    if interaug:
        cmd.append("--interaug")
    else:
        cmd.append("--no_interaug")

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True,
                           cwd="D:/EEGnet/external/TCFormer")
    return result.returncode


def find_latest_results_dir(model_name, dataset_name="bcic2a"):
    """Find the most recent results directory for a given model."""
    results_root = Path("D:/EEGnet/external/TCFormer/results")
    if not results_root.exists():
        return None

    pattern = f"{model_name}_{dataset_name}_seed-*"
    dirs = sorted(results_root.glob(pattern), key=os.path.getmtime, reverse=True)
    return dirs[0] if dirs else None


def parse_results_txt(results_dir):
    """Parse official results.txt and extract per-subject metrics."""
    results_file = results_dir / "results.txt"
    if not results_file.exists():
        return None, None

    with open(results_file) as f:
        text = f.read()

    # Extract per-subject results
    # Format: Subject X => Train Time: Y.YYm, Test Time: Z.ZZs, Test Acc: A.AAAA, Test Loss: L.LLLL, Test Kappa: K.KKKK
    pattern = r"Subject (\d+) => Train Time: ([\d.]+)m, Test Time: ([\d.]+)s, Test Acc: ([\d.]+), Test Loss: ([\d.]+), Test Kappa: ([\d.]+)"
    per_subject = []
    for m in re.finditer(pattern, text):
        per_subject.append({
            'subject': int(m.group(1)),
            'train_time_min': float(m.group(2)),
            'test_time_s': float(m.group(3)),
            'accuracy': float(m.group(4)),
            'loss': float(m.group(5)),
            'kappa': float(m.group(6)),
        })

    # Extract summary
    summary = {}
    m = re.search(r'Average Test Accuracy: ([\d.]+) ± ([\d.]+)', text)
    if m:
        summary['acc_mean'] = float(m.group(1))
        summary['acc_std'] = float(m.group(2))
    m = re.search(r'Average Test Kappa:\s+([\d.]+) ± ([\d.]+)', text)
    if m:
        summary['kappa_mean'] = float(m.group(1))
        summary['kappa_std'] = float(m.group(2))
    m = re.search(r'#Params: (\d+)', text)
    if m:
        summary['params'] = int(m.group(1))
    m = re.search(r'Average Response Time: ([\d.]+) ms', text)
    if m:
        summary['avg_latency_ms'] = float(m.group(1))

    return per_subject, summary


def save_unified_csv(per_subject, summary, model_name, dataset_name, output_dir):
    """Save results in the unified project CSV format."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Per-subject results
    csv_path = output_path / f"{model_name}_{dataset_name}_results.csv"
    fieldnames = ['model_name', 'test_subject', 'accuracy', 'kappa',
                   'train_time_min', 'test_time_s']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in per_subject:
            writer.writerow({
                'model_name': model_name,
                'test_subject': r['subject'],
                'accuracy': r['accuracy'],
                'kappa': r['kappa'],
                'train_time_min': r['train_time_min'],
                'test_time_s': r['test_time_s'],
            })

    print(f"Per-subject results: {csv_path}")

    # Summary
    if summary:
        summary_path = output_path / f"{model_name}_{dataset_name}_summary.csv"
        with open(summary_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['model_name', 'mean_acc', 'std_acc',
                                                    'mean_kappa', 'std_kappa',
                                                    'params', 'avg_latency_ms'])
            writer.writeheader()
            writer.writerow({
                'model_name': model_name,
                'mean_acc': summary.get('acc_mean', ''),
                'std_acc': summary.get('acc_std', ''),
                'mean_kappa': summary.get('kappa_mean', ''),
                'std_kappa': summary.get('kappa_std', ''),
                'params': summary.get('params', ''),
                'avg_latency_ms': summary.get('avg_latency_ms', ''),
            })
        print(f"Summary: {summary_path}")


def main():
    args = parse_args()

    models = ["tcformer", "atcnet", "eegconformer", "eegtcnet",
              "eegnet", "shallownet", "basenet"] if args.all else [args.model]

    for model in models:
        print(f"\n{'#'*60}")
        print(f"# Training: {model} on {args.dataset}")
        print(f"{'#'*60}")

        rc = run_official_pipeline(model, args.dataset, args.interaug,
                                   args.gpu_id, args.seed)
        if rc != 0:
            print(f"ERROR: {model} training failed with code {rc}")
            continue

        results_dir = find_latest_results_dir(model, args.dataset)
        if results_dir:
            per_subject, summary = parse_results_txt(results_dir)
            if per_subject:
                save_unified_csv(per_subject, summary, model, args.dataset,
                                args.output_dir)


if __name__ == "__main__":
    main()
