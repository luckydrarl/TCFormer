"""
Unified LOSO (Leave-One-Subject-Out) training script for MI-EEG classification.
Supports multiple models: tcformer, atcnet, eegconformer, eegtcnet, eegnet, etc.

Usage:
    python scripts/train_loso.py --model tcformer --seed 42
    python scripts/train_loso.py --model atcnet --gpu_id 0
    python scripts/train_loso.py --all --gpu_id 0

Output:
    results/tables/loso_results.csv   — per-subject results
    results/tables/loso_summary.csv   — aggregated summary
"""
import sys
sys.path.insert(0, 'D:/EEGnet/external/TCFormer')

import os
import time
import csv
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, f1_score,
    recall_score, confusion_matrix
)

from utils.get_model_cls import get_model_cls
from utils.get_datamodule_cls import get_datamodule_cls
from utils.seed import seed_everything
from utils.latency import measure_latency


def compute_specificity(y_true, y_pred, n_classes=4):
    """Compute per-class specificity and return macro average."""
    cm = confusion_matrix(y_true, y_pred, labels=range(n_classes))
    specificities = []
    for i in range(n_classes):
        tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        fp = cm[:, i].sum() - cm[i, i]
        if tn + fp > 0:
            specificities.append(tn / (tn + fp))
        else:
            specificities.append(0.0)
    return np.mean(specificities)


def compute_flops(model, input_shape=(1, 22, 1000)):
    """Estimate FLOPs using thop (if available) or return None."""
    try:
        from thop import profile
        dummy = torch.randn(*input_shape)
        flops, _ = profile(model, inputs=(dummy,), verbose=False)
        return flops
    except ImportError:
        return None


def evaluate_model(model, datamodule, result_dir, config):
    """Train and evaluate a single model on all LOSO subjects."""
    model_name = config["model"]
    dataset_name = config["dataset_name"]

    # Get subject IDs
    subject_ids = list(range(1, 10))  # BCI2a: 9 subjects

    results = []
    train_times_all = []

    for subject_id in subject_ids:
        print(f"\n{'='*60}")
        print(f">>> LOSO: Test subject {subject_id} / 9")
        print(f"{'='*60}")

        seed_everything(config["seed"])

        # Initialize datamodule for LOSO
        datamodule_cls = get_datamodule_cls("bcic2a_loso")
        datamodule = datamodule_cls(config["preprocessing"], subject_id=subject_id)

        # Initialize model
        model_cls = get_model_cls(model_name)
        model = model_cls(**config["model_kwargs"], max_epochs=config["max_epochs"])

        # Trainer
        trainer = Trainer(
            max_epochs=config["max_epochs"],
            devices=[config.get("gpu_id", 0)] if config.get("gpu_id", 0) != -1 else 1,
            accelerator="auto",
            strategy="auto",
            logger=False,
            enable_checkpointing=False,
            num_sanity_val_steps=0,
        )

        # Train
        st_train = time.time()
        trainer.fit(model, datamodule=datamodule)
        train_time_min = (time.time() - st_train) / 60
        train_times_all.append(train_time_min)

        # Test — get predictions
        test_loader = datamodule.test_dataloader()
        model.eval()
        model = model.cpu()

        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                y_hat = model(batch_x)
                preds = torch.argmax(y_hat, dim=-1)
                all_preds.append(preds.numpy())
                all_labels.append(batch_y.numpy())

        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_labels)

        # Metrics
        acc = accuracy_score(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        specificity = compute_specificity(y_true, y_pred, n_classes=4)

        # Latency
        sample_x, _ = datamodule.test_dataset[0]
        input_shape = (1, *sample_x.shape)
        lat_ms = measure_latency(model, input_shape, device="cpu")

        # Params
        params = sum(p.numel() for p in model.parameters())

        # FLOPs
        flops = compute_flops(model, input_shape)

        result = {
            'model_name': model_name,
            'test_subject': subject_id,
            'accuracy': acc,
            'kappa': kappa,
            'macro_f1': macro_f1,
            'recall': recall,
            'specificity': specificity,
            'params': params,
            'flops': flops if flops else '',
            'latency_ms': lat_ms,
            'train_time': train_time_min,
        }
        results.append(result)

        print(f"Subject {subject_id}: Acc={acc:.4f}, Kappa={kappa:.4f}, "
              f"F1={macro_f1:.4f}, Time={train_time_min:.1f}min")

    # Save per-subject results
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    per_subject_path = output_dir / "loso_results.csv"
    fieldnames = ['model_name', 'test_subject', 'accuracy', 'kappa', 'macro_f1',
                  'recall', 'specificity', 'params', 'flops', 'latency_ms', 'train_time']

    # Append to existing file if present
    file_exists = per_subject_path.exists()
    with open(per_subject_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"\nPer-subject results saved to: {per_subject_path}")

    # Compute and save summary
    accs = [r['accuracy'] for r in results]
    kappas = [r['kappa'] for r in results]
    f1s = [r['macro_f1'] for r in results]
    lats = [r['latency_ms'] for r in results]

    summary = {
        'model_name': model_name,
        'mean_acc': np.mean(accs),
        'std_acc': np.std(accs),
        'mean_kappa': np.mean(kappas),
        'std_kappa': np.std(kappas),
        'mean_f1': np.mean(f1s),
        'std_f1': np.std(f1s),
        'mean_latency_ms': np.mean(lats),
        'params': results[0]['params'],
        'flops': results[0]['flops'] if results[0]['flops'] else '',
    }

    summary_path = output_dir / "loso_summary.csv"
    summary_fields = ['model_name', 'mean_acc', 'std_acc', 'mean_kappa', 'std_kappa',
                      'mean_f1', 'std_f1', 'mean_latency_ms', 'params', 'flops']

    summary_exists = summary_path.exists()
    with open(summary_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields, extrasaction='ignore')
        if not summary_exists:
            writer.writeheader()
        writer.writerow(summary)

    print(f"Summary saved to: {summary_path}")
    print(f"\n=== LOSO Summary: {model_name} ===")
    print(f"Accuracy: {summary['mean_acc']*100:.2f} ± {summary['std_acc']*100:.2f}%")
    print(f"Kappa:    {summary['mean_kappa']:.4f} ± {summary['std_kappa']:.4f}")
    print(f"F1:       {summary['mean_f1']:.4f} ± {summary['std_f1']:.4f}")

    return results, summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified LOSO training for EEG MI models")
    parser.add_argument("--model", type=str, default="tcformer",
                        choices=["tcformer", "atcnet", "eegconformer", "eegtcnet",
                                 "eegnet", "shallownet", "basenet", "ctnet"])
    parser.add_argument("--all", action="store_true",
                        help="Run all models sequentially")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU ID (-1 for CPU)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str,
                        default="D:/EEGnet/results/tables")
    return parser.parse_args()


def get_config(model_name, gpu_id, seed, output_dir):
    """Build config dict for a given model."""
    # Default preprocessing for BCI2a LOSO
    preprocessing = {
        "sfreq": 250,
        "low_cut": None,
        "high_cut": None,
        "start": 0.0,
        "stop": 0.0,
        "batch_size": 48,
        "num_workers": 0,  # Windows-safe
        "interaug": False,  # No augmentation for LOSO
        "z_scale": True,
        "seed": seed,
    }

    # Model-specific kwargs
    model_kwargs = {
        "n_channels": 22,
        "n_classes": 4,
    }

    # Common config
    config = {
        "model": model_name,
        "dataset_name": "bcic2a_loso",
        "preprocessing": preprocessing,
        "model_kwargs": model_kwargs,
        "max_epochs": 125,  # LOSO setting from TCFormer paper
        "seed": seed,
        "gpu_id": gpu_id,
        "output_dir": output_dir,
    }

    # Model-specific overrides (using TCFormer paper defaults)
    if model_name == "tcformer":
        model_kwargs.update({
            "F1": 32,
            "temp_kernel_lengths": [20, 32, 64],
            "d_group": 16,
            "D": 2,
            "pool_length_1": 8,
            "pool_length_2": 7,
            "dropout_conv": 0.4,
            "use_group_attn": True,
            "q_heads": 4,
            "kv_heads": 2,
            "trans_depth": 5,
            "trans_dropout": 0.4,
            "tcn_depth": 2,
            "kernel_length_tcn": 4,
            "dropout_tcn": 0.3,
            "lr": 0.0009,
            "beta_1": 0.5,
            "weight_decay": 0.001,
            "optimizer": "adam",
            "scheduler": True,
            "warmup_epochs": 3,
        })
    elif model_name == "atcnet":
        model_kwargs.update({
            "lr": 0.0009,
            "beta_1": 0.5,
            "weight_decay": 0.001,
            "optimizer": "adam",
            "scheduler": True,
            "warmup_epochs": 3,
        })
    elif model_name == "eegconformer":
        model_kwargs.update({
            "lr": 0.0009,
            "weight_decay": 0.001,
            "optimizer": "adam",
            "scheduler": True,
            "warmup_epochs": 3,
        })
    elif model_name == "eegtcnet":
        model_kwargs.update({
            "lr": 0.0009,
            "weight_decay": 0.001,
            "optimizer": "adam",
            "scheduler": True,
            "warmup_epochs": 3,
        })

    return config


def main():
    args = parse_args()

    models_to_run = ["tcformer", "atcnet", "eegconformer", "eegtcnet",
                     "eegnet", "shallownet", "basenet"] if args.all else [args.model]

    for model_name in models_to_run:
        print(f"\n{'#'*60}")
        print(f"# Running LOSO for: {model_name}")
        print(f"{'#'*60}")

        config = get_config(model_name, args.gpu_id, args.seed, args.output_dir)

        try:
            evaluate_model(None, None, None, config)  # We re-init inside evaluate_model
        except Exception as e:
            print(f"ERROR running {model_name}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
