"""
Minimal script to verify BCI Competition IV 2a data loading.
Reads A01T.gdf, prints raw.info, events, event_id.
"""
import sys
sys.path.insert(0, 'D:/EEGnet/external/TCFormer')

import os
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:10808'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:10808'

from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import (
    Preprocessor,
    create_windows_from_events,
    preprocess,
    scale,
)
import numpy as np

# Load BCI IV 2a data for subject 1 via MOABB
print("Loading BCI Competition IV 2a dataset (subject 1)...")
dataset = MOABBDataset("BNCI2014001", subject_ids=[1])

print(f"\nNumber of runs: {len(dataset.datasets)}")
for i, ds in enumerate(dataset.datasets):
    raw = ds.raw
    print(f"\n--- Run {i}: {ds.description} ---")
    print(f"Raw info: {raw.info['sfreq']} Hz, channels: {len(raw.ch_names)}")
    print(f"Channel names: {raw.ch_names}")
    print(f"Events shape: {raw.get_data().shape}")
    # Events come from original raw's stim channel before resampling
    try:
        events = ds.raw.find_events()
        if events is not None and len(events) > 0:
            unique, counts = np.unique(events[:, -1], return_counts=True)
            print(f"Event counts: {dict(zip(unique, counts))}")
    except Exception:
        print("Events not available after preprocessing (RawArray)")

# Apply preprocessing as in the official pipeline
print("\n\n--- Applying preprocessing ---")
preprocessors = [
    Preprocessor("pick_types", eeg=True, meg=False, stim=False),
    Preprocessor(scale, factor=1e6, apply_on_array=True),
    Preprocessor("resample", sfreq=250)
]
preprocess(dataset, preprocessors)

# Create windows (0 to 4s after cue)
sfreq = dataset.datasets[0].raw.info["sfreq"]
trial_start_offset_samples = int(0.0 * sfreq)
trial_stop_offset_samples = int(4.0 * sfreq)  # Full 4s trial

windows_dataset = create_windows_from_events(
    dataset,
    trial_start_offset_samples=trial_start_offset_samples,
    trial_stop_offset_samples=trial_stop_offset_samples,
    preload=False,
)

# Split by session
splitted = windows_dataset.split("session")
session_T = splitted["session_T"]
session_E = splitted["session_E"]

# Load data
X_train = np.concatenate([run.windows.load_data()._data for run in session_T.datasets], axis=0)
y_train = np.concatenate([run.y for run in session_T.datasets], axis=0)
X_test = np.concatenate([run.windows.load_data()._data for run in session_E.datasets], axis=0)
y_test = np.concatenate([run.y for run in session_E.datasets], axis=0)

print(f"\nTraining data shape: {X_train.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Test data shape: {X_test.shape}")
print(f"Test labels shape: {y_test.shape}")
print(f"\nLabel distribution (train): {dict(zip(*np.unique(y_train, return_counts=True)))}")
print(f"Label distribution (test): {dict(zip(*np.unique(y_test, return_counts=True)))}")
print(f"\nLabel mapping: 0=feet, 1=left_hand, 2=right_hand, 3=tongue")
print(f"Data format: [samples, channels, time] = [{X_train.shape[0]}, {X_train.shape[1]}, {X_train.shape[2]}]")
print("\n✓ Data verification complete!")
