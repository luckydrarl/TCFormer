from typing import Optional
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data.dataloader import DataLoader

from .base import BaseDataModule
from utils.load_bcic4 import load_bcic4
from utils.euclidean_alignment import apply_ea_to_subject
from sklearn.model_selection import train_test_split
import os


class BCICIV2a(BaseDataModule):
    all_subject_ids = list(range(1, 10))
    class_names = ["feet", "hand(L)", "hand(R)", "tongue"]
    channels = 22
    classes = 4

    def __init__(self, preprocessing_dict, subject_id):
        super().__init__(preprocessing_dict, subject_id)

    def prepare_data(self) -> None:
        self.dataset = load_bcic4(subject_ids=[self.subject_id], dataset="2a",
                                 preprocessing_dict=self.preprocessing_dict)

    def setup(self, stage: Optional[str] = None) -> None:
        if self.dataset is None:
            self.prepare_data()
        # split the data
        splitted_ds = self.dataset.split("session")
        train_dataset, test_dataset = splitted_ds["session_T"], splitted_ds["session_E"]

        # load the data
        X = np.concatenate(
            [run.windows.load_data()._data for run in train_dataset.datasets], axis=0)
        y = np.concatenate([run.y for run in train_dataset.datasets], axis=0)
        X_test = np.concatenate(
            [run.windows.load_data()._data for run in test_dataset.datasets], axis=0)
        y_test = np.concatenate([run.y for run in test_dataset.datasets], axis=0)

        # ── Euclidean Alignment (per-subject, before z-scale) ──
        use_ea = self.preprocessing_dict.get("use_ea", False)
        if use_ea:
            print(f"[EA] Subject {self.subject_id}: aligning train shape={X.shape}, test shape={X_test.shape}")
            X = apply_ea_to_subject(X)
            X_test = apply_ea_to_subject(X_test)
            print(f"[EA] Subject {self.subject_id}: aligned train shape={X.shape}, test shape={X_test.shape}")
            if np.any(np.isnan(X)) or np.any(np.isnan(X_test)):
                print(f"[EA] WARNING: NaN detected after alignment for subject {self.subject_id}")

        # scale data
        if self.preprocessing_dict["z_scale"]:
            X, X_test = BaseDataModule._z_scale(X, X_test)

        # make datasets
        self.train_dataset = BaseDataModule._make_tensor_dataset(X, y)
        self.test_dataset = BaseDataModule._make_tensor_dataset(X_test, y_test)


class BCICIV2aTVT(BaseDataModule):
    val_dataset = None
    all_subject_ids = list(range(1, 10))
    class_names = ["feet", "hand(L)", "hand(R)", "tongue"]
    channels = 22
    classes = 4

    def __init__(self, preprocessing_dict, subject_id):
        super().__init__(preprocessing_dict, subject_id)

    def prepare_data(self) -> None:
        self.dataset = load_bcic4(subject_ids=[self.subject_id], dataset="2a",
                                 preprocessing_dict=self.preprocessing_dict)

    def setup(self, stage: Optional[str] = None) -> None:
        if self.dataset is None:
            self.prepare_data()

        # Split by session
        splitted_ds = self.dataset.split("session")
        session1 = splitted_ds["session_T"]  # training + validation
        session2 = splitted_ds["session_E"]  # testing only

        # Load session 1 data
        X = np.concatenate([run.windows.load_data()._data for run in session1.datasets], axis=0)
        y = np.concatenate([run.y for run in session1.datasets], axis=0)

        # Split session 1: 80% train, 20% validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=self.preprocessing_dict.get("seed", 42), stratify=y)

        # Load session 2 as test set
        X_test = np.concatenate([run.windows.load_data()._data for run in session2.datasets], axis=0)
        y_test = np.concatenate([run.y for run in session2.datasets], axis=0)

        # ── Euclidean Alignment (per-subject, before z-scale) ──
        use_ea = self.preprocessing_dict.get("use_ea", False)
        if use_ea:
            print(f"[EA] Subject {self.subject_id} (TVT): aligning train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")
            # Align train and test separately for this subject
            # (val is from same subject as train, so apply same R? No - each split gets its own R)
            # Per requirements: each split uses its own data for EA
            X_train = apply_ea_to_subject(X_train)
            X_val = apply_ea_to_subject(X_val)
            X_test = apply_ea_to_subject(X_test)
            print(f"[EA] Subject {self.subject_id} (TVT): aligned train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")

        # scale data
        if self.preprocessing_dict["z_scale"]:
            X_train, X_val, X_test = BaseDataModule._z_scale_tvt(X_train, X_val, X_test)

        # Create datasets
        self.train_dataset = BaseDataModule._make_tensor_dataset(X_train, y_train)
        self.val_dataset = BaseDataModule._make_tensor_dataset(X_val, y_val)
        self.test_dataset = BaseDataModule._make_tensor_dataset(X_test, y_test)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset,
                          batch_size=self.preprocessing_dict["batch_size"],
                          num_workers=self.preprocessing_dict.get("num_workers", os.cpu_count() // 2),
                          pin_memory=True,
                        )


class BCICIV2aLOSO(BCICIV2a):
    val_dataset = None

    def __init__(self, preprocessing_dict: dict, subject_id: int):
        super().__init__(preprocessing_dict, subject_id)

    def prepare_data(self) -> None:
        self.dataset = load_bcic4(subject_ids=self.all_subject_ids, dataset="2a",
                                  preprocessing_dict=self.preprocessing_dict)

    def setup(self, stage: Optional[str] = None) -> None:
        if self.dataset is None:
            self.prepare_data()

        use_ea = self.preprocessing_dict.get("use_ea", False)
        eps_ea = self.preprocessing_dict.get("ea_eps", 1e-6)

        # split the data
        splitted_ds = self.dataset.split("subject")
        train_subjects = [
            subj_id for subj_id in self.all_subject_ids if subj_id != self.subject_id]

        # ── Load data: with or without per-subject EA ──
        if use_ea:
            print(f"\n[EA] LOSO fold: test subject = {self.subject_id}")
            print(f"[EA] Source subjects: {train_subjects}")

            # --- Per-subject EA for source subjects (train) ---
            X_list, y_list = [], []
            for subj_id in train_subjects:
                subj_ds = splitted_ds[str(subj_id)].split("session")["session_T"]
                X_subj = np.concatenate([run.windows.load_data()._data for run in subj_ds.datasets], axis=0)
                y_subj = np.concatenate([run.y for run in subj_ds.datasets], axis=0)
                print(f"[EA] Source subject {subj_id}: pre-EA shape = {X_subj.shape}")
                X_subj = apply_ea_to_subject(X_subj, eps=eps_ea)
                print(f"[EA] Source subject {subj_id}: post-EA shape = {X_subj.shape}")
                if np.any(np.isnan(X_subj)):
                    print(f"[EA] WARNING: NaN in source subject {subj_id} after EA!")
                X_list.append(X_subj)
                y_list.append(y_subj)
            X = np.concatenate(X_list, axis=0)
            y = np.concatenate(y_list, axis=0)

            # --- Per-subject EA for source subjects (val) ---
            X_val_list, y_val_list = [], []
            for subj_id in train_subjects:
                subj_ds = splitted_ds[str(subj_id)].split("session")["session_E"]
                X_subj = np.concatenate([run.windows.load_data()._data for run in subj_ds.datasets], axis=0)
                y_subj = np.concatenate([run.y for run in subj_ds.datasets], axis=0)
                print(f"[EA] Source subject {subj_id} val: pre-EA shape = {X_subj.shape}")
                X_subj = apply_ea_to_subject(X_subj, eps=eps_ea)
                print(f"[EA] Source subject {subj_id} val: post-EA shape = {X_subj.shape}")
                X_val_list.append(X_subj)
                y_val_list.append(y_subj)
            X_val = np.concatenate(X_val_list, axis=0)
            y_val = np.concatenate(y_val_list, axis=0)

            # --- Per-subject EA for target subject (test) ---
            # NOTE: EA on target subject uses only unlabeled EEG signals.
            # No target labels are used for covariance estimation.
            test_dataset = splitted_ds[str(self.subject_id)].split("session")["session_E"]
            X_test = np.concatenate([run.windows.load_data()._data for run in test_dataset.datasets], axis=0)
            y_test = np.concatenate([run.y for run in test_dataset.datasets], axis=0)
            print(f"[EA] Target subject {self.subject_id}: pre-EA shape = {X_test.shape}")
            X_test = apply_ea_to_subject(X_test, eps=eps_ea)
            print(f"[EA] Target subject {self.subject_id}: post-EA shape = {X_test.shape}")
            if np.any(np.isnan(X_test)):
                print(f"[EA] WARNING: NaN in target subject {self.subject_id} after EA!")
            print(f"[EA] Final: train={X.shape}, val={X_val.shape}, test={X_test.shape}")

        else:
            # --- Original baseline path (no EA) ---
            train_datasets = [splitted_ds[str(subj_id)].split("session")["session_T"]
                              for subj_id in train_subjects]
            val_datasets = [splitted_ds[str(subj_id)].split("session")["session_E"]
                            for subj_id in train_subjects]
            test_dataset = splitted_ds[str(self.subject_id)].split("session")["session_E"]

            X = np.concatenate([run.windows.load_data()._data for train_dataset in
                                train_datasets for run in train_dataset.datasets], axis=0)
            y = np.concatenate([run.y for train_dataset in train_datasets for run in
                                train_dataset.datasets], axis=0)
            X_val = np.concatenate([run.windows.load_data()._data for val_dataset in
                                val_datasets for run in val_dataset.datasets], axis=0)
            y_val = np.concatenate([run.y for val_dataset in val_datasets for run in
                                val_dataset.datasets], axis=0)
            X_test = np.concatenate([run.windows.load_data()._data for run in test_dataset.datasets],
                                    axis=0)
            y_test = np.concatenate([run.y for run in test_dataset.datasets], axis=0)

        # scale data
        if self.preprocessing_dict["z_scale"]:
            X, X_val, X_test = BaseDataModule._z_scale_tvt(X, X_val, X_test)

        self.train_dataset = BaseDataModule._make_tensor_dataset(X, y)
        self.val_dataset = BaseDataModule._make_tensor_dataset(X_val, y_val)
        self.test_dataset = BaseDataModule._make_tensor_dataset(X_test, y_test)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset,
                          batch_size=self.preprocessing_dict["batch_size"],
                          num_workers=self.preprocessing_dict.get("num_workers", os.cpu_count() // 2),
                          pin_memory=True,
                        )
