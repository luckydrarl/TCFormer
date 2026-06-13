"""
Euclidean Alignment (EA) for EEG per-subject preprocessing.

EA is an unsupervised input-level alignment that reduces cross-subject
distribution shift by whitening each subject's EEG signals using their
own reference covariance matrix.

Reference:
    He, H., & Wu, D. (2020). Transfer Learning for Brain-Computer Interfaces:
    A Euclidean Space Data Alignment Approach. IEEE TNSRE.

IMPORTANT: EA only uses unlabeled EEG signals for covariance estimation
and does NOT use target labels.
"""

import numpy as np


def compute_reference_covariance(X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Compute subject-level reference covariance matrix R from all trials.

    For each trial X_i of shape [C, T]:
        cov_i = X_i @ X_i.T
        cov_i = cov_i / trace(cov_i)           # trace normalization
        R = mean_i(cov_i) + eps * I

    Args:
        X: numpy array, shape [N, C, T]  (N trials, C channels, T timepoints)
        eps: regularization constant for numerical stability

    Returns:
        R: numpy array, shape [C, C]
    """
    N, C, T = X.shape
    R = np.zeros((C, C), dtype=X.dtype)

    for i in range(N):
        cov_i = X[i] @ X[i].T              # [C, C]
        trace = np.trace(cov_i)
        if trace > 0:
            cov_i = cov_i / trace           # trace normalization
        R += cov_i

    R = R / N                               # average over trials
    R = R + eps * np.eye(C, dtype=X.dtype)  # regularize

    return R


def inv_sqrtm(R: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Compute the inverse square root of a symmetric positive semi-definite matrix.

    Using eigen decomposition:
        R = V @ diag(w) @ V.T
        R^{-1/2} = V @ diag(1 / sqrt(max(w, eps))) @ V.T

    Args:
        R: numpy array, shape [C, C]
        eps: minimum eigenvalue clamp for numerical stability

    Returns:
        R_inv_sqrt: numpy array, shape [C, C]
    """
    w, V = np.linalg.eigh(R)                # eigenvalues, eigenvectors (already sorted)
    w = np.maximum(w, eps)                   # clamp small eigenvalues
    w_inv_sqrt = 1.0 / np.sqrt(w)           # 1 / sqrt(w)
    R_inv_sqrt = V @ np.diag(w_inv_sqrt) @ V.T
    return R_inv_sqrt


def apply_ea_to_subject(X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply Euclidean Alignment to all trials of a single subject.

    Steps:
        1. Compute subject reference covariance R from all trials
        2. Compute R^{-1/2}
        3. X_aligned[i] = R^{-1/2} @ X[i]   for each trial

    Args:
        X: numpy array, shape [N, C, T]
        eps: regularization constant

    Returns:
        X_aligned: numpy array, shape [N, C, T]  (same shape & dtype)

    Note:
        If input is a torch tensor, it will be converted to numpy,
        aligned, and returned as numpy. The caller should convert back
        if needed.
    """
    # Handle torch tensor
    import torch
    is_torch = torch.is_tensor(X)
    if is_torch:
        device = X.device
        dtype_torch = X.dtype
        X_np = X.cpu().numpy().astype(np.float64)
    else:
        X_np = X.astype(np.float64)

    # Compute per-subject reference covariance and its inverse sqrt
    R = compute_reference_covariance(X_np, eps=eps)
    R_inv_sqrt = inv_sqrtm(R, eps=eps)

    # Apply alignment to each trial
    N = X_np.shape[0]
    X_aligned_np = np.zeros_like(X_np)
    for i in range(N):
        X_aligned_np[i] = R_inv_sqrt @ X_np[i]   # [C, C] @ [C, T] -> [C, T]

    if is_torch:
        X_aligned = torch.from_numpy(X_aligned_np).to(device).type(dtype_torch)
    else:
        X_aligned = X_aligned_np.astype(X.dtype) if X.dtype != np.float64 else X_aligned_np

    return X_aligned
