"""data.py — PSEUDO-CODE. Bạn phải tự hoàn thiện mọi hàm có `raise NotImplementedError`.

Nhiệm vụ: nạp tập train/eval đã chia sẵn, tách validation từ train, chuẩn hoá, đưa lên thiết bị.

Điều kiện trước: đã chạy `python scripts/split_data.py` (tạo data/processed/train.npz, eval.npz).

Quy ước dữ liệu (xem README mục 2 và 3):
    X : float32, shape (N, 54)   — 10 cột đầu là số liên tục, 44 cột sau là nhị phân (one-hot)
    y : int64,   shape (N,)      — nhãn 0..6
Tập eval CHỈ dùng để chấm điểm cuối. Không dùng nó để chọn cấu hình, chuẩn hoá hay dừng sớm.
"""
from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split

N_NUMERIC = 10  # số cột liên tục cần chuẩn hoá (cột 0..9)


def load_split(processed_dir: str = "data/processed"):
    """Nạp train và eval từ file .npz.

    Trả về: X_train_full, y_train_full, X_eval, y_eval, eval_row_id
    Các bước:
      1. np.load(f"{processed_dir}/train.npz") -> khoá "X", "y"
      2. np.load(f"{processed_dir}/eval.npz")  -> khoá "X", "y", "row_id"
      3. assert shape/dtype đúng quy ước ở đầu file
    """
    processed_dir = Path(processed_dir)
    with np.load(processed_dir / "train.npz", allow_pickle=False) as train:
        X_train_full = train["X"]
        y_train_full = train["y"]
    with np.load(processed_dir / "eval.npz", allow_pickle=False) as evaluation:
        X_eval = evaluation["X"]
        y_eval = evaluation["y"]
        eval_row_id = evaluation["row_id"]

    for name, X, y in (("train", X_train_full, y_train_full),
                       ("eval", X_eval, y_eval)):
        assert X.ndim == 2 and X.shape[1] == 54, f"{name}: X must have shape (N, 54)"
        assert X.dtype == np.float32, f"{name}: X must be float32"
        assert y.shape == (len(X),), f"{name}: y must have shape (N,)"
        assert y.dtype == np.int64, f"{name}: y must be int64"
        assert np.all((y >= 0) & (y <= 6)), f"{name}: labels must be in 0..6"
    assert eval_row_id.shape == (len(X_eval),), "eval: row_id must have shape (N,)"

    return X_train_full, y_train_full, X_eval, y_eval, eval_row_id


def make_val_split(X, y, val_fraction: float = 0.2, seed: int = 42):
    """Tách validation TỪ train (không đụng eval). Phân tầng theo nhãn.

    Trả về: X_tr, y_tr, X_val, y_val
    Gợi ý: sklearn.model_selection.train_test_split(..., stratify=y, random_state=seed)
    Dùng CÙNG seed và val_fraction cho mọi thí nghiệm để so sánh công bằng.
    """
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=val_fraction, random_state=seed, stratify=y
    )
    return X_tr, y_tr, X_val, y_val


def fit_standardizer(X_tr):
    """Tính mean và std của N_NUMERIC cột đầu CHỈ trên tập train (sau khi tách val).

    Trả về: mean (shape (10,)), std (shape (10,))
    Câu hỏi: vì sao không được tính trên toàn bộ dữ liệu hay trên eval?
    """
    numeric = X_tr[:, :N_NUMERIC]  # Chỉ lấy 10 cột số của tập train đã tách validation.
    mean = numeric.mean(axis=0, dtype=np.float64)  # Tính trung bình từng cột; dùng 64 bit để giảm sai số khi cộng nhiều hàng.
    std = numeric.std(axis=0, dtype=np.float64)  # Tính độ lệch chuẩn (mức phân tán quanh trung bình) từng cột.
    return mean, std  # Giữ hai mảng này để dùng chung cho train, validation và eval.


def apply_standardizer(X, mean, std):
    """Trả về bản sao của X, trong đó 10 cột đầu được (x - mean) / std; 44 cột nhị phân giữ nguyên.

    Chú ý: không sửa X tại chỗ nếu bạn còn dùng lại nó; chú ý std = 0 (nếu có).
    """
    X_scaled = X.copy()  # Tạo bản sao để không thay đổi dữ liệu gốc; giữ kiểu float32 của X.
    safe_std = np.where(std == 0, 1.0, std)  # Dùng 1 thay cho std bằng 0 để tránh chia cho 0; không sửa mảng std gốc.
    X_scaled[:, :N_NUMERIC] = (X[:, :N_NUMERIC] - mean) / safe_std  # Chỉ chuẩn hoá 10 cột số; các cột nhị phân giữ nguyên.
    return X_scaled  # Trả về dữ liệu đã chuẩn hoá trong một mảng riêng.


def prepare_data(device: str, val_fraction: float = 0.2, seed: int = 42,
                 processed_dir: str = "data/processed") -> dict:
    """Gộp các bước trên và đưa TOÀN BỘ dữ liệu lên `device` một lần (không dùng DataLoader).

    Trả về dict gồm các tensor trên device:
        X_tr, y_tr, X_val, y_val, X_eval, y_eval        (y là int64)
    và các mảng numpy: eval_row_id
    Các bước:
      1. load_split -> make_val_split -> fit_standardizer (chỉ trên X_tr)
      2. apply_standardizer cho X_tr, X_val, X_eval bằng CÙNG mean/std
      3. torch.tensor(..., device=device); X là float32, y là int64
      4. in ra kích thước các tập và accuracy của chiến lược "luôn đoán lớp đa số" trên val
    """
    X_full, y_full, X_eval, y_eval, row_id = load_split(processed_dir)
    X_tr, y_tr, X_val, y_val = make_val_split(X_full, y_full, val_fraction, seed)
    mean, std = fit_standardizer(X_tr)
    data = {"eval_row_id": row_id}
    for name, X, y in (("tr", X_tr, y_tr), ("val", X_val, y_val),
                       ("eval", X_eval, y_eval)):
        data[f"X_{name}"] = torch.tensor(
            apply_standardizer(X, mean, std), dtype=torch.float32, device=device
        )
        data[f"y_{name}"] = torch.tensor(y, dtype=torch.int64, device=device)
        print(f"X_{name}: {tuple(data[f'X_{name}'].shape)}, y_{name}: {tuple(data[f'y_{name}'].shape)}")
    majority_class = int(np.bincount(y_tr, minlength=7).argmax())
    majority_acc = float(np.mean(y_val == majority_class))
    print(f"Majority class (from train): {majority_class}; val accuracy: {majority_acc:.6f}")
    return data


def iterate_batches(X, y, batch_size: int, generator: torch.Generator | None = None, shuffle: bool = True):
    """Generator trả về từng cặp (xb, yb), thay cho DataLoader.

    Các bước:
      1. nếu shuffle: perm = torch.randperm(len(X), generator=generator, device=X.device); ngược lại arange
      2. for i in range(0, N, batch_size): idx = perm[i:i+batch_size]; yield X[idx], y[idx]
    Chú ý: batch cuối có thể nhỏ hơn batch_size; hãy quyết định bạn xử lý thế nào và ghi lại.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if len(X) != len(y) or X.device != y.device:
        raise ValueError("X and y must have the same length and device")
    if shuffle:
        perm = torch.randperm(len(X), generator=generator, device=X.device)
    else:
        perm = torch.arange(len(X), device=X.device)
    # Giữ lô cuối nhỏ hơn batch_size để mỗi mẫu được dùng đúng một lần.
    for start in range(0, len(X), batch_size):
        idx = perm[start:start + batch_size]
        yield X[idx], y[idx]
