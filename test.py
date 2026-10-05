import numpy as np

with np.load("data/processed/train.npz", allow_pickle=False) as data:
    print(data.files)          # Các tên mảng trong file, ví dụ ['X', 'y']

    X = data["X"]              # Lấy bảng đầu vào
    y = data["y"]              # Lấy mảng nhãn

    print("Kích thước X:", X.shape)  # Ví dụ (1000, 54): 1000 hàng, 54 cột
    print("5 hàng đầu:\n", X[:5])
    print("5 nhãn đầu:", y[:5])