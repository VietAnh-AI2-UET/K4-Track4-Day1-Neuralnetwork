"""plots.py — PSEUDO-CODE. Bạn phải tự hoàn thiện mọi hàm có `raise NotImplementedError`.

Ảnh biểu đồ là sản phẩm nộp (xem README mục 6): mỗi thí nghiệm một ảnh figures/<exp_id>.png.
Khi notebook chạy trong code/, lưu vào "../figures/" (ví dụ path = f"../figures/{exp_id}.png").
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def plot_run(result: dict, path: str) -> None:
    """Vẽ MỘT thí nghiệm thành một ảnh PNG có ít nhất 3 ô:
         (1) train_loss và val_loss theo epoch (cùng một trục)
         (2) val_acc (và nên có val_macro_f1) theo epoch
         (3) grad_norm theo epoch (đo TRƯỚC khi clip)
    Yêu cầu: tiêu đề ghi exp_id và cấu hình chính (optimizer, lr, batch, ...), có nhãn trục và chú thích.
    Các bước: fig, axes = plt.subplots(1, 3, figsize=...); plot; set_title/xlabel/legend;
              fig.savefig(path, dpi=..., bbox_inches="tight"); plt.close(fig)
    Gợi ý: đánh dấu best_epoch bằng đường thẳng đứng.
    """
    # Lấy cấu hình và các chỉ số đã được run_experiment ghi lại sau mỗi vòng học.
    cfg = result["cfg"]
    history = result["history"]
    epochs = history["epoch"]
    # Mỗi đường phải có đúng một giá trị cho mỗi vòng học để vẽ đúng vị trí.
    for key in ("train_loss", "val_loss", "val_acc", "val_macro_f1", "grad_norm"):
        if len(history[key]) != len(epochs):
            raise ValueError(f"history[{key!r}] must have the same length as history['epoch']")

    # Tạo thư mục chứa ảnh nếu chưa có; path là đường dẫn đầy đủ tới file cần lưu.
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Tạo một ảnh gồm ba ô nằm ngang; axes chứa ba vùng dùng để vẽ.
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    try:
        # Ô 1: loss là mức sai lệch của dự đoán; giá trị thấp hơn thường tốt hơn.
        axes[0].plot(epochs, history["train_loss"], label="Train loss")
        axes[0].plot(epochs, history["val_loss"], label="Validation loss")
        axes[0].set_title("Train and validation loss")
        axes[0].set_ylabel("Loss")

        # Ô 2: accuracy là tỷ lệ dự đoán đúng; macro-F1 lấy trung bình F1 của các lớp.
        # F1 kết hợp mức đoán đúng và mức tìm đủ mẫu của từng lớp; hai chỉ số càng cao càng tốt.
        axes[1].plot(epochs, history["val_acc"], label="Validation accuracy")
        axes[1].plot(epochs, history["val_macro_f1"], label="Validation macro-F1")
        axes[1].set_title("Validation accuracy and macro-F1")
        axes[1].set_ylabel("Score (0–1)")
        axes[1].set_ylim(0, 1)

        # Ô 3: độ lớn gradient cho biết loss nhạy với thay đổi tham số đến mức nào.
        # Giá trị đã ghi là trước khi giới hạn gradient, giúp thấy những vòng tăng đột biến.
        axes[2].plot(epochs, history["grad_norm"], label="Gradient norm before clipping")
        axes[2].set_title("Gradient norm")
        axes[2].set_ylabel("Mean gradient norm")

        # Đánh dấu vòng được chọn theo validation trên cả ba ô để dễ đối chiếu.
        best_epoch = result["summary"].get("best_epoch", 0)
        for ax in axes:
            if best_epoch in epochs:
                ax.axvline(best_epoch, color="gray", linestyle="--",
                           label=f"Best epoch: {best_epoch}")
            ax.set_xlabel("Epoch")  # Một epoch là một lượt học qua toàn bộ tập train.
            ax.grid(True, alpha=0.3)  # Lưới mờ giúp đọc giá trị mà không che đường vẽ.
            ax.legend()  # Hiện tên các đường và ý nghĩa của đường đánh dấu.

        # Ghi cấu hình lên ảnh để biết biểu đồ thuộc thí nghiệm nào khi xem lại.
        config_keys = ("loss", "optimizer", "lr", "batch", "hidden", "dropout",
                       "clip_norm", "precision", "init", "seed")
        config_text = ", ".join(f"{key}={cfg[key]}" for key in config_keys if key in cfg)
        fig.suptitle(f"{cfg.get('exp_id', 'Experiment')}\n{config_text}", fontsize=10)
        fig.tight_layout()  # Tự điều chỉnh khoảng cách để tiêu đề và nhãn không chồng nhau.
        fig.savefig(output_path, format="png", dpi=150, bbox_inches="tight")
    finally:
        # Luôn giải phóng ảnh, kể cả khi lưu lỗi, tránh giữ bộ nhớ qua nhiều lần chạy.
        plt.close(fig)


def plot_compare(results: list[dict], metric: str, path: str, title: str = "") -> None:
    """Vẽ chồng một chỉ số (ví dụ "val_loss", "val_macro_f1", "grad_norm") của nhiều thí nghiệm
    trên cùng một trục, mỗi thí nghiệm một đường, chú thích bằng exp_id.

    Dùng cho ảnh figures/compare_<nhóm>.png (ví dụ compare_optimizer.png).
    """
    raise NotImplementedError  # TODO
