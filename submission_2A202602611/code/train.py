"""train.py — PSEUDO-CODE. Bạn phải tự hoàn thiện mọi hàm có `raise NotImplementedError`.

Gồm: đặt seed, đánh giá, vòng huấn luyện `run_experiment(cfg, data)`, dự đoán và ghi file nộp.
Mọi thí nghiệm chỉ là *đổi dict cfg* rồi gọi lại run_experiment (xem GUIDE, Part 2).

Mọi chỉ số (loss, accuracy, macro-F1) dùng cùng định nghĩa với scripts/evaluate.py.
"""
from __future__ import annotations

import random  # Thư viện tạo số ngẫu nhiên có sẵn của Python.
import time

import numpy as np
import torch
import torch.nn.functional as F

from data import iterate_batches
from model import MLP, EXPECTED_PARAMS, count_params
from optimizer import build_optimizer, clip_gradients

# Cấu hình mặc định = BASELINE (M-base). `lr` do bạn tự chọn bằng val rồi điền vào.
DEFAULT_CFG = dict(
    exp_id="base-s1", group="baseline", description="Baseline M-base",
    loss="ce",                 # "ce" | "mse"
    optimizer="sgd_momentum",  # "sgd" | "sgd_momentum" | "adam" | "adamw"
    lr=None,                   # TODO: chọn bằng val, không dùng eval
    weight_decay=0.0, momentum=0.9,
    batch=512, epochs=20,
    hidden=(256, 128), dropout=0.0, init="he",
    clip_norm=None,            # None = không clip; hoặc số, ví dụ 1.0
    precision="fp32",          # "fp32" | "fp16" | "bf16"
    seed=1,
)


def set_seed(seed: int) -> None:
    """Đặt seed cho random, numpy, torch (và torch.cuda nếu có)."""
    random.seed(seed)  # Đặt điểm bắt đầu cho chuỗi số ngẫu nhiên của Python.
    np.random.seed(seed)  # Đặt điểm bắt đầu cho bộ tạo số ngẫu nhiên mặc định của NumPy.
    torch.manual_seed(seed)  # Đặt seed cho PyTorch để có thể lặp lại khởi tạo trọng số và các thao tác ngẫu nhiên.
    if torch.cuda.is_available():  # Kiểm tra máy có GPU CUDA dùng được hay không.
        torch.cuda.manual_seed_all(seed)  # Đặt cùng seed cho bộ tạo số ngẫu nhiên trên tất cả GPU CUDA.


# Chưa hiểu lắm phần f1 này
def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """macro-F1 = trung bình cộng F1 của 7 lớp; F1_c = 2PR/(P+R), bằng 0 nếu P+R = 0.

    cm: ma trận nhầm lẫn (7, 7), hàng = nhãn thật, cột = dự đoán.
    """
    tp = np.diag(cm).astype(float)  # Đường chéo chứa số mẫu dự đoán đúng của từng lớp; đổi sang số thực để tính tỷ lệ.
    predicted = cm.sum(axis=0)  # Tổng mỗi cột là số mẫu được model dự đoán thuộc lớp đó.
    actual = cm.sum(axis=1)  # Tổng mỗi hàng là số mẫu thực sự thuộc lớp đó.
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)  # Tỷ lệ đúng trong các mẫu dự đoán thuộc lớp; dùng 0 nếu không có mẫu dự đoán.
    recall = np.divide(tp, actual, out=np.zeros_like(tp), where=actual > 0)  # Tỷ lệ tìm đúng trong các mẫu thực sự thuộc lớp; dùng 0 nếu lớp không có mẫu thật.
    denominator = precision + recall  # Tính mẫu số của công thức F1 cho từng lớp.
    f1 = np.divide(2 * precision * recall, denominator, out=np.zeros_like(tp), where=denominator > 0)  # Tính F1 từng lớp; dùng 0 khi mẫu số bằng 0 để tránh chia cho 0.
    return float(f1.mean())  # Lấy trung bình F1 của tất cả 7 lớp, kể cả lớp có F1 bằng 0; trả về số Python.


@torch.no_grad()  # Không tính gradient khi dự đoán để tiết kiệm bộ nhớ và thời gian.
def predict(model, X, batch_size: int = 8192) -> torch.Tensor:
    """Trả về nhãn dự đoán int64 (N,) = argmax của logits.

    Các bước: model.eval(); duyệt X theo từng lô (không cần xáo); gom argmax(dim=1); torch.cat.
    """
    if batch_size <= 0:  # Mỗi lô phải chứa ít nhất một mẫu.
        raise ValueError("batch_size must be positive")  # Báo lỗi nếu kích thước lô không hợp lệ.
    model.eval()  # Chuyển model sang chế độ đánh giá, tắt dropout khi dự đoán.
    if len(X) == 0:  # Xử lý trường hợp dữ liệu không có mẫu nào.
        return torch.empty(0, dtype=torch.int64, device=X.device)  # Trả tensor nhãn rỗng trên cùng thiết bị với dữ liệu.
    predictions = []  # Lưu nhãn dự đoán của từng lô trước khi ghép lại.
    for start in range(0, len(X), batch_size):  # Duyệt dữ liệu đúng thứ tự ban đầu, không xáo trộn.
        xb = X[start:start + batch_size]  # Lấy lô hiện tại; lô cuối có thể ít mẫu hơn batch_size.
        logits = model(xb)  # Tính điểm của các lớp cho mỗi mẫu; X và model cần ở cùng thiết bị.
        predictions.append(logits.argmax(dim=1))  # Chọn chỉ số lớp có điểm cao nhất cho từng mẫu, tạo nhãn int64.
    return torch.cat(predictions, dim=0)  # Ghép nhãn các lô thành tensor một chiều có N phần tử theo thứ tự dữ liệu.


# Tắt tính gradient vì đánh giá không cập nhật tham số của model.
@torch.no_grad()
def evaluate(model, X, y, loss_name: str = "ce", batch_size: int = 8192) -> dict:
    """Trả về dict(loss, acc, macro_f1) ở chế độ eval() (dropout tắt) và no_grad.

    Các bước:
      1. model.eval()
      2. tính logits theo từng lô; cộng dồn tổng loss (reduction="sum") rồi chia N cuối cùng
      3. pred = argmax; acc = (pred == y).mean()
      4. dựng ma trận nhầm lẫn 7x7 -> macro_f1_from_confusion
    Dùng hàm này cho: train loss (trên toàn bộ hoặc một tập con CỐ ĐỊNH của train), val, và eval cuối cùng.
    """
    # Kiểm tra mỗi lô có ít nhất một mẫu.
    if batch_size <= 0:
        # Báo lỗi khi kích thước lô không hợp lệ.
        raise ValueError("batch_size must be positive")
    # Cần dữ liệu không rỗng và mỗi mẫu có một nhãn tương ứng.
    if len(X) == 0 or len(X) != len(y):
        # Tránh chia cho 0 hoặc ghép sai mẫu với nhãn.
        raise ValueError("X and y must be non-empty and have the same length")
    # Chỉ chấp nhận hai loại loss được hỗ trợ trong bài.
    if loss_name not in ("ce", "mse"):
        # Báo lỗi thay vì âm thầm dùng nhầm loại loss.
        raise ValueError(f"Unknown loss: {loss_name!r}")
    # Chuyển sang chế độ đánh giá để tắt dropout, giúp so sánh train và val công bằng.
    model.eval()
    # Cộng dồn loss của tất cả mẫu để tính trung bình sau cùng.
    total_loss = 0.0
    # Tạo bảng đếm: hàng là lớp thật, cột là lớp dự đoán.
    cm = torch.zeros((7, 7), dtype=torch.int64, device=X.device)
    # Duyệt lần lượt từng lô, giữ cả lô cuối nhỏ hơn batch_size.
    for start in range(0, len(X), batch_size):
        # Lấy dữ liệu của lô hiện tại; dữ liệu và model cần ở cùng thiết bị.
        xb = X[start:start + batch_size]
        # Lấy nhãn tương ứng với đúng các mẫu trong lô.
        yb = y[start:start + batch_size]
        # Tính điểm dự đoán cho bảy lớp của mỗi mẫu.
        logits = model(xb)
        # Nhánh cross-entropy: dùng điểm thô và nhãn lớp.
        if loss_name == "ce":
            # Cộng loss của mọi mẫu trong lô, chưa lấy trung bình.
            batch_loss = F.cross_entropy(logits, yb, reduction="sum")
        # Nhánh MSE: đo sai lệch bình phương giữa điểm dự đoán và nhãn dạng vector.
        else:
            # Đổi nhãn thành bảy số: lớp thật là 1, các lớp khác là 0.
            targets = F.one_hot(yb, num_classes=7).to(dtype=logits.dtype)
            # Lấy trung bình trên bảy lớp rồi cộng trên các mẫu; cuối cùng sẽ chia tiếp cho số mẫu.
            batch_loss = F.mse_loss(logits, targets, reduction="sum") / 7
        # Lấy số Python và cộng loss của lô vào tổng loss.
        total_loss += batch_loss.item()
        # Chọn lớp có điểm cao nhất cho mỗi mẫu.
        preds = logits.argmax(dim=1)
        # Mã hóa cặp lớp thật/dự đoán thành chỉ số 0–48 và đếm số lần xuất hiện.
        counts = torch.bincount(yb * 7 + preds, minlength=49)
        # Đưa các số đếm về bảng 7×7 và cộng vào bảng chung.
        cm += counts.reshape(7, 7)
    # Tính loss trung bình theo mẫu, không để lô cuối nhỏ làm lệch kết quả.
    loss = total_loss / len(X)
    # Accuracy là số mẫu dự đoán đúng trên đường chéo chia tổng số mẫu.
    acc = cm.diag().sum().item() / len(X)
    # Chuyển bảng đếm về NumPy trên CPU và tính trung bình F1 của bảy lớp.
    macro_f1 = macro_f1_from_confusion(cm.cpu().numpy())
    # Trả ba chỉ số để ghi lịch sử và so sánh cấu hình.
    return {"loss": loss, "acc": acc, "macro_f1": macro_f1}


def compute_loss(logits, y, loss_name: str):
    """"ce"  : cross-entropy nhận logit thô và nhãn int64 (F.cross_entropy).
       "mse" : MSE giữa logit và one-hot của y, trung bình trên cả mẫu và lớp.
    """
    # Chọn cross-entropy: logits là điểm thô của các lớp, y là chỉ số lớp thật.
    if loss_name == "ce":
        # Lấy loss trung bình trên các mẫu; không cần đổi logits thành xác suất trước.
        return F.cross_entropy(logits, y, reduction="mean")
    # Chọn MSE: đo sai lệch bình phương giữa điểm dự đoán và nhãn dạng vector.
    if loss_name == "mse":
        # Đổi mỗi nhãn thành vector: lớp thật là 1, các lớp còn lại là 0.
        targets = F.one_hot(y, num_classes=logits.shape[1])
        # Đổi nhãn sang cùng kiểu số thực với logits để tính loss.
        targets = targets.to(dtype=logits.dtype)
        # Lấy trung bình sai lệch bình phương trên tất cả mẫu và lớp, khớp cách đo trong evaluate.
        return F.mse_loss(logits, targets, reduction="mean")
    # Báo lỗi khi tên loss không được hỗ trợ, tránh âm thầm dùng sai công thức.
    raise ValueError(f"Unknown loss: {loss_name!r}")


def run_experiment(cfg: dict, data: dict) -> dict:
    """Huấn luyện một cấu hình và trả về lịch sử + tóm tắt.

    Args:
        cfg : dict cấu hình (xem DEFAULT_CFG)
        data: kết quả của data.prepare_data (tensor X_tr, y_tr, X_val, y_val, X_eval, y_eval trên device)

    Trả về dict:
        {"cfg": cfg,
         "history": {"epoch": [...], "train_loss": [...], "val_loss": [...], "val_acc": [...],
                     "val_macro_f1": [...], "grad_norm": [...], "epoch_time_s": [...]},
         "summary": {"step0_loss", "best_val_loss", "best_epoch", "final_train_loss", "final_val_loss",
                     "val_acc", "val_macro_f1", "time_per_epoch_s", "peak_mem_MB", "diverged"},
         "best_state": state_dict của epoch có val_loss thấp nhất (giữ trong RAM để dự đoán eval)}
    (tên khoá của summary trùng tên cột trong experiments.xlsx)

    Các bước:
      0. set_seed(cfg["seed"]); tạo model = MLP(...), assert count_params(model) == EXPECTED_PARAMS[hidden]
         chuyển model lên device; tạo optimizer = build_optimizer(...)
         nếu precision == "fp16": scaler = torch.amp.GradScaler(...)
      1. step0_loss = evaluate(model, X_val, y_val)["loss"]   # TRƯỚC bước cập nhật đầu tiên; kỳ vọng ≈ ln 7
      2. for epoch in 1..epochs:
           model.train()
           for xb, yb in iterate_batches(X_tr, y_tr, cfg["batch"], generator):
               with torch.autocast(...)  nếu precision != "fp32":   # chỉ bọc forward + loss
                   logits = model(xb); loss = compute_loss(logits, yb, cfg["loss"])
               optimizer.zero_grad(set_to_none=True)
               backward (qua scaler nếu fp16)
               nếu fp16 và có clip: scaler.unscale_(optimizer)  TRƯỚC khi clip
               gn = clip_gradients(model.parameters(), cfg["clip_norm"])   # chuẩn TRƯỚC khi cắt; ghi lại
               bước cập nhật (scaler.step(optimizer); scaler.update() nếu fp16, ngược lại optimizer.step())
               nếu loss là NaN/inf: đặt diverged=True và dừng sớm, ĐỪNG để notebook treo
           cuối epoch (dùng evaluate, chế độ eval):
               train_loss trên toàn bộ train (hoặc 1 tập con CỐ ĐỊNH ~50 000 mẫu), val_loss/val_acc/val_macro_f1
               grad_norm trung bình của epoch; thời gian epoch (torch.cuda.synchronize() nếu dùng GPU)
               nếu val_loss tốt nhất từ trước tới giờ: lưu best_state (bản sao state_dict) và best_epoch
      3. tổng hợp summary tại best_epoch (val_acc, val_macro_f1 lấy ở best_epoch); peak_mem_MB nếu có GPU
    TUYỆT ĐỐI không đưa X_eval vào hàm này để chọn epoch/cấu hình. Chỉ dùng val.
    """
    # Bổ sung giá trị mặc định vào bản sao, không sửa cấu hình người gọi truyền vào.
    cfg = {**DEFAULT_CFG, **cfg}
    if cfg["lr"] is None or not np.isfinite(cfg["lr"]) or cfg["lr"] <= 0:
        raise ValueError("lr must be a finite positive number")
    if cfg["epochs"] < 1 or cfg["batch"] < 1:
        raise ValueError("epochs and batch must be positive")
    if cfg["loss"] not in ("ce", "mse"):
        raise ValueError(f"Unknown loss: {cfg['loss']!r}")
    if cfg["precision"] not in ("fp32", "fp16", "bf16"):
        raise ValueError(f"Unknown precision: {cfg['precision']!r}")
    if cfg["clip_norm"] is not None and (
        not np.isfinite(cfg["clip_norm"]) or cfg["clip_norm"] <= 0
    ):
        raise ValueError("clip_norm must be None or a finite positive number")

    # Chỉ lấy train và validation; tập eval dành cho chấm điểm cuối cùng.
    X_tr, y_tr = data["X_tr"], data["y_tr"]
    X_val, y_val = data["X_val"], data["y_val"]
    device = X_tr.device
    if any(t.device != device for t in (y_tr, X_val, y_val)):
        raise ValueError("Train and validation tensors must share the same device")
    if len(X_tr) == 0 or len(X_tr) != len(y_tr):
        raise ValueError("Training tensors must be non-empty and have the same length")
    if cfg["precision"] == "fp16" and device.type != "cuda":
        raise ValueError("fp16 training requires a CUDA device; use fp32 or bf16 on CPU")
    if cfg["precision"] == "bf16" and device.type == "cuda":
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported():
                raise ValueError("This CUDA device does not support bf16")

    set_seed(cfg["seed"])
    hidden = tuple(cfg["hidden"])
    if hidden not in EXPECTED_PARAMS:
        raise ValueError(f"Unsupported hidden architecture: {hidden!r}")
    model = MLP(hidden=hidden, dropout=cfg["dropout"], init=cfg["init"]).to(device)
    assert count_params(model) == EXPECTED_PARAMS[hidden]
    optimizer = build_optimizer(
        cfg["optimizer"], model.parameters(), lr=cfg["lr"],
        weight_decay=cfg["weight_decay"], momentum=cfg["momentum"],
    )
    # Bộ xáo trộn riêng giúp thứ tự các lô lặp lại được với cùng seed.
    generator = torch.Generator(device=device).manual_seed(cfg["seed"])
    use_amp = cfg["precision"] != "fp32"
    amp_dtype = torch.float16 if cfg["precision"] == "fp16" else torch.bfloat16
    # FP16 cần tăng tạm giá trị loss để gradient nhỏ không bị làm tròn về 0.
    scaler = torch.amp.GradScaler("cuda") if cfg["precision"] == "fp16" else None
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    initial_val = evaluate(model, X_val, y_val, loss_name=cfg["loss"])
    step0_loss = initial_val["loss"]
    history = {key: [] for key in (
        "epoch", "train_loss", "val_loss", "val_acc", "val_macro_f1",
        "grad_norm", "epoch_time_s",
    )}
    # Giữ bản sao trên CPU để tránh chiếm thêm bộ nhớ GPU; clone tránh bị cập nhật theo model.
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_epoch, best_val_loss = 0, float("inf")
    best_metrics = initial_val
    diverged = not np.isfinite(step0_loss)

    for epoch in range(1, cfg["epochs"] + 1):
        if diverged:
            break
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        model.train()  # Bật lại dropout vì evaluate chuyển model sang chế độ đánh giá.
        grad_norms = []
        for xb, yb in iterate_batches(X_tr, y_tr, cfg["batch"], generator):
            optimizer.zero_grad(set_to_none=True)  # Xóa gradient của lô trước.
            # Chỉ tính đầu ra và loss bằng độ chính xác thấp; trọng số vẫn là FP32.
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logits = model(xb)
                loss = compute_loss(logits, yb, cfg["loss"])
            if not torch.isfinite(loss).item():
                diverged = True  # Dừng trước khi loss lỗi làm hỏng trọng số.
                break
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)  # Đưa gradient về đúng tỷ lệ trước khi đo/cắt.
            else:
                loss.backward()  # Tính gradient: mức thay đổi loss theo từng tham số.
            gn = clip_gradients(model.parameters(), cfg["clip_norm"])
            grad_norms.append(gn)  # Hàm trả độ lớn gradient trước khi giới hạn.
            if scaler is not None:
                # Khi FP16 bị tràn gradient, scaler bỏ qua cập nhật và giảm hệ số tăng loss.
                scaler.step(optimizer)
                scaler.update()
            elif not np.isfinite(gn):
                diverged = True
                break
            else:
                optimizer.step()  # Cập nhật trọng số để giảm loss.

        # Đo lại cả hai tập bằng FP32, tắt dropout để so sánh loss công bằng.
        train_metrics = evaluate(model, X_tr, y_tr, loss_name=cfg["loss"])
        val_metrics = evaluate(model, X_val, y_val, loss_name=cfg["loss"])
        if not np.isfinite(train_metrics["loss"]) or not np.isfinite(val_metrics["loss"]):
            diverged = True
        if not diverged and val_metrics["loss"] < best_val_loss:
            best_val_loss, best_epoch = val_metrics["loss"], epoch
            best_metrics = val_metrics.copy()
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if device.type == "cuda":
            torch.cuda.synchronize(device)  # Chờ GPU xong trước khi chốt thời gian.
        epoch_values = {
            "epoch": epoch, "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"], "val_acc": val_metrics["acc"],
            "val_macro_f1": val_metrics["macro_f1"],
            "grad_norm": float(np.mean(grad_norms)) if grad_norms else float("nan"),
            "epoch_time_s": time.perf_counter() - started,
        }
        for key, value in epoch_values.items():
            history[key].append(value)

    # Nếu chưa hoàn tất vòng nào hợp lệ, best_epoch = 0 ứng với trọng số ban đầu.
    if best_epoch == 0:
        best_val_loss = step0_loss
    final_train_loss = history["train_loss"][-1] if history["epoch"] else evaluate(
        model, X_tr, y_tr, loss_name=cfg["loss"]
    )["loss"]
    summary = {
        "step0_loss": step0_loss, "best_val_loss": best_val_loss, "best_epoch": best_epoch,
        # Hai loss cuối là của vòng cuối; accuracy và F1 là của vòng được chọn.
        "final_train_loss": final_train_loss,
        "final_val_loss": history["val_loss"][-1] if history["epoch"] else step0_loss,
        "val_acc": best_metrics["acc"], "val_macro_f1": best_metrics["macro_f1"],
        "time_per_epoch_s": float(np.mean(history["epoch_time_s"])) if history["epoch"] else 0.0,
        "peak_mem_MB": torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else 0.0,
        "diverged": bool(diverged),
    }
    return {"cfg": cfg, "history": history, "summary": summary, "best_state": best_state}


def write_predictions(row_id, preds, path: str) -> None:
    """Ghi file nộp cho scripts/evaluate.py: CSV có tiêu đề `row_id,pred`.

    row_id : mảng row_id của tập eval (data["eval_row_id"])
    preds  : nhãn dự đoán int64 0..6 (cùng thứ tự với row_id)
    Phải đủ mọi dòng của tập eval, mỗi row_id đúng một lần.
    """
    raise NotImplementedError  # TODO


def final_eval(cfg: dict, result: dict, data: dict, pred_path: str) -> None:
    """Dùng MỘT LẦN cho cấu hình cuối cùng (và baseline): nạp best_state, dự đoán eval, ghi predictions.

    Các bước:
      1. model = MLP(...); model.load_state_dict(result["best_state"]); lên device
      2. preds = predict(model, data["X_eval"])  # fp32, eval mode
      3. write_predictions(data["eval_row_id"], preds.cpu().numpy(), pred_path)
      4. chạy `python scripts/evaluate.py --pred <pred_path>` và ghi kết quả vào bảng/báo cáo
    """
    raise NotImplementedError  # TODO
