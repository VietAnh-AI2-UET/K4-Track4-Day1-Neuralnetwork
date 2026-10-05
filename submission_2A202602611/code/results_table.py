"""results_table.py — PSEUDO-CODE. Bạn phải tự hoàn thiện mọi hàm có `raise NotImplementedError`.

Nhiệm vụ: lưu kết quả từng lần chạy ra JSON, rồi điền vào experiments.xlsx từ mẫu
templates/experiment_table_template.xlsx (đừng gõ tay hàng chục dòng, rất dễ sai).

Tên cột của sheet "Experiments" (giữ nguyên, đúng thứ tự mẫu):
    exp_id, group, description, loss, optimizer, lr, weight_decay, batch, epochs, hidden, dropout,
    clip_norm, precision, init, seed, step0_loss, best_val_loss, best_epoch, final_train_loss,
    final_val_loss, val_acc, val_macro_f1, time_per_epoch_s, peak_mem_MB, diverged,
    eval_acc, eval_macro_f1, figure_file, notes
(các cột công thức ở cuối bảng mẫu tự tính, đừng ghi đè)
"""
from __future__ import annotations

import json
import math
from copy import copy
from pathlib import Path


def save_result(result: dict, results_dir: str = "../results") -> str:
    """Ghi result["cfg"], result["history"], result["summary"] (KHÔNG ghi best_state) ra
    <results_dir>/<exp_id>.json. Trả về đường dẫn file. Tạo thư mục nếu chưa có."""
    # Chỉ lấy cấu hình, lịch sử và tóm tắt; không lưu best_state chứa trọng số model.
    # JSON là dạng file văn bản dùng để lưu các giá trị và danh sách có tên rõ ràng.
    saved_result = {key: result[key] for key in ("cfg", "history", "summary")}
    # Dùng mã thí nghiệm làm tên file, ví dụ exp_id = "base-s1" -> base-s1.json.
    exp_id = result["cfg"]["exp_id"]
    if not isinstance(exp_id, str) or not exp_id.strip() or any(
        char in exp_id for char in '<>:"/\\|?*'
    ):
        raise ValueError("exp_id must be a non-empty string without filename special characters")

    # Path giúp ghép đường dẫn; parents=True tạo cả thư mục cha nếu cần.
    output_dir = Path(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{exp_id}.json"
    # Chuyển sang văn bản trước khi mở file để lỗi chuyển đổi không xóa file cũ.
    # ensure_ascii=False giữ nguyên tiếng Việt; indent=2 thụt dòng cho dễ đọc.
    json_text = json.dumps(saved_result, ensure_ascii=False, indent=2)
    # UTF-8 lưu được tiếng Việt; ghi đè kết quả nếu chạy lại cùng exp_id.
    output_path.write_text(json_text + "\n", encoding="utf-8")
    return str(output_path)  # Trả đường dẫn dạng chuỗi để notebook dùng tiếp.


def load_results(results_dir: str = "../results") -> list[dict]:
    """Đọc mọi file *.json trong results_dir, trả về danh sách dict (sắp theo exp_id)."""
    results = []
    seen = set()
    for path in sorted(Path(results_dir).glob("*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        if set(result) != {"cfg", "history", "summary"}:
            raise ValueError(f"Not an experiment result: {path}")
        exp_id = result["cfg"]["exp_id"]
        if path.stem != exp_id or exp_id in seen:
            raise ValueError(f"Duplicate or mismatched exp_id: {path}")
        seen.add(exp_id)
        results.append(result)
    return sorted(results, key=lambda result: result["cfg"]["exp_id"])


def to_row(result: dict, eval_scores: dict | None = None, notes: str = "") -> dict:
    """Biến một kết quả thành một dòng của bảng: gộp cfg + summary (+ eval_acc, eval_macro_f1 nếu có)
    + figure_file = f"figures/{exp_id}.png". Khoá phải trùng tên cột ở đầu file.
    Chỉ truyền eval_scores cho baseline và cấu hình cuối cùng."""
    cfg, summary = result["cfg"], result["summary"]
    row = {**cfg, **summary}
    row["group"] = "other" if cfg["group"] == "baseline_lr" else cfg["group"]
    row["diverged"] = "Y" if summary["diverged"] else "N"
    row["hidden"] = "-".join(map(str, cfg["hidden"]))
    row["loss"] = {"ce": "CE", "mse": "MSE"}[cfg["loss"]]
    row["optimizer"] = {"sgd": "SGD", "sgd_momentum": "SGD+momentum",
                        "adam": "Adam", "adamw": "AdamW"}[cfg["optimizer"]]
    row["clip_norm"] = "none" if cfg["clip_norm"] is None else cfg["clip_norm"]
    row["figure_file"] = f"figures/{cfg['exp_id']}.png"
    details = [cfg.get("notes", ""), notes]
    if cfg["optimizer"] == "sgd_momentum":
        details.append(f"momentum={cfg['momentum']}; dampening=0; nesterov=False")
    elif cfg["optimizer"] == "sgd":
        details.append("momentum=0")
    else:
        details.append(f"betas={tuple(cfg.get('betas', (0.9, 0.999)))}; eps={cfg.get('eps', 1e-8)}")
    if summary.get("peak_mem_MB") == 0:
        row["peak_mem_MB"] = None
        details.append("CPU: bộ nhớ GPU không áp dụng; chưa đo RAM")
    if summary.get("diverged"):
        details.append(summary.get("divergence_reason") or "Phân kỳ; xem loss/gradient trong JSON")
    if summary.get("device"):
        details.append(f"device={summary['device']}; torch={summary.get('torch_version', 'unknown')}; "
                       f"cpu_threads={summary.get('cpu_threads')}")
    row["notes"] = "; ".join(detail for detail in details if detail)
    # Giá trị chưa đo hoặc không hữu hạn để trống, kèm lý do, không điền số giả.
    for key, value in list(row.items()):
        if isinstance(value, float) and not math.isfinite(value):
            row[key] = None
            row["notes"] += f"; {key} không hữu hạn"
    row["eval_acc"] = row["eval_macro_f1"] = None
    if eval_scores is not None:
        row["eval_acc"] = eval_scores["accuracy"]
        row["eval_macro_f1"] = eval_scores["macro_f1"]
    return row


def write_xlsx(rows: list[dict], template_path: str, out_path: str,
               summary_notes: dict | None = None) -> None:
    """Điền các dòng vào sheet "Experiments" của mẫu, từ dòng 2 trở xuống, rồi lưu thành out_path.

    Các bước (openpyxl):
      1. wb = openpyxl.load_workbook(template_path)   # KHÔNG dùng data_only=True (sẽ mất công thức)
      2. ws = wb["Experiments"]; đọc tiêu đề dòng 1 để biết cột nào ứng với khoá nào
      3. với mỗi row: ghi giá trị vào đúng cột; BỎ QUA các cột công thức (step0_gap_vs_lnC, gap_val_minus_train,
         delta_val_f1_vs_base, beyond_noise)
      4. wb.save(out_path)
    Sau khi lưu, mở file bằng Excel/LibreOffice để các công thức tính lại.
    """
    import openpyxl
    from openpyxl.formula.translate import Translator
    from openpyxl.workbook.properties import CalcProperties

    ids = [row["exp_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Each run must have a unique exp_id")
    wb = openpyxl.load_workbook(template_path)
    ws = wb["Experiments"]
    headers = {cell.value: cell.column for cell in ws[1] if cell.value}
    formula_keys = {"step0_gap_vs_lnC", "gap_val_minus_train",
                    "delta_val_f1_vs_base", "beyond_noise"}
    original_last = ws.max_row
    for cells in ws.iter_rows(min_row=2):
        for cell in cells:
            if ws.cell(1, cell.column).value not in formula_keys:
                cell.value = None
    for index, row in enumerate(rows, 2):
        for key, column in headers.items():
            target = ws.cell(index, column)
            if index > original_last:
                source = ws.cell(2, column)
                target._style = copy(source._style)
                if key in formula_keys:
                    target.value = Translator(source.value, origin=source.coordinate).translate_formula(target.coordinate)
            if key not in formula_keys:
                target.value = row.get(key)
    # Mẫu có 60 dòng; mở rộng các tham chiếu khi bảng cần thêm dòng.
    last_row = max(original_last, len(rows) + 1)
    if last_row > original_last:
        for sheet in wb:
            for cells in sheet:
                for cell in cells:
                    if cell.data_type == "f":
                        cell.value = cell.value.replace(f"${original_last}", f"${last_row}")
    seeds = wb["Seeds"]
    baseline_ids = [row["exp_id"] for row in rows if row["group"] == "baseline"]
    if len(baseline_ids) > 5:
        raise ValueError("Template Seeds supports five baseline seeds")
    for index in range(2, 7):
        seeds.cell(index, 1).value = baseline_ids[index - 2] if index - 2 < len(baseline_ids) else None
    if summary_notes:
        summary = wb["Summary"]
        for index in range(2, 12):
            group = summary.cell(index, 1).value
            if group in summary_notes:
                summary.cell(index, 8).value = summary_notes[group]
    wb.calculation = CalcProperties(calcMode="auto", fullCalcOnLoad=True)
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    wb.save(output)
