"""Chuẩn bị và chạy riêng Part 3, giữ output đã có của Part 0–2 và khung Part 4.

Chạy từ bất kỳ thư mục nào: python run_part3.py --execute
Mặc định chỉ cập nhật các ô Part 3, chưa huấn luyện.
"""
from __future__ import annotations

import argparse
import os
import sys
import uuid
from pathlib import Path

import nbformat

from part3_experiments import TRIALS


CODE_DIR = Path(__file__).resolve().parent
NOTEBOOK_PATH = CODE_DIR / "lab.ipynb"


def make_cells():
    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    cells = [md("""## Part 3 — Hyper-parameter và bộ tối ưu

Tôi khảo sát **SGD + momentum so với Adam**, và **batch size**. SGD + momentum đã được thử `lr=0.01, 0.03, 0.1` với đủ 20 epoch ở Part 2; tôi dùng lại kết quả đó, không tạo thêm mã cho cùng một lần chạy. Adam thử `lr=0.001, 0.003`.

Giữ phép tách val `20%`, split seed `42`, training seed `1`, 20 epoch, M-base **47 879 tham số**, CE, He, FP32, weight decay `0`, dropout `0`, không clip. Thí nghiệm Adam đổi cả optimizer và lr so với baseline; điều này được ghi trong `notes`. Hai lần thử batch giữ SGD + momentum `0.9` và `lr=0.1`.

**Quy tắc chọn đặt trước khi chạy:** mỗi lần chọn epoch có **val loss thấp nhất**, rồi so **macro-F1 tại epoch đó**. Chọn lr tốt nhất cho từng optimizer, sau đó chọn cấu hình cuối theo macro-F1; nếu bằng nhau, ưu tiên val loss thấp hơn rồi lr nhỏ hơn. Giữ seed 1 để so, không chọn seed có điểm cao nhất. Không dùng điểm eval.

Chênh lệch được so với `2σ` của ba seed baseline (`σ`: mức dao động giữa các seed). Ngưỡng này chỉ tham khảo; các cấu hình mới chưa được kiểm tra nhiều seed. Chỉ khảo sát hai trong bảy chủ đề theo lựa chọn của bài này."""), code("""from part3_experiments import (
    TRIALS, run_trial, compare_text, comparison_table, select_by_val, export_part3_table,
)
from IPython.display import display, Markdown, Image
import math

baseline_cfg = dict(baseline_result["cfg"])
assert baseline_cfg["seed"] == 1 and baseline_cfg["epochs"] == 20
assert baseline_cfg["batch"] == 512 and baseline_cfg["optimizer"] == "sgd_momentum"
assert baseline_cfg["lr"] == 0.1
assert count_params(MLP(hidden=baseline_cfg["hidden"], init=baseline_cfg["init"])) == 47_879
if device == "cpu":
    torch.set_num_threads(1)  # Mạng nhỏ: cố định một luồng để đo thời gian nhất quán.
# Chỉ truyền train/val vào thí nghiệm; các tensor eval chưa được sử dụng.
part3_data = {key: data[key] for key in ("X_tr", "y_tr", "X_val", "y_val")}
n_train = len(part3_data["y_tr"])
part3_results = []
print("Train:", n_train, "Val:", len(part3_data["y_val"]), "Device:", device)
print(f"Baseline noise: F1 2σ={noise_f1:.6f}; accuracy 2σ={noise_acc:.6f}")
print("CPU threads:", torch.get_num_threads(), "; bộ nhớ GPU không áp dụng trên CPU")
print("Baseline time là log cũ, chưa ghi số luồng CPU; chỉ dùng để tham khảo.")
"""), md("""### 3a. Bộ tối ưu — dùng lại các learning rate của SGD + momentum

Part 2 đã đo ba lr với cùng seed, split và số epoch. Đây là **kết quả đã chạy**, không phải dự đoán mới trước lần chạy cũ. Khi so optimizer, chọn lr tốt nhất của SGD + momentum từ các log này và lr tốt nhất của Adam từ hai lần chạy bên dưới. Không kết luận Adam tốt hơn chỉ vì dùng một lr chung cho hai bộ."""), code("""sgdm_trials = [r for r in lr_results if r["cfg"]["optimizer"] == "sgd_momentum"]
assert len({r["cfg"]["lr"] for r in sgdm_trials}) >= 2
for r in sgdm_trials:
    for key in ("epochs", "seed", "batch", "loss", "hidden", "dropout", "init",
                "weight_decay", "clip_norm", "precision", "momentum"):
        assert r["cfg"][key] == baseline_cfg[key], key
best_sgdm = select_by_val(sgdm_trials)
display(comparison_table(sgdm_trials, n_train))
print("SGD+momentum tốt nhất theo val:", best_sgdm["cfg"]["exp_id"])
""")]
    for index, trial in enumerate(TRIALS):
        if index == 2:
            cells.append(md("""### 3b. Hyper-parameter — batch size

Chỉ đổi số mẫu trong mỗi lô học: **128 / 512 / 2048**. Giữ lô cuối nhỏ hơn batch, không bỏ mẫu. Với 371 847 mẫu train, số bước mỗi epoch lần lượt là **2 906 / 727 / 182**; qua 20 epoch là **58 120 / 14 540 / 3 640**. Khác số bước là một phần của thí nghiệm này; chưa thử giữ cùng số bước hoặc tăng lr theo batch."""))
        cells.append(md(f"### Thí nghiệm `{trial['exp_id']}`\n\n"
                        f"**Yếu tố thay đổi:** {trial['description']}.\n\n"
                        f"**Dự đoán trước:** {trial['prediction']}\n\n"
                        f"**Cấu hình bổ sung / giới hạn:** {trial['notes']}."))
        cells.append(code(f"""trial = TRIALS[{index}]
result = run_trial(baseline_cfg, trial, part3_data, OUT_DIR)
part3_results.append(result)
# Cùng split và khởi tạo seed 1 phải có cùng loss trước bước cập nhật đầu tiên.
assert abs(result["summary"]["step0_loss"] - baseline_result["summary"]["step0_loss"]) < 1e-6
display(Image(filename=str(figures_dir / (trial["exp_id"] + ".png"))))
"""))
        kind = "adam" if index < 2 else ("small_batch" if index == 2 else "large_batch")
        cells.append(code(f"""display(Markdown(compare_text(
    part3_results[-1], baseline_result, noise_f1, noise_acc, n_train, {kind!r},
)))
"""))
    cells += [md("""### 3c. So trực tiếp các đường cong trong từng nhóm

Ảnh learning rate chứa ba cấu hình SGD + momentum của Part 2 và hai Adam mới. Ảnh optimizer chỉ so **lr tốt nhất của mỗi bộ**. Ảnh batch so hai batch mới với baseline. Accuracy và macro-F1 trong bảng đều lấy tại epoch có val loss thấp nhất, không lấy epoch F1 cao nhất."""), code("""adam_trials = [r for r in part3_results if r["cfg"]["optimizer"] == "adam"]
assert len({r["cfg"]["lr"] for r in adam_trials}) >= 2
best_adam = select_by_val(adam_trials)
optimizer_winners = [best_sgdm, best_adam]
batch_trials = [baseline_result] + [r for r in part3_results if r["cfg"]["group"] == "hparam"]
groups = {
    "optimizer_lr": sgdm_trials + adam_trials,
    "optimizer": optimizer_winners,
    "hparam": batch_trials,
}
for group_name, group_results in groups.items():
    for metric, suffix in (("val_loss", ""), ("val_macro_f1", "_f1")):
        path = figures_dir / f"compare_{group_name}{suffix}.png"
        plot_compare(group_results, metric, str(path), title=f"{group_name}: {metric}")
    display(Image(filename=str(figures_dir / f"compare_{group_name}.png")))
display(comparison_table(sgdm_trials + part3_results, n_train))
comparison_table(sgdm_trials + part3_results, n_train).to_csv(
    Path(OUT_DIR) / "part3_comparison.csv", index=False,
)
f1_difference = best_adam["summary"]["val_macro_f1"] - best_sgdm["summary"]["val_macro_f1"]
adam_low, adam_high = sorted(adam_trials, key=lambda r: r["cfg"]["lr"])
high_faster = adam_high["history"]["val_loss"][2] < adam_low["history"]["val_loss"][2]
low_jitter = float(np.std(np.diff(adam_low["history"]["val_loss"])))
high_jitter = float(np.std(np.diff(adam_high["history"]["val_loss"])))
display(Markdown(
    f"**Đối chiếu hai lr của Adam:** dự đoán lr=0.003 giảm loss nhanh hơn lr=0.001 "
    f"{'khớp' if high_faster else 'khác kết quả'} tại epoch 3: "
    f"{adam_high['history']['val_loss'][2]:.4f} so với {adam_low['history']['val_loss'][2]:.4f}. "
    f"Độ dao động của thay đổi val loss giữa các epoch (độ lệch chuẩn) là "
    f"{high_jitter:.4f} so với {low_jitter:.4f}; số này cũng bị ảnh hưởng bởi tốc độ giảm loss, "
    f"chưa tách được dao động quanh xu hướng. Lr Adam được chọn: **{best_adam['cfg']['lr']}**."
))
optimizer_note = (
    f"Ở lr tốt nhất đã thử, Adam ({best_adam['cfg']['exp_id']}) so với SGD+momentum "
    f"({best_sgdm['cfg']['exp_id']}) có chênh lệch F1 {f1_difference:+.6f}; "
    f"{'vượt' if abs(f1_difference) > noise_f1 else 'chưa vượt'} nhiễu baseline 2σ={noise_f1:.6f}. "
    "Mỗi bộ đã thử ít nhất hai lr; chưa kiểm tra nhiều seed của Adam, nên chưa kết luận thắng ổn định. "
    "Đây là so sánh trong phạm vi các lr và ngân sách 20 epoch đã thử."
)
display(Markdown(optimizer_note))
"""), md("""### 3d. Chốt cấu hình cuối bằng validation — trước Part 4

Áp dụng quy tắc đã đặt ở đầu Part 3. Cấu hình được chọn là một cấu hình đã chạy; không tự ghép optimizer tốt nhất với batch tốt nhất khi chưa đo tổ hợp đó. Seed 2 và 3 chỉ dùng để ước lượng nhiễu baseline, không đưa vào danh sách để chọn seed tốt nhất."""), code("""candidates = [baseline_result] + sgdm_trials + part3_results
result_final = select_by_val(candidates)
cfg_final = dict(result_final["cfg"])
final_epoch = result_final["summary"]["best_epoch"]
final_difference = result_final["summary"]["val_macro_f1"] - baseline_result["summary"]["val_macro_f1"]
final_note = (
    f"Chọn **{cfg_final['exp_id']}**, optimizer={cfg_final['optimizer']}, lr={cfg_final['lr']}, "
    f"batch={cfg_final['batch']}, seed={cfg_final['seed']}, epoch có val loss thấp nhất **{final_epoch}**. "
    f"Val loss={result_final['summary']['best_val_loss']:.6f}, "
    f"accuracy={result_final['summary']['val_acc']:.2%}, "
    f"macro-F1={result_final['summary']['val_macro_f1']:.6f}. "
    f"Chênh lệch F1 so với baseline seed 1 là {final_difference:+.6f}; "
    f"{'vượt' if abs(final_difference) > noise_f1 else 'chưa vượt'} ngưỡng 2σ={noise_f1:.6f}. "
    "Đây là lựa chọn theo val trong các cấu hình đã thử; mức cải thiện chưa được xác nhận bằng nhiều seed. "
    "Part 4 phải dùng trọng số của epoch này, không điều chỉnh cấu hình theo điểm eval."
)
selection = {
    "cfg_final": cfg_final, "best_epoch": final_epoch,
    "val_metrics": result_final["summary"],
    "selection_rule": "Highest val_macro_f1 at the minimum-val-loss epoch; tie: lower loss, lower lr",
    "candidate_exp_ids": list(dict.fromkeys(r["cfg"]["exp_id"] for r in candidates)),
    "baseline_noise_f1_2sigma": noise_f1, "reason": final_note, "eval_used": False,
}
(Path(OUT_DIR) / "part3_selection.json").write_text(
    json.dumps(selection, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8",
)
display(Markdown(final_note))
print(json.dumps(cfg_final, ensure_ascii=False, indent=2))
if "best_state" not in result_final:
    print("Log cũ không lưu trọng số. Khi chạy notebook từ đầu, result_final có best_state ở RAM; "
          "nếu chỉ đọc JSON, cần chạy lại đúng cấu hình đã chốt trước Part 4.")
"""), md("""### 3e. Ghi bảng thí nghiệm

Lưu tất cả các lần chạy của Part 2 và Part 3 vào `experiments.xlsx`, gồm cả các lần thử lr. Giữ bốn sheet và cột của mẫu. `eval_acc` và `eval_macro_f1` để trống vì chưa đo eval. Cấu hình cuối được đánh dấu trong `notes`, không nhân đôi một lần chạy thành dòng giả mới."""), code("""summary_notes = {
    "baseline": f"Ba seed; F1 2σ={noise_f1:.6f}; accuracy 2σ={noise_acc:.6f}. "
                "Ba lr của Part 2 nằm ở group baseline_lr và được dùng lại để so optimizer.",
    "optimizer": optimizer_note,
    "hparam": "Đã thử batch 128/2048, baseline 512; cùng lr=0.1, seed 1 và 20 epoch. "
              "Số bước lần lượt 58120/3640/14540. Chưa kiểm tra cùng số bước hoặc tăng lr theo batch; "
              "thời gian baseline thiếu log số luồng CPU; cấu hình mới mới có một seed.",
    "final": final_note + " Không chạy lại một cấu hình trùng lặp nên đánh dấu lựa chọn trong notes "
             "của dòng gốc; chưa có group final mới.",
    "other": "baseline_lr: ba lần tìm lr của Part 2 đã ghi đủ dòng/ảnh; dùng lại trong so optimizer.",
}
table_path = export_part3_table(
    OUT_DIR, Path(REPO_ROOT) / "templates" / "experiment_table_template.xlsx",
    result_final, summary_notes,
)
print("Đã ghi:", table_path)
print("Các công thức được giữ; Excel tính lại khi mở file.")
print("Hoàn tất Part 3. Cấu hình đã chốt trước khi dùng tập eval ở Part 4.")
""")]
    return cells


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    nb.nbformat_minor = max(nb.nbformat_minor, 5)
    for cell in nb.cells:
        if "id" not in cell:
            cell.id = uuid.uuid4().hex[:8]
    start = next(i for i, c in enumerate(nb.cells) if c.cell_type == "markdown" and c.source.startswith("## Part 3"))
    end = next(i for i, c in enumerate(nb.cells) if c.cell_type == "markdown" and c.source.startswith("## Part 4"))
    cells = make_cells()
    nb.cells[start:end] = cells
    # Giữ môi trường một luồng cả khi Restart & Run All trên CPU.
    setup = nb.cells[1]
    marker = 'print("PyTorch:", torch.__version__, "Device:", device)'
    if "torch.set_num_threads(1)" not in setup.source:
        setup.source = setup.source.replace(marker,
            'if device == "cpu":\n    torch.set_num_threads(1)\n' + marker)
    # Part 4 sử dụng lựa chọn đã chốt ở Part 3.
    for cell in nb.cells[start + len(cells):]:
        if cell.cell_type == "code" and cell.source.startswith("# TODO: chọn cấu hình cuối"):
            cell.source = cell.source.replace(
                "# TODO: chọn cấu hình cuối cùng theo VAL (có thể là baseline nếu bạn không kết hợp thêm gì)",
                "# cfg_final, result_final và final_epoch đã chốt bằng val ở Part 3. "
                "Dùng result_final['best_state']; không chọn lại bằng eval.")
    nbformat.write(nb, NOTEBOOK_PATH)
    print(f"Prepared {len(cells)} Part 3 cells in {NOTEBOOK_PATH}", flush=True)
    if not args.execute:
        return
    from nbclient import NotebookClient
    from jupyter_client import KernelManager

    # Chuẩn bị đúng biến từ Part 0–2 mà không huấn luyện lại hoặc nạp eval.
    context = nbformat.v4.new_code_cell("""import json, math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from data import prepare_data
from model import MLP, count_params
from plots import plot_compare
from train import DEFAULT_CFG
from part3_experiments import prepare_part3_data
REPO_ROOT = "../.."
OUT_DIR = ".."
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(1)
data = prepare_part3_data(str(Path(REPO_ROOT) / "data" / "processed"), device)
results_dir = Path(OUT_DIR) / "results"
figures_dir = Path(OUT_DIR) / "figures"
read_result = lambda exp_id: json.loads((results_dir / f"{exp_id}.json").read_text(encoding="utf-8"))
baseline_results = [read_result(f"base-s{s}") for s in (1, 2, 3)]
baseline_result = baseline_results[0]
lr_results = [read_result(f"base-lr{lr:g}-s1") for lr in (0.01, 0.03, 0.1)]
baseline_summary = json.loads((Path(OUT_DIR) / "part2_summary.json").read_text(encoding="utf-8"))
noise_f1 = baseline_summary["val_macro_f1"]["noise_2sigma"]
noise_acc = baseline_summary["val_acc"]["noise_2sigma"]
cfg = dict(baseline_result["cfg"])
""")
    for var in ("IPYTHONDIR", "JUPYTER_RUNTIME_DIR", "MPLCONFIGDIR"):
        local_path = CODE_DIR.parents[1] / ".mplconfig" / var.lower()
        local_path.mkdir(parents=True, exist_ok=True)
        os.environ[var] = str(local_path)
    os.environ["MPLBACKEND"] = "Agg"
    fragment = nbformat.v4.new_notebook(cells=[context] + cells)
    km = KernelManager(kernel_name="python3")
    km.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    client = NotebookClient(fragment, km=km, timeout=1800,
                            resources={"metadata": {"path": str(CODE_DIR)}})

    def on_start(cell, cell_index, **kwargs):
        if cell.cell_type == "code":
            print(f"Executing Part 3 cell {cell_index}: {cell.source.splitlines()[0]}", flush=True)

    def on_done(cell, cell_index, **kwargs):
        if cell_index:
            nb.cells[start + cell_index - 1] = cell
            nbformat.write(nb, NOTEBOOK_PATH)
        for output in cell.get("outputs", []):
            if output.output_type == "stream":
                print(output.text, end="", flush=True)
            elif output.output_type in ("display_data", "execute_result"):
                if "text/markdown" in output.data:
                    print(output.data["text/markdown"], flush=True)

    client.on_cell_start = on_start
    client.on_cell_executed = on_done
    try:
        client.execute()
    finally:
        nb.cells[start:start + len(cells)] = fragment.cells[1:]
        nbformat.write(nb, NOTEBOOK_PATH)
    print("Part 3 executed; outputs saved. Part 4 was not executed.", flush=True)


if __name__ == "__main__":
    main()
