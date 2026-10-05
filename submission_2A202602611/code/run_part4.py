"""Hoàn thiện/chạy riêng Part 4; giữ log và quyết định cấu hình ở Part 3.

Từ gốc repo: .venv/Scripts/python.exe submission_2A202602611/code/run_part4.py --execute
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import nbformat

CODE_DIR = Path(__file__).resolve().parent
NOTEBOOK = CODE_DIR / "lab.ipynb"


def make_cells():
    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    return [md("""## Part 4 — Đánh giá cuối trên eval, bảng và báo cáo

Giữ cấu hình/seed/epoch đã chọn ở Part 3. Dùng `best_state` có val loss thấp nhất; nếu chỉ còn JSON thì khôi phục đúng cấu hình và ghi riêng lần chạy lại khi số val khác log gốc. Chỉ baseline và cấu hình cuối được chấm eval. Các output cũ Part 2 có số khác JSON hiện có; bảng/báo cáo dùng JSON và ghi rõ lần tạo CSV.

CSV baseline lưu riêng; `predictions_eval.csv` dành cho cấu hình cuối. Không chọn lại tham số từ eval."""), code("""from part4_exports import (
    recover_result, evaluate_selected, error_analysis, export_table, write_report, verify_submission,
)
from IPython.display import display, Markdown, Image
from pathlib import Path
import json

out = Path(OUT_DIR)
selection = json.loads((out / "part3_selection.json").read_text(encoding="utf-8"))
assert selection["eval_used"] is False
read_log = lambda exp_id: json.loads((out / "results" / f"{exp_id}.json").read_text(encoding="utf-8"))
if "baseline_result" not in globals():
    baseline_result = read_log("base-s1")
if "result_final" not in globals():
    result_final = read_log(selection["cfg_final"]["exp_id"])
assert result_final["cfg"]["exp_id"] == selection["cfg_final"]["exp_id"]
assert result_final["summary"]["best_epoch"] == selection["best_epoch"]

baseline_eval_result = recover_result(baseline_result, data, out / "checkpoints")
final_eval_result = recover_result(result_final, data, out / "checkpoints")
manifest = {
    "baseline_exp_id": baseline_eval_result["cfg"]["exp_id"],
    "final_exp_id": final_eval_result["cfg"]["exp_id"],
    "seed": final_eval_result["cfg"]["seed"],
    "best_epoch": final_eval_result["summary"]["best_epoch"],
    "selected_in_part3": selection["cfg_final"]["exp_id"], "selection_changed": False,
    "baseline_replayed": baseline_eval_result["cfg"]["exp_id"] != baseline_result["cfg"]["exp_id"],
    "final_replayed": final_eval_result["cfg"]["exp_id"] != result_final["cfg"]["exp_id"],
}
(out / "part4_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
print(json.dumps(manifest, ensure_ascii=False, indent=2))
"""), code("""# Chấm bằng scripts/evaluate.py từ gốc repo; không tự sửa JSON metric.
baseline_eval_scores, final_eval_scores = evaluate_selected(
    baseline_eval_result, final_eval_result, data, REPO_ROOT, OUT_DIR,
)
assert baseline_eval_scores["n_eval"] == final_eval_scores["n_eval"] == 116203
print("Seed nộp:", manifest["seed"], "; epoch:", manifest["best_epoch"])
"""), code("""import pandas as pd
display(pd.DataFrame(final_eval_scores["per_class"]))
error_note = error_analysis(final_eval_scores, out / "figures" / "compare_eval_confusion.png")
display(Image(filename=str(out / "figures" / "compare_eval_confusion.png"), width=600))
display(Markdown(error_note))
"""), md("""**Phân tích lỗi:** lớp khó nhất, cặp nhầm và số mẫu được lấy trực tiếp từ JSON ở ô trên. Nguyên nhân chỉ là giả thuyết; chưa có thí nghiệm xác nhận. Đề xuất kiểm tra tiếp trên train/val, giữ nguyên cấu hình nộp."""), code("""# Kiểm tra lại khởi tạo He; không huấn luyện hoặc dùng nhãn eval.
from train import set_seed
from model import MLP
set_seed(1)
he_model = MLP(hidden=(256, 128), dropout=0, init="he").to(data["X_val"].device)
he_stds = []
with torch.no_grad():
    h = data["X_val"][:512]
    for layer in he_model.net:
        h = layer(h)
        if isinstance(layer, torch.nn.Linear):
            he_stds.append(float(h.std(correction=0)))
print("He: độ phân tán sau mỗi Linear:", he_stds)

all_results = export_table(
    OUT_DIR, Path(REPO_ROOT) / "templates" / "experiment_table_template.xlsx",
    baseline_eval_result, final_eval_result, baseline_eval_scores, final_eval_scores,
)
report_path = write_report(
    OUT_DIR, baseline_eval_result, final_eval_result,
    baseline_eval_scores, final_eval_scores, error_note,
)
verify_submission(OUT_DIR, manifest["baseline_exp_id"], manifest["final_exp_id"])
print("Đã ghi:", out / "experiments.xlsx", report_path)
print("Mở experiments.xlsx bằng Excel, tính lại công thức và lưu; chạy ô cuối để kiểm tra.")
"""), code("""# Sau khi Excel đã tính lại và lưu, kiểm tra cả giá trị công thức.
import openpyxl
cached = openpyxl.load_workbook(out / "experiments.xlsx", data_only=True)
calculated = cached["Seeds"]["C10"].value is not None
verify_submission(OUT_DIR, manifest["baseline_exp_id"], manifest["final_exp_id"], require_calculated=calculated)
print("Công thức đã tính lại:", calculated)
""")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--verify", action="store_true", help="Chạy ô cuối sau khi Excel đã tính lại")
    args = parser.parse_args()
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    start = next(i for i, c in enumerate(notebook.cells) if c.source.startswith("## Part 4"))
    if args.verify:
        cells = [notebook.cells[-1]]
        start = len(notebook.cells) - 1
    else:
        cells = make_cells()
        notebook.cells[start:] = cells
        nbformat.write(notebook, NOTEBOOK)
    if not args.execute and not args.verify:
        return
    from nbclient import NotebookClient
    from jupyter_client import KernelManager

    for name in ("IPYTHONDIR", "JUPYTER_RUNTIME_DIR", "MPLCONFIGDIR"):
        local = CODE_DIR.parents[1] / ".mplconfig" / name.lower()
        local.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(local)
    os.environ["MPLBACKEND"] = "Agg"
    context = nbformat.v4.new_code_cell("""from pathlib import Path
import json
from part4_exports import verify_submission
OUT_DIR = ".."
out = Path(OUT_DIR)
manifest = json.loads((out / "part4_manifest.json").read_text(encoding="utf-8"))
""" if args.verify else """import torch
from data import prepare_data
REPO_ROOT = "../.."
OUT_DIR = ".."
torch.set_num_threads(1)
data = prepare_data("cpu", val_fraction=0.2, seed=42, processed_dir="../../data/processed")
""")
    fragment = nbformat.v4.new_notebook(cells=[context] + cells)
    km = KernelManager(kernel_name="python3")
    km.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    client = NotebookClient(fragment, km=km, timeout=1800, resources={"metadata": {"path": str(CODE_DIR)}})

    def on_done(cell, cell_index, **kwargs):
        if cell_index:
            notebook.cells[start + cell_index - 1] = cell
            nbformat.write(notebook, NOTEBOOK)
        for result in cell.get("outputs", []):
            if result.output_type == "stream":
                print(result.text, end="", flush=True)
            elif result.output_type in ("display_data", "execute_result") and "text/markdown" in result.data:
                print(result.data["text/markdown"], flush=True)

    client.on_cell_executed = on_done
    try:
        client.execute()
    finally:
        notebook.cells[start:] = fragment.cells[1:]
        nbformat.write(notebook, NOTEBOOK)
    print("Part 4 đã chạy; giữ output trong notebook.")


if __name__ == "__main__":
    main()
