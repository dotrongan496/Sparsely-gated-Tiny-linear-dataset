from collections import defaultdict
from pathlib import Path
import csv
import re

import plotly.graph_objects as go
from plotly.subplots import make_subplots

log_dir = Path("run_logs")
out_dir = Path("report_outputs")
out_dir.mkdir(exist_ok=True)

metric_names = ("train_loss", "train_perplexity", "train_tokens_per_sec")
number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
metric_re = {
    name: re.compile(rf"\b{name}:\s*({number})")
    for name in metric_names
}

rows = []
runs = []

for path in sorted(log_dir.glob("*.log")):
    try:
        model, budget_text = path.stem.rsplit("_", 1)
        budget = int(budget_text)
    except ValueError:
        print(f"Bỏ qua tên file không nhận dạng được: {path.name}")
        continue

    text = path.read_text(encoding="utf-8", errors="replace")
    step_match = re.search(r"train for\s+(\d+)\s+steps", text)
    max_steps = int(step_match.group(1)) if step_match else None

    run_rows = []
    for line in text.splitlines():
        step_match = re.search(r"\bstep:\s*(\d+)", line)
        if not step_match:
            continue

        values = {name: pattern.search(line) for name, pattern in metric_re.items()}
        if not all(values.values()):
            continue

        step = int(step_match.group(1))
        metrics = {name: float(match.group(1)) for name, match in values.items()}

        run_rows.append({
            "model": model,
            "budget_flops": budget,
            "step": step,
            **metrics,
        })

    if not run_rows:
        print(f"Không tìm thấy metric trong: {path.name}")
        continue

    if max_steps is None:
        max_steps = max(row["step"] for row in run_rows)

    for row in run_rows:
        row["compute_fraction"] = row["step"] / max_steps if max_steps else 0
        row["estimated_flops"] = row["compute_fraction"] * budget
        rows.append(row)

    final = max(run_rows, key=lambda row: row["step"])
    runs.append({
        "model": model,
        "budget_flops": budget,
        "planned_steps": max_steps,
        "last_logged_step": final["step"],
        **{name: final[name] for name in metric_names},
    })

if not rows:
    raise SystemExit("Không đọc được metric nào. Kiểm tra thư mục run_logs/.")

def write_csv(filename, data):
    with (out_dir / filename).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(data[0].keys()))
        writer.writeheader()
        writer.writerows(data)

write_csv("training_curves.csv", rows)
write_csv("run_summary.csv", runs)

budgets = sorted({row["budget_flops"] for row in rows})
models = sorted({row["model"] for row in rows})

for metric, title, filename in (
    ("train_loss", "Train loss", "train_loss.html"),
    ("train_perplexity", "Train perplexity", "train_perplexity.html"),
    ("train_tokens_per_sec", "Tokens per second", "tokens_per_sec.html"),
):
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[f"{budget:.0e} FLOP" for budget in budgets],
    )

    for index, budget in enumerate(budgets):
        row_index, col_index = divmod(index, 2)
        for model in models:
            series = sorted(
                (
                    item for item in rows
                    if item["budget_flops"] == budget and item["model"] == model
                ),
                key=lambda item: item["step"],
            )
            if not series:
                continue

            fig.add_trace(
                go.Scatter(
                    x=[item["compute_fraction"] * 100 for item in series],
                    y=[item[metric] for item in series],
                    mode="lines",
                    name=model,
                    legendgroup=model,
                    showlegend=(index == 0),
                ),
                row=row_index + 1,
                col=col_index + 1,
            )

    fig.update_xaxes(title_text="Ngân sách FLOP đã dùng (%)")
    fig.update_yaxes(title_text=title)
    fig.update_layout(
        title=title,
        height=800,
        width=1200,
        template="plotly_white",
    )
    fig.write_html(out_dir / filename)

print(f"Đã tạo {len(runs)} dòng tổng kết và {len(rows)} điểm dữ liệu.")
print(f"Kết quả nằm trong: {out_dir.resolve()}")

