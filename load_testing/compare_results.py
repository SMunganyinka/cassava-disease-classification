"""Summarize Locust replica runs and plot their performance comparison."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RUN_NAME = re.compile(r"replicas-(\d+)-users-(\d+)_stats\.csv$")


def _column(row: pd.Series, *names: str) -> float:
    normalized = {str(key).strip().lower(): value for key, value in row.items()}
    for name in names:
        value = normalized.get(name.lower())
        if value is not None and pd.notna(value):
            return float(value)
    raise ValueError(f"Locust stats row is missing columns: {', '.join(names)}")


def collect_runs(results_dir: Path) -> pd.DataFrame:
    """Read aggregate rows from Locust CSV files named by replica/user count."""
    records: list[dict[str, float | int]] = []
    for stats_path in sorted(Path(results_dir).glob("replicas-*-users-*_stats.csv")):
        match = RUN_NAME.fullmatch(stats_path.name)
        if match is None:
            continue
        replicas, users = map(int, match.groups())
        stats = pd.read_csv(stats_path)
        name_column = next(
            (column for column in stats.columns if column.strip().lower() == "name"),
            None,
        )
        if name_column is None:
            raise ValueError(f"Locust stats file has no Name column: {stats_path}")
        aggregate = stats[
            stats[name_column].astype(str).str.strip().str.lower() == "aggregated"
        ]
        if aggregate.empty:
            raise ValueError(f"Locust stats file has no Aggregated row: {stats_path}")
        row = aggregate.iloc[0]
        request_count = int(_column(row, "Request Count"))
        failure_count = int(_column(row, "Failure Count"))
        records.append(
            {
                "replicas": replicas,
                "users": users,
                "requests": request_count,
                "failures": failure_count,
                "requests_per_second": _column(row, "Requests/s"),
                "average_ms": _column(row, "Average Response Time"),
                "median_ms": _column(row, "Median Response Time", "50%"),
                "p95_ms": _column(row, "95%"),
                "p99_ms": _column(row, "99%"),
                "failure_rate_percent": (
                    100.0 * failure_count / request_count if request_count else 0.0
                ),
            }
        )
    if not records:
        raise FileNotFoundError(
            f"No matching Locust stats files found in {results_dir}"
        )
    return pd.DataFrame(records).sort_values(["users", "replicas"]).reset_index(drop=True)


def save_comparison(results_dir: Path) -> tuple[Path, Path]:
    """Write comparison CSV and chart from completed Locust runs."""
    results_dir = Path(results_dir)
    comparison = collect_runs(results_dir)
    csv_path = results_dir / "comparison.csv"
    plot_path = results_dir / "comparison.png"
    comparison.to_csv(csv_path, index=False)

    label = comparison.apply(
        lambda row: f"{int(row['replicas'])} replicas\n{int(row['users'])} users",
        axis=1,
    )
    measures = [
        ("requests_per_second", "Requests per second"),
        ("average_ms", "Average response (ms)"),
        ("median_ms", "Median response (ms)"),
        ("p95_ms", "95th percentile (ms)"),
        ("p99_ms", "99th percentile (ms)"),
        ("failure_rate_percent", "Failures (%)"),
    ]
    figure, axes = plt.subplots(2, 3, figsize=(15, 8))
    for axis, (column, title) in zip(axes.flat, measures):
        axis.bar(label, comparison[column], color="#4C956C")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(plot_path, dpi=150)
    plt.close(figure)
    return csv_path, plot_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results/locust"))
    args = parser.parse_args()
    csv_path, plot_path = save_comparison(args.results_dir)
    print(f"Saved {csv_path} and {plot_path}")


if __name__ == "__main__":
    main()
