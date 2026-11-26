from pathlib import Path

import pandas as pd

from load_testing.compare_results import collect_runs, save_comparison


def test_locust_results_are_summarized_and_plotted(tmp_path: Path) -> None:
    pd.DataFrame(
        [
            {
                "Type": "",
                "Name": "Aggregated",
                "Request Count": 100,
                "Failure Count": 2,
                "Median Response Time": 120,
                "Average Response Time": 150,
                "Requests/s": 4.5,
                "95%": 300,
                "99%": 450,
            }
        ]
    ).to_csv(tmp_path / "replicas-2-users-5_stats.csv", index=False)

    result = collect_runs(tmp_path).iloc[0]
    csv_path, plot_path = save_comparison(tmp_path)

    assert result["replicas"] == 2
    assert result["users"] == 5
    assert result["failure_rate_percent"] == 2.0
    assert csv_path.is_file()
    assert plot_path.is_file()
