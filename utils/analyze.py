from pathlib import Path

import pandas as pd


def profile(path: str = "perf_report.txt"):
    """
    Generate csv from perf report, which may get from:
    python -m cProfile -s cumtime main.py >> perf_report.txt
    """
    with open(path, "r", encoding="utf8") as f:
        profile: list[str] = f.readlines()
        # remove head info, empty lines, and \n
        profile = [line.rstrip() for line in profile[4:]]
        profile = [line for line in profile if line]

    headers = profile[0].split()
    data = [_.split(maxsplit=len(headers) - 1) for _ in profile[1:]]
    df = pd.DataFrame(data, columns=headers)
    df.to_csv("perf_report.csv", index=False)
