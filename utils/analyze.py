from pathlib import Path

import open3d as o3d
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


def display(path: str):
    "display construction result, saved as .ply file"
    assert Path(path).exists()
    # pcd = o3d.io.read_point_cloud(path) # 这个也可以
    mesh = o3d.io.read_triangle_mesh(path)
    o3d.visualization.draw_geometries([mesh])
