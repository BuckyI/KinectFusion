from pathlib import Path

import numpy as np
import open3d as o3d
import pandas as pd


def profile(path: str = "perf_report.txt"):
    """
    Generate csv from perf report, which may get from:
    python -m cProfile -s cumtime main.py >> perf_report.txt
    """
    with open(path, "r", encoding="utf16") as f:
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


def display_frame(depth, color, K):
    H, W = depth.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    Y, X = np.meshgrid(np.arange(0, W), np.arange(0, H))  # [H, W]
    vertex = np.stack([(X - cx) / fx, (Y - cy) / fy, np.ones_like(X)], -1) * depth[..., None]  # [H, W, 3]

    points = vertex.reshape(-1, 3)  # [H*W, 3]
    colors = color.reshape(-1, 3)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)

    o3d.visualization.draw_geometries([pcd])


def save_frame(depth, color, K, path: str = "frame.npz"):
    np.savez(path, depth=depth, color=color, K=K)


def load_frame(path: str = "frame.npz"):
    data = np.load(path)
    return data["depth"], data["color"], data["K"]


# profile("../perf_report_0419_norender.txt")
