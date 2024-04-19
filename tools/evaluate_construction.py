from pathlib import Path

import numpy as np
import open3d as o3d
import torch


def draw_camera(c2w, cam_width=0.2, cam_height=0.15, f=0.1):
    points = [
        [0, 0, 0],
        [-cam_width, -cam_height, f],
        [cam_width, -cam_height, f],
        [cam_width, cam_height, f],
        [-cam_width, cam_height, f],
    ]
    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
    colors = [[1, 0, 1] for i in range(len(lines))]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    line_set.transform(c2w)

    return line_set


mesh_path = r"E:\3d-datasets\pig_kinect\20240414 - constructed_mesh 0416.ply"
traj_path = r"E:\3d-datasets\pig_kinect\20240414 traj.pth"
assert Path(mesh_path).exists() and Path(traj_path).exists()

mesh = o3d.io.read_triangle_mesh(mesh_path)
traj = torch.load(traj_path)


vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window(width=1280, height=960)
for c in [draw_camera(traj[i][1].cpu().numpy()) for i in np.arange(0, len(traj), 10)]:
    vis.add_geometry(c)

coord_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
vis.add_geometry(coord_axes)

# vol_bounds = np.array([1, 1, 1, 1]).reshape(3, 2)
# bbox = o3d.geometry.AxisAlignedBoundingBox(vol_bounds[:, 0], vol_bounds[:, 1])
# bbox.color = (1, 0, 0)
# vis.add_geometry(bbox)

vis.add_geometry(mesh)
vis.run()
vis.destroy_window()
