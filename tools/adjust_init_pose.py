"""
Kinect Fusion 重建时，第一帧的位姿不使用 I, 而是使用一个比较好的初始化位姿，以减小内存占用和改进可视化效果
"""

from functools import partial

import numpy as np
import open3d as o3d
from loguru import logger
from scipy.spatial.transform import Rotation as R


class Window:
    "创建一个可视化窗口，展示点云，"

    def __init__(self, pcd, bound=[-1, 1, -1, 1, -1, 1]):
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(width=1280, height=960)

        self.original_pcd = pcd
        self.current_pcd = o3d.geometry.PointCloud(pcd)
        self.vis = vis

        self.xyz_rotate = [0, 0, 0]
        self.xyz_translate = [0.0, 0.0, 0.0]
        self._rotate_unit = 1
        self._translate_unit = 0.01

        # show initial point cloud
        coord_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)

        vol_bounds = np.array(bound).reshape(3, 2)
        bbox = o3d.geometry.AxisAlignedBoundingBox(vol_bounds[:, 0], vol_bounds[:, 1])
        bbox.color = (1, 0, 0)

        for g in [self.current_pcd, coord_axes, bbox]:
            vis.add_geometry(g)

        vis.register_key_callback(88, partial(self._rotate, axis="x"))  # X
        vis.register_key_callback(89, partial(self._rotate, axis="y"))  # Y
        vis.register_key_callback(90, partial(self._rotate, axis="z"))  # Z
        vis.register_key_callback(49, partial(self._translate, axis="x"))  # 1
        vis.register_key_callback(50, partial(self._translate, axis="y"))  # 2
        vis.register_key_callback(51, partial(self._translate, axis="z"))  # 3
        vis.register_key_action_callback(32, self._space)  # 按下空格改变调整方向

        vis.run()
        vis.destroy_window()

    def transformed_pcd(self):
        pcd = o3d.geometry.PointCloud(self.original_pcd)
        pcd.rotate(R.from_euler("xyz", self.xyz_rotate, degrees=True).as_matrix())
        pcd.translate(self.xyz_translate)
        return pcd

    def _update_pcd(self, vis):
        pcd = self.transformed_pcd()
        vis.remove_geometry(self.current_pcd, reset_bounding_box=False)
        vis.add_geometry(pcd, reset_bounding_box=False)
        self.current_pcd = pcd

        # indicate current pose
        logger.info("xyz_rotate: {}, xyz_translate: {}".format(self.xyz_rotate, self.xyz_translate))
        rotaion = R.from_euler("xyz", self.xyz_rotate, degrees=True).as_matrix()
        transformation = np.eye(4)
        transformation[:3, :3] = rotaion
        transformation[:3, 3] = self.xyz_translate

        logger.info("rotation matrix: \n{}".format(rotaion))
        logger.info("transformation matrix: \n{}".format(transformation))

    def _rotate(self, vis, *, axis: str):
        "axis: 'x', 'y', 'z'"
        match axis:
            case "x":
                self.xyz_rotate[0] += self._rotate_unit
            case "y":
                self.xyz_rotate[1] += self._rotate_unit
            case "z":
                self.xyz_rotate[2] += self._rotate_unit
        self._update_pcd(vis)
        return True

    def _translate(self, vis, *, axis: str):
        "axis: 'x', 'y', 'z'"
        match axis:
            case "x":
                self.xyz_translate[0] += self._translate_unit
            case "y":
                self.xyz_translate[1] += self._translate_unit
            case "z":
                self.xyz_translate[2] += self._translate_unit
        self._update_pcd(vis)
        return True

    def _space(self, vis, action, mods):
        if action == 1 or action == 0:  # key down
            self._rotate_unit = self._rotate_unit * -1
            self._translate_unit = self._translate_unit * -1
        return False


if __name__ == "__main__":
    logger.info("Load frame")

    # NOTE: 这里只要加载好这三项即可，可以根据需要调整 np.savez("frame.npz", color=frame.color, depth=frame.depth, K=frame.K)
    data = np.load("../frame.npz")
    depth = data["depth"]
    color = data["color"]
    K = data["K"]

    logger.info("Transform frame to point cloud")

    H, W = depth.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    Y, X = np.meshgrid(np.arange(0, W), np.arange(0, H))  # [H, W]
    vertex = np.stack([(X - cx) / fx, (Y - cy) / fy, np.ones_like(X)], -1) * depth[..., None]  # [H, W, 3]

    points = vertex.reshape(-1, 3)  # [H*W, 3]
    colors = color.reshape(-1, 3)
    colors = colors[:, ::-1]  # BGR2RGB

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)

    logger.info("Now visualize point cloud, use keyboard to adjust pose until it looks good")
    logger.info("x, y, z for rotate, 1, 2, 3 for translate, space for change direction")
    Window(pcd)
