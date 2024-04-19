import argparse
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import open3d as o3d
import torch
import trimesh
from loguru import logger

from dataset.azure_kinect import KinectDataset, visualize_frame
from fusion import TSDFVolumeTorch
from tracker import ICPTracker
from utils.analyze import display_frame, save_frame
from utils.utils import get_time, get_volume_setting, load_config


@dataclass
class State:
    args: dict  # 行程序脚本时，传入的原始参数 args， Dict 类型
    device: torch.device

    dataset: KinectDataset  # 数据集 TUMDataset
    H: int  # image height
    W: int  # image width
    map: TSDFVolumeTorch  # 重建模型 TSDFVolumeTorch
    tracker: ICPTracker  # ICPTracker

    frame_id: int = 0  # 当前帧数
    n_frames: int = 0  # 总共的帧数
    current_mesh: Optional[o3d.geometry.TriangleMesh] = None  # 用于可视化的当前重建效果 o3d.geometry.TriangleMesh
    current_camera: Optional[o3d.geometry.LineSet] = None  # # 用于可视化的相机 o3d.geometry.LineSet
    curr_pose: Optional[torch.Tensor] = None  # 当前相机位姿，利用 ray-casting 以及 ICP 计算得到

    poses: list[tuple[int, torch.Tensor]] = field(default_factory=list)  # 记录历史位姿 (timestamp, pose)
    finished: bool = False  # 用来标记整个重建流程是否结束，简化动画更新逻辑


def refresh(vis):
    if vis_param.finished:
        return False

    if vis:
        # This spares slots for meshing thread to emit commands.
        # time.sleep(0.01)
        time.sleep(0.01)

    if vis_param.dataset.finished or vis_param.frame_id == vis_param.args.early_stop:
        logger.info("Finished Construnction")
        save_model(vis, False)  # 保存模型
        vis_param.finished = True
        return False

    frame = vis_param.dataset.get_next_frame()  # 取出当前帧
    logger.debug("Frame: {}/{} at {}".format(vis_param.frame_id, vis_param.n_frames, vis_param.dataset.current_timestamp))

    if frame is None:  # not finished, got an unexpected frame
        logger.warning("Got an unexpected frame")
        return True

    # use live image as template image (0)
    color0 = torch.from_numpy(frame.color).to(vis_param.device)
    depth0 = torch.from_numpy(frame.depth).to(vis_param.device)
    K: torch.Tensor = torch.from_numpy(frame.K).to(vis_param.device)

    # Tcw
    if vis_param.curr_pose is None:  # 第一帧的初始位姿
        if "init_pose" in vis_param.args:
            pose: torch.Tensor = torch.tensor(vis_param.args.init_pose, device=vis_param.device)
        else:
            pose = torch.eye(4, device=vis_param.device)
        vis_param.curr_pose = pose

    else:
        H, W = frame.depth.shape
        # render depth image (1) from tsdf volume
        depth1, color1, vertex01, normal1, mask1 = vis_param.map.render_model(
            vis_param.curr_pose,
            K,
            H,
            W,
            near=vis_param.args.near,
            far=vis_param.args.far,
            n_samples=vis_param.args.n_steps,
        )
        # NOTE: 这里对齐模板与当前帧时，只使用了深度信息
        T10 = vis_param.tracker(depth0, depth1, K)  # transform from 0 to 1
        # NOTE: vis_param.curr_pose 存储了上一帧的位姿，与 T10 相乘，得到当前帧的位姿
        # TODO: 没明白这里的计算公式，
        # 我想了想，相机的 pose 是全局 I 到相机的变换（三个轴在全局坐标系中的位置）， T01 是 0 到 1 的变换
        # 那么当前 pose 应该是 T10 @ last_pose ? 因为全局的点先到 0，再到 1
        vis_param.curr_pose = vis_param.curr_pose @ T10

    vis_param.poses.append((vis_param.dataset.current_timestamp, vis_param.curr_pose))  # 记录历史位姿

    # update view-point
    if vis_param.args.follow_camera:
        follow_camera(vis, vis_param.curr_pose.cpu().numpy())
    # fusion
    vis_param.map.integrate(depth0, K, vis_param.curr_pose, obs_weight=1.0, color_img=color0)
    # update mesh
    if vis_param.frame_id % 5 == 0:
        # NOTE: 不断提取可视化的 mesh，会减缓运行速度
        mesh = vis_param.map.to_o3d_mesh()
        if vis_param.current_mesh is not None:
            vis.remove_geometry(vis_param.current_mesh, reset_bounding_box=False)
        vis.add_geometry(mesh, reset_bounding_box=False)
        vis_param.current_mesh = mesh
    # update camera
    camera = draw_camera(vis_param.curr_pose.cpu().numpy())
    if vis_param.current_camera is not None:
        vis.remove_geometry(vis_param.current_camera, reset_bounding_box=False)
    vis.add_geometry(camera, reset_bounding_box=False)
    vis_param.current_camera = camera

    # 继续处理下一帧
    vis_param.frame_id += 1
    return True


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


def follow_camera(vis, c2w, z_offset=-2):
    """
    :param vis: visualizer handle
    :param c2w: world to camera transform Twc
    :param z_offset: offset along z-direction of eye wrt camera
    :return:
    """
    e2c = np.eye(4)
    e2c[2, 3] = z_offset  # z 轴给一个固定的偏移
    e2w = c2w @ e2c
    # TODO: 这里的原理还不太明白
    # e2w 应该是代表了视角 eye 到 world 的变换，不过设定观察视角时，需要的是 world 到 eye 的变换，即 eye 的位姿
    set_view(vis, np.linalg.inv(e2w))


def set_view(vis, w2e=np.eye(4)):
    """
    :param vis: visualizer handle
    :param w2e: world-to-eye transform
    :return:
    """
    vis_ctl = vis.get_view_control()
    cam = vis_ctl.convert_to_pinhole_camera_parameters()
    # world to eye w2e
    cam.extrinsic = w2e
    vis_ctl.convert_from_pinhole_camera_parameters(cam)


def get_view(vis):
    vis_ctl = vis.get_view_control()
    cam = vis_ctl.convert_to_pinhole_camera_parameters()
    print(cam.extrinsic)


def save_model(vis, overwrite=False):
    # verts, faces, norms, colors = vis_param.map.get_mesh()
    # partial_tsdf = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=norms, vertex_colors=colors)
    # partial_tsdf.export("mesh.ply")
    if overwrite or not (os.path.exists("mesh.ply") and os.path.exists("traj.pth")):
        mesh = vis_param.map.to_o3d_mesh()
        o3d.io.write_triangle_mesh("mesh.ply", mesh)
        torch.save(vis_param.poses, "traj.pth")
        logger.info("Model saved to mesh.ply and traj.pth")


if __name__ == "__main__":
    logger.add(".log")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/kinect.yaml", help="Path to config file.")
    parser.add_argument("--follow_camera", action="store_true", help="Make view-point follow the camera motion")
    args = load_config(parser.parse_args())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = KinectDataset(
        os.path.join(args.video_path),
        scale=args.scale,
        near=args.near,
        far=args.far,
        start=args.start,
        end=args.end,
    )

    vol_dims, vol_origin, voxel_size = get_volume_setting(args)
    vis_param = State(
        args=args,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        current_mesh=None,
        current_camera=None,
        curr_pose=None,
        map=TSDFVolumeTorch(vol_dims, vol_origin, voxel_size, device, margin=3, fuse_color=True),
        tracker=ICPTracker(args, device),
        dataset=dataset,
        frame_id=0,
        n_frames=len(dataset),  # estimated total frames
        H=dataset.record_config.height,
        W=dataset.record_config.width,
    )

    # # DEBUG: CHECK
    # while not (frame := dataset.get_next_frame()):
    #     continue
    # logger.debug(f"frame: {frame.timestamp=}, {frame.color.shape=}, {frame.depth.shape=}")
    # logger.debug(f"K: {frame.K=}")
    # visualize_frame(frame)
    # display_frame(frame.depth, frame.color, frame.K)
    # save_frame(frame.depth, frame.color, frame.K)

    # visualize
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(width=1280, height=960)
    # vis.get_view_control().unset_constant_z_near()
    # vis.get_view_control().unset_constant_z_far()
    vis.get_render_option().mesh_show_back_face = True
    # NOTE: use `refresh` to create animation, see: https://www.open3d.org/docs/latest/tutorial/Advanced/customized_visualization.html
    vis.register_animation_callback(callback_func=refresh)
    vis.register_key_callback(83, save_model)  # 保存重建模型，83 is 'S'

    # 增加坐标轴
    coord_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    vis.add_geometry(coord_axes)

    vol_bounds = np.array(vis_param.args.vol_bounds).reshape(3, 2)
    bbox = o3d.geometry.AxisAlignedBoundingBox(vol_bounds[:, 0], vol_bounds[:, 1])
    bbox.color = (1, 0, 0)
    vis.add_geometry(bbox)

    # set initial view-point
    # follow_camera(vis, np.eye(4))
    # start reconstruction and visualization
    vis.run()
    vis.destroy_window()
