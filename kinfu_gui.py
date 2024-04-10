import argparse
import os
import time

import numpy as np
import open3d as o3d
import torch

from dataset.tum_rgbd import TUMDataset, TUMDatasetOnline
from fusion import TSDFVolumeTorch
from tracker import ICPTracker
from utils import get_time, get_volume_setting, load_config

# 全局参数 感觉这里使用 dataclass 可读性会更强一点
vis_param = argparse.Namespace()
vis_param.frame_id = 0  # 当前帧数
vis_param.current_mesh = None  # 用于可视化的当前重建效果 o3d.geometry.TriangleMesh
vis_param.current_camera = None  # 用于可视化的相机 o3d.geometry.LineSet
vis_param.curr_pose = None  # 当前相机位姿，利用 ray-casting 以及 ICP 计算得到
# vis_param.n_frames: 总共的帧数
# vis_param.H: dataset.H
# vis_param.W: dataset.W
# vis_param.args: 运行程序脚本时，传入的原始参数 args， Dict 类型
# vis_param.dataset: 数据集 TUMDataset
# vis_param.map: 重建模型 TSDFVolumeTorch
# vis_param.tracker: ICPTracker
#
# ##


def refresh(vis):
    if vis:
        # This spares slots for meshing thread to emit commands.
        time.sleep(0.01)

    if vis_param.frame_id == vis_param.n_frames:
        # 播放完毕，重建结束
        return False

    sample = vis_param.dataset[vis_param.frame_id]  # 取出当前帧
    color0, depth0, pose_gt, K = sample  # use live image as template image (0)
    #  NOTE: 数据集中，提供了 pose 的真值，但是算法中实际并没有使用真值参与重建
    # depth0[depth0 <= 0.5] = 0.
    if vis_param.frame_id == 0:  # 第一帧的初始位姿使用了 ground-truth，不过我觉得使用单位矩阵就好了
        vis_param.curr_pose = pose_gt
    else:
        # render depth image (1) from tsdf volume
        depth1, color1, vertex01, normal1, mask1 = vis_param.map.render_model(
            vis_param.curr_pose,
            K,
            vis_param.H,
            vis_param.W,
            near=args.near,
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
    # update view-point
    if vis_param.args.follow_camera:
        follow_camera(vis, vis_param.curr_pose.cpu().numpy())
    # fusion
    vis_param.map.integrate(depth0, K, vis_param.curr_pose, obs_weight=1.0, color_img=color0)
    # update mesh
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/fr1_desk.yaml", help="Path to config file.")
    parser.add_argument("--follow_camera", action="store_true", help="Make view-point follow the camera motion")
    args = load_config(parser.parse_args())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    dataset = TUMDatasetOnline(os.path.join(args.data_root), device, near=args.near, far=args.far, img_scale=0.25)
    vol_dims, vol_origin, voxel_size = get_volume_setting(args)

    vis_param.args = args
    vis_param.dataset = dataset
    vis_param.map = TSDFVolumeTorch(vol_dims, vol_origin, voxel_size, device, margin=3, fuse_color=True)
    vis_param.tracker = ICPTracker(args, device)
    vis_param.n_frames = len(dataset)
    vis_param.H = dataset.H
    vis_param.W = dataset.W

    # visualize
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(width=1280, height=960)
    # vis.get_view_control().unset_constant_z_near()
    # vis.get_view_control().unset_constant_z_far()
    vis.get_render_option().mesh_show_back_face = True
    # NOTE: use `refresh` to create animation, see: https://www.open3d.org/docs/latest/tutorial/Advanced/customized_visualization.html
    vis.register_animation_callback(callback_func=refresh)

    # 增加坐标轴
    # coord_axes = o3d.geometry.TriangleMesh.create_coordinate_frame()
    # vis.add_geometry(coord_axes)

    # set initial view-point
    c2w0 = dataset[0][2]
    follow_camera(vis, c2w0.cpu().numpy())
    # start reconstruction and visualization
    vis.run()
    vis.destroy_window()
