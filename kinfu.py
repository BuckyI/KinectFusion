import argparse
import os

import cv2
import numpy as np
import torch

# 与 kinfu_gui.py 相比，没有使用 open3d，而是使用了下面的两个包
import trimesh  # 处理 triangular meshes 的包，这里主要用于导出 mesh.ply
from matplotlib import pyplot as plt

from dataset.tum_rgbd import TUMDataset, TUMDatasetOnline
from fusion import TSDFVolumeTorch
from tracker import ICPTracker
from utils.utils import get_time, get_volume_setting, load_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # standard configs
    parser.add_argument("--config", type=str, default="configs/fr1_desk.yaml", help="Path to config file.")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory of saving results.")
    args = load_config(parser.parse_args())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    # NOTE: 这里 image_scale=0.25 让输出的数据集图片变为原始图像的 1/4，作用应该是减小内存占用 / 图片处理时间
    dataset = TUMDatasetOnline(os.path.join(args.data_root), device, near=args.near, far=args.far, img_scale=0.25)
    H, W = dataset.H, dataset.W

    vol_dims, vol_origin, voxel_size = get_volume_setting(args)
    tsdf_volume = TSDFVolumeTorch(vol_dims, vol_origin, voxel_size, device, margin=3, fuse_color=True)
    icp_tracker = ICPTracker(args, device)

    t, poses, poses_gt = list(), list(), list()
    curr_pose, depth1, color1 = None, None, None
    for i in range(0, len(dataset), 1):
        t0 = get_time()
        sample = dataset[i]
        color0, depth0, pose_gt, K = sample  # use live image as template image (0)
        # depth0[depth0 <= 0.5] = 0.

        if i == 0:  # initialize
            curr_pose = pose_gt
        else:  # tracking
            # 1. render depth image (1) from tsdf volume
            depth1, color1, vertex01, normal1, mask1 = tsdf_volume.render_model(
                curr_pose, K, H, W, near=args.near, far=args.far, n_samples=args.n_steps
            )
            T10 = icp_tracker(depth0, depth1, K)  # transform from 0 to 1
            curr_pose = curr_pose @ T10

        # fusion
        tsdf_volume.integrate(depth0, K, curr_pose, obs_weight=1.0, color_img=color0)
        t1 = get_time()
        t += [t1 - t0]
        print("processed frame: {:d}, time taken: {:f}s".format(i, t1 - t0))
        poses += [curr_pose.cpu().numpy()]
        poses_gt += [pose_gt.cpu().numpy()]

    avg_time = np.array(t).mean()
    print("average processing time: {:f}s per frame, i.e. {:f} fps".format(avg_time, 1.0 / avg_time))

    # 以下是评估相机位姿估计效果的代码
    # compute tracking ATE
    poses_gt = np.stack(poses_gt, 0)
    poses = np.stack(poses, 0)
    traj_gt = np.array(poses_gt)[:, :3, 3]
    traj = np.array(poses)[:, :3, 3]
    rmse = np.sqrt(np.mean(np.linalg.norm(traj_gt - traj, axis=-1) ** 2))
    print("RMSE: {:f}".format(rmse))
    # plt.plot(traj[:, 0], traj[:, 1])
    # plt.plot(traj_gt[:, 0], traj_gt[:, 1])
    # plt.legend(['Estimated', 'GT'])
    # plt.show()

    # save results
    # 保存重构的模型与相机位姿
    if args.save_dir is not None:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        verts, faces, norms, colors = tsdf_volume.get_mesh()
        partial_tsdf = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=norms, vertex_colors=colors)
        partial_tsdf.export(os.path.join(args.save_dir, "mesh.ply"))
        np.savez(os.path.join(args.save_dir, "traj.npz"), poses=poses)
        np.savez(os.path.join(args.save_dir, "traj_gt.npz"), poses=poses_gt)
