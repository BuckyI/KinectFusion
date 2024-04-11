"""
加载数据集
"""

from os import path

import cv2
import imageio
import numpy as np
import open3d as o3d
import torch
from tqdm import tqdm


def get_calib():
    """
    相机内参
    数据来源：https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats
    """
    # fx, fy, cx, cy
    return {
        "fr1": [517.306408, 516.469215, 318.643040, 255.313989],
        "fr2": [520.908620, 521.007327, 325.141442, 249.701764],
        "fr3": [535.4, 539.2, 320.1, 247.6],
    }


# Note,this step converts w2c (Tcw) to c2w (Twc)
def load_K_Rt_from_P(P):
    """
    modified from IDR https://github.com/lioryariv/idr
    """
    # 分解投影矩阵(3x4) 定义详见 https://amroamroamro.github.io/mexopencv/matlab/cv.decomposeProjectionMatrix.html
    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]  # 内参矩阵 3x3
    R = out[1]  # 旋转矩阵 3x3
    t = out[2]  # 平移向量 4x1

    # 归一化 K
    K = K / K[2, 2]
    intrinsics = np.eye(4)  # NOTE: 这里的内参矩阵给的 4x4 形式，只有左上角 3x3 有效
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    # 旋转矩阵 R 为正交矩阵，R^{-1} = R^T
    pose[:3, :3] = R.transpose()  # convert from w2c to c2w
    # 出于某种原因，位移 t 计算方法为 - t[:3] / t[3]
    # see: https://stackoverflow.com/questions/62686618/opencv-decompose-projection-matrix
    # 这里由于 w2c -> c2w，取逆，则位移取负数
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


class TUMDataset(torch.utils.data.Dataset):
    """
    TUM dataset loader, pre-load images in advance
    """

    def __init__(
        self,
        rootdir,
        device,
        near: float = 0.2,
        far: float = 5.0,
        img_scale: float = 1.0,  # image scale factor
        start: int = -1,
        end: int = -1,
    ):
        super().__init__()
        assert path.isdir(rootdir), f"'{rootdir}' is not a directory"
        self.device = device
        self.c2w_all = []
        self.K_all = []
        self.rgb_all = []
        self.depth_all = []

        # root should be tum_sequence
        data_path = path.join(rootdir, "processed")
        cam_file = path.join(data_path, "cameras.npz")
        print("LOAD DATA", data_path)

        # world_mats, normalize_mat
        cam_dict = np.load(cam_file)
        world_mats = cam_dict["world_mats"]  # K @ w2c

        d_min = []
        d_max = []
        # TUM saves camera poses in OpenCV convention
        for i, world_mat in enumerate(tqdm(world_mats)):
            # ignore all the frames betfore
            if start > 0 and i < start:
                continue
            # ignore all the frames after
            if 0 < end < i:
                break

            # 这里从 3x4 的投影矩阵 world_mat 中提取内参矩阵和外参矩阵
            intrinsics, c2w = load_K_Rt_from_P(world_mat)
            c2w = torch.tensor(c2w, dtype=torch.float32)
            # read images
            rgb = np.array(imageio.imread(path.join(data_path, "rgb/{:04d}.png".format(i)))).astype(np.float32)
            depth = np.array(imageio.imread(path.join(data_path, "depth/{:04d}.png".format(i)))).astype(np.float32)

            # NOTE: 深度值除以一个因数，让深度为 1 时，实际距离为 1 米。
            # 数据来源：https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats
            depth /= 5000.0  # TODO: put depth factor to args
            d_max += [depth.max()]
            d_min += [depth.min()]

            # [这是合并的别人的注释，不明觉厉] 双边滤波 但想了想又不滤了 因为后面要用原始深度图来重构 滤波后的图片用于相机轨迹估计
            # depth = cv2.bilateralFilter(depth, 5, 0.2, 15)
            # print(depth[depth > 0.].min())

            # ray-casting 对深度区间进行限制
            # 这里把 [near, far] 区间外的深度值都设为 -1.0，作为不可见区域
            invalid = (depth < near) | (depth > far)
            depth[invalid] = -1.0  # NOTE: 我个人感觉，截断的地方深度值直接设为 0 就好了，后续处理时过滤掉了 depth=0 的像素

            # downscale the image size if needed
            if img_scale < 1.0:
                full_size = list(rgb.shape[:2])  # H, W
                # 计算缩小后的图像尺寸，使用四舍五入获得整数
                rsz_h, rsz_w = [round(hw * img_scale) for hw in full_size]

                # TODO: figure out which way is better: skimage.rescale or cv2.resize
                # 缩小图片 see: https://zhuanlan.zhihu.com/p/38493205
                rgb = cv2.resize(rgb, (rsz_w, rsz_h), interpolation=cv2.INTER_AREA)  # 使用像素区域关系进行重采样
                # 对深度图采用最近邻插值，感觉不使用线性插值是因为边界处线性插值获得的深度是没有意义的。
                depth = cv2.resize(depth, (rsz_w, rsz_h), interpolation=cv2.INTER_NEAREST)

                # NOTE: 相机内参会随着图像尺寸同比缩小
                intrinsics[0, 0] *= img_scale
                intrinsics[1, 1] *= img_scale
                intrinsics[0, 2] *= img_scale
                intrinsics[1, 2] *= img_scale

            self.c2w_all.append(c2w)
            self.K_all.append(torch.from_numpy(intrinsics[:3, :3]))
            self.rgb_all.append(torch.from_numpy(rgb))
            self.depth_all.append(torch.from_numpy(depth))
        print("Depth min: {:f}".format(np.array(d_min).min()))
        print("Depth max: {:f}".format(np.array(d_max).max()))
        self.n_images = len(self.rgb_all)
        self.H, self.W, _ = self.rgb_all[0].shape

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        "rgb, depth, c2w (相机位姿), K (内参)"
        return (
            self.rgb_all[idx].to(self.device),
            self.depth_all[idx].to(self.device),
            self.c2w_all[idx].to(self.device),
            self.K_all[idx].to(self.device),
        )


class TUMDatasetOnline(torch.utils.data.Dataset):
    """
    Online TUM dataset loader, load images when __getitem__() is called
    """

    def __init__(
        self,
        rootdir,
        device,
        near: float = 0.2,
        far: float = 5.0,
        img_scale: float = 1.0,  # image scale factor
        start: int = -1,
        end: int = -1,
    ):
        super().__init__()
        assert path.isdir(rootdir), f"'{rootdir}' is not a directory"
        self.device = device
        self.img_scale = img_scale
        self.near = near
        self.far = far
        self.c2w_all = []
        self.K_all = []
        self.rgb_files_all = []
        self.depth_files_all = []

        # root should be tum_sequence
        data_path = path.join(rootdir, "processed")
        cam_file = path.join(data_path, "cameras.npz")
        print("LOAD DATA", data_path)

        # world_mats, normalize_mat
        cam_dict = np.load(cam_file)
        world_mats = cam_dict["world_mats"]  # K @ w2c

        # TUM saves camera poses in OpenCV convention
        for i, world_mat in enumerate(world_mats):
            # ignore all the frames betfore
            if start > 0 and i < start:
                continue
            # ignore all the frames after
            if 0 < end < i:
                break

            intrinsics, c2w = load_K_Rt_from_P(world_mat)
            c2w = torch.tensor(c2w, dtype=torch.float32)
            self.c2w_all.append(c2w)
            self.K_all.append(torch.from_numpy(intrinsics[:3, :3]))
            self.rgb_files_all.append(path.join(data_path, "rgb/{:04d}.png".format(i)))
            self.depth_files_all.append(path.join(data_path, "depth/{:04d}.png".format(i)))

        self.n_images = len(self.rgb_files_all)
        H, W, _ = np.array(imageio.imread(self.rgb_files_all[0])).shape
        self.H = round(H * img_scale)
        self.W = round(W * img_scale)

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        K = self.K_all[idx].to(self.device)
        c2w = self.c2w_all[idx].to(self.device)
        # read images
        rgb = np.array(imageio.imread(self.rgb_files_all[idx])).astype(np.float32)
        depth = np.array(imageio.imread(self.depth_files_all[idx])).astype(np.float32)
        depth /= 5000.0
        # NOTE: TODO: 目前这个双边滤波看起来效果并不明显…… 如果有时间可以研究下参数设定
        # depth = cv2.bilateralFilter(depth, 5, 0.2, 15)

        # ray-casting 对深度区间进行限制
        depth[depth < self.near] = 0.0
        depth[depth > self.far] = -1.0
        # downscale the image size if needed
        if self.img_scale < 1.0:
            full_size = list(rgb.shape[:2])
            rsz_h, rsz_w = [round(hw * self.img_scale) for hw in full_size]
            rgb = cv2.resize(rgb, (rsz_w, rsz_h), interpolation=cv2.INTER_AREA)
            depth = cv2.resize(depth, (rsz_w, rsz_h), interpolation=cv2.INTER_NEAREST)
            K[0, 0] *= self.img_scale
            K[1, 1] *= self.img_scale
            K[0, 2] *= self.img_scale
            K[1, 2] *= self.img_scale

        rgb = torch.from_numpy(rgb).to(self.device)
        depth = torch.from_numpy(depth).to(self.device)

        return rgb, depth, c2w, K
