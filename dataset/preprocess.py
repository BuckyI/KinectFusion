"""
该脚本主要对 TUM 数据集进行预处理

数据集提供了 depth maps (时间戳, 图片路径), rgb (时间戳, 图片路径), ground truth trajectory (时间戳, 位置和四元数姿态)
脚本首先是将三者合并，因为三者时间戳并非对齐的，所以根据时间相差最小的原则进行一一对应（关联时间戳），获得 rgb_dep_traj.txt
然后将四元数表示的姿态转化成矩阵形式，得到 raw_poses.npz
然后将相机内参与外参（刚才得到的位姿）再次合并，得到相机的投影矩阵，保存在 cameras.npz 中
数据集读取时，从投影矩阵中分解得到相机内外参数。

"""

import argparse
import math
import os
import shutil
import sys

import numpy as np
from tum_rgbd import get_calib

# 这里源代码不能正常导入
# 为了正确导入 utils.py
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from utils.utils import load_config


def read_file_list(filename):
    """
    Reads a trajectory from a text file.

    File format:
    The file format is "stamp d1 d2 d3 ...", where stamp denotes the time stamp (to be matched)
    and "d1 d2 d3.." is arbitary data (e.g., a 3D position and 3D orientation) associated to this timestamp.

    Input:
    filename -- File name

    Output:
    dict -- dictionary of (stamp,data) tuples

    """
    file = open(filename)
    data = file.read()
    lines = data.replace(",", " ").replace("\t", " ").split("\n")
    list = [[v.strip() for v in line.split(" ") if v.strip() != ""] for line in lines if len(line) > 0 and line[0] != "#"]
    list = [(float(l[0]), l[1:]) for l in list if len(l) > 1]
    return dict(list)


def associate(first_list, second_list, offset=0.0, max_difference=0.02):
    """
    Associate two dictionaries of (stamp,data). As the time stamps never match exactly, we aim
    to find the closest match for every input tuple.

    Input:
    first_list -- first dictionary of (stamp,data) tuples
    second_list -- second dictionary of (stamp,data) tuples
    offset -- time offset between both dictionaries (e.g., to models the delay between the sensors)
    max_difference -- search radius for candidate generation

    Output:
    matches -- list of matched tuples ((stamp1,data1),(stamp2,data2))

    """
    first_keys = list(first_list)
    second_keys = list(second_list)
    # 暴力枚举找出可能匹配的帧
    potential_matches = [
        (abs(a - (b + offset)), a, b) for a in first_keys for b in second_keys if abs(a - (b + offset)) < max_difference
    ]
    # 按照差值大小排序 尽可能找出最匹配的帧
    potential_matches.sort()
    # 去重
    matches = []
    for diff, a, b in potential_matches:
        if a in first_keys and b in second_keys:
            first_keys.remove(a)
            second_keys.remove(b)
            matches.append((a, b))
    # 按照时间戳排序
    matches.sort()
    return matches


def get_association(file_a, file_b, out_file):
    """生成关联文件 将两个文件中的时间戳对齐 可以嵌套使用"""
    first_list = read_file_list(file_a)
    second_list = read_file_list(file_b)
    matches = associate(first_list, second_list)
    with open(out_file, "w") as f:
        for a, b in matches:
            line = "%f %s %f %s\n" % (
                a,
                " ".join(first_list[a]),
                b,
                " ".join(second_list[b]),
            )
            f.write(line)


def tum2matrix(pose):
    """Return homogeneous rotation matrix from quaternion."""
    t = pose[:3]
    # under TUM format q is in the order of [x, y, z, w], need change to [w, x, y, z]
    quaternion = [pose[6], pose[3], pose[4], pose[5]]
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < np.finfo(np.float64).eps:
        return np.identity(4)

    # 这里是四元数转旋转矩阵的公式 和位移向量组合成齐次变换矩阵
    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array(
        [
            [1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0], t[0]],
            [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0], t[1]],
            [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2], t[2]],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def get_poses_from_associations(fname):
    poses = []
    with open(fname) as f:
        for line in f.readlines():
            pose_str = line.strip("\n").split(" ")[-7:]
            pose = [float(p) for p in pose_str]
            poses += [tum2matrix(pose)]

    return poses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # standard configs
    parser.add_argument(
        "--config",
        type=str,
        default="../configs/fr1_desk.yaml",
        help="Path to config file.",
    )
    args = load_config(parser.parse_args())
    out_dir = os.path.join(args.data_root, "processed")

    # create association files
    # 新生成的文件中包含RGB图像文件名、深度图像文件名、相机位姿三者的匹配结果
    # 因为位姿采样频率较高，所以先匹配RGB图像和深度图像
    get_association(
        os.path.join(args.data_root, "depth.txt"),
        os.path.join(args.data_root, "groundtruth.txt"),
        os.path.join(args.data_root, "dep_traj.txt"),
    )
    get_association(
        os.path.join(args.data_root, "rgb.txt"),
        os.path.join(args.data_root, "dep_traj.txt"),
        os.path.join(args.data_root, "rgb_dep_traj.txt"),
    )

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_rgb_dir = os.path.join(out_dir, "rgb")
    if not os.path.exists(out_rgb_dir):
        os.makedirs(out_rgb_dir)
    out_dep_dir = os.path.join(out_dir, "depth")
    if not os.path.exists(out_dep_dir):
        os.makedirs(out_dep_dir)

    # rename image files and save c2w poses
    poses = []
    with open(os.path.join(args.data_root, "rgb_dep_traj.txt")) as f:
        for i, line in enumerate(f.readlines()):
            line_list = line.strip().split(" ")
            rgb_file = line_list[1]
            shutil.copyfile(
                os.path.join(args.data_root, rgb_file),
                os.path.join(out_rgb_dir, "%04d.png" % i),
            )
            dep_file = line_list[3]
            shutil.copyfile(
                os.path.join(args.data_root, dep_file),
                os.path.join(out_dep_dir, "%04d.png" % i),
            )
            poses += [tum2matrix([float(x) for x in line_list[5:]])]

    # 存储每一帧的相机位姿（齐次矩阵形式）
    np.savez(os.path.join(out_dir, "raw_poses.npz"), c2w_mats=poses)

    # save projection matrices
    # K是相机内参矩阵 格式如下 其作用是将三维坐标变换到相机的成像平面上
    # K = [[fx, 0, cx],
    #      [0, fy, cy],
    #      [0, 0, 1]]
    K = np.eye(3)
    intri = get_calib()[args.data_type]
    K[0, 0] = intri[0]
    K[1, 1] = intri[1]
    K[0, 2] = intri[2]
    K[1, 2] = intri[3]
    camera_dict = np.load(os.path.join(out_dir, "raw_poses.npz"))
    poses = camera_dict["c2w_mats"]

    # 以下计算投影矩阵 投影矩阵将三维坐标变换到当前相机坐标系下的二维坐标
    P_mats = []
    for c2w in poses:
        w2c = np.linalg.inv(c2w)  # 4x4 外参矩阵 只取前三行 最后一行所表示的透视除法已经隐式地完成了
        P = K @ w2c[:3, :]  # 投影矩阵 3x4 = 内参矩阵 3x3 @ 外参矩阵 3x4
        P_mats += [P]

    # 存储每一帧的投影矩阵
    np.savez(os.path.join(out_dir, "cameras.npz"), world_mats=P_mats)
