"""
这里定义了一些工具函数，主要是对配置选项的处理。
"""

import argparse
import time

import numpy as np
import torch
import yaml
from addict import Dict


class ForceKeyErrorDict(Dict):
    """
    `addict.Dict` 提供了一种可以通过 `.` 访问键对应值的方法: dict.key = value
    默认情况下，如果键不存在，不会报错
    创建该类目的为，读取不存在的键时，强制报错
    """

    def __missing__(self, key):
        raise KeyError(key)


def load_yaml(path):
    "加载配置文件"
    with open(path, encoding="utf8") as yaml_file:
        config_dict = yaml.load(yaml_file, Loader=yaml.FullLoader)
        config = ForceKeyErrorDict(**config_dict)

    return config


def load_config(args):
    """把 args 中的键值对以及 args.config 指向文件的配置信息合并，
    以 ForceKeyErrorDict 类型返回"""
    config_dict = load_yaml(args.config)
    # merge args and config
    other_dict = vars(args)
    config_dict.update(other_dict)
    return config_dict


def get_volume_setting(args):
    """读取 args 设置中关于体素的设置，返回TSDF的尺寸(int)、原点、体素大小
    从配置文件中获取体素网格的设置
    :return:
        vol_dims: 体素网格各个维度的体素数量
        vol_origin: 体素网格原点在世界坐标系下的坐标
        voxel_size: 体素网格的体素大小（米）
    """
    voxel_size = args.voxel_size
    vol_bnds = np.array(args.vol_bounds).reshape(3, 2)  # [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
    vol_dims = (vol_bnds[:, 1] - vol_bnds[:, 0]) // voxel_size + 1  # x, y, z 的长度（voxel 格数）
    vol_origin = vol_bnds[:, 0]  # 以 x_min, y_min, z_min 为原点
    return vol_dims, vol_origin, voxel_size


def get_time():
    """
    :return: get timing statistics
    """
    torch.cuda.synchronize()  # 等待当前 CUDA 设备上所有的异步操作完成，用于统计计算时间
    return time.time()
