"""
Azure Kinect Dataset
"""

from pathlib import Path
from typing import NamedTuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pykinect_azure as pykinect
from loguru import logger


class RecordConfig(NamedTuple):
    width: int
    height: int
    intrinsics: list[float]  # fx, fy, cx, cy
    video_length: int  # in microsecond (1e-6 second)
    fps: int


class Frame(NamedTuple):
    timestamp: int  # in microsecond (1e-6 second)
    depth: np.ndarray
    color: np.ndarray
    K: np.ndarray  # 3x3 intrinsic matrix


class KinectDataset:
    def __init__(
        self,
        video_path: str,
        *,
        sample_timestep: int = -1,
        scale: float = 1.0,
        near: float = 0.1,
        far: float = 5.0,
        start: float = 0,
        end: float = -1,
    ):
        """
        sample_timestep: sample frames every sample_timestep microseconds. -1 for all frames.
        scale: downsample image by scale. 1 for no downsample.
        near, far: valid depth range in meters, drop invalid values.
        start, end: valid time range in seconds. end = -1 means end at the end of video.
        """
        assert Path(video_path).is_file() and video_path.endswith(".mkv"), "Invalid video: {}".format(video_path)
        self.video_path: str = video_path
        self.sample_timestep: int = int(sample_timestep)  # must be int
        self.scale: float = scale
        self.near: float = near
        self.far: float = far
        self.start: int = int(max(0, start * 1e6))  # store as microsecond

        pykinect.initialize_libraries()
        self.playback = pykinect.start_playback(video_path)
        self.record_config = self.get_record_config()
        if sample_timestep == -1:
            self.sample_timestep = round(1 / self.record_config.fps * 1e6)
        self.end: int = int(min(end * 1e6, self.playback.get_recording_length()))  # store as microsecond

        # timestamp of the first frame
        self.current_timestamp: int = self.start

    @property
    def finished(self) -> bool:
        return self.current_timestamp > self.end or self.current_timestamp < self.start

    def get_record_config(self) -> RecordConfig:
        video_length = self.playback.get_recording_length()
        playback_config = self.playback.get_record_configuration()
        # fps (0:5 FPS, 1:15 FPS, 2:30 FPS)
        match playback_config._handle.camera_fps:  # type: ignore
            case 0:
                camera_fps = 5
            case 1:
                camera_fps = 15
            case 2:
                camera_fps = 30
            case _:
                raise ValueError

        calibration = self.playback.get_calibration()

        width: int = calibration._handle.color_camera_calibration.resolution_width  # type:ignore
        height: int = calibration._handle.color_camera_calibration.resolution_height  # type:ignore
        # also can be retrieved by playback_config._handle.color_resolution (0:OFF, 1:720p, 2:1080p, 3:1440p, 4:1536p, 5:2160p, 6:3072p)

        intrinsics: list[float] = [
            calibration.color_params.fx,
            calibration.color_params.fy,
            calibration.color_params.cx,
            calibration.color_params.cy,
        ]

        return RecordConfig(width, height, intrinsics, video_length, camera_fps)

    def __iter__(self):
        self.current_timestamp: int = self.start
        while not self.finished:
            frame = self.get_next_frame()
            if frame is not None:
                yield frame
        logger.info(f"Finished loading dataset at timestamp {self.current_timestamp}")

    def preprocess_frame(self, frame: Frame):
        depth, color, K = frame.depth, frame.color, frame.K

        # downscale the image size if needed
        if self.scale < 1.0:
            # resized height and width
            h, w = (round(i * self.scale) for i in depth.shape)

            # 使用像素区域关系进行重采样
            color = cv2.resize(color, (w, h), interpolation=cv2.INTER_AREA)
            # 对深度图采用最近邻插值，感觉不使用线性插值是因为边界处线性插值获得的深度是没有意义的。
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)

            # NOTE: 相机内参会随着图像尺寸同比缩小
            K[:2, :] = K[:2, :] * self.scale

        assert isinstance(depth, np.ndarray) and isinstance(color, np.ndarray)  # for type hints
        depth = depth.astype(np.float32) / 1000.0  # Kinect provides depth in millimeters, turn into meters.
        depth[(depth > self.far) | (depth < self.near)] = 0
        # TODO: turn into tensor
        # TODO: maybe use torchvision.transforms
        # TODO: bilateral filter ?
        # depth1 = cv2.bilateralFilter(depth, 5, 0.2, 15)

        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)  # BGR to RGB
        return Frame(frame.timestamp, depth, color, K)

    def get_next_frame(self):
        self.current_timestamp += self.sample_timestep

        playback = self.playback
        playback.seek_timestamp(self.current_timestamp)
        res, capture = playback.update()
        if not res:
            logger.debug(f"failed to get frame at timestamp {self.current_timestamp}")
            return None

        # load color and depth image, both in color shape
        res1, depth = capture.get_transformed_depth_image()  # type: ignore
        res2, color = capture.get_color_image()  # type: ignore
        if not res1 or not res2:
            logger.warning(f"unexpexted frame ({self.current_timestamp}), skip")
            return None

        assert isinstance(depth, np.ndarray) and isinstance(color, np.ndarray)  # for type hints

        K = np.eye(3)
        K[[0, 1, 0, 1], [0, 1, 2, 2]] = self.record_config.intrinsics

        raw_frame = Frame(self.current_timestamp, depth, color, K)
        frame = self.preprocess_frame(raw_frame)
        return frame

    def __len__(self):
        "estimated frame count"
        # return int((self.end - self.start) / 1e6 * self.record_config.fps)
        return int((self.end - self.start) / self.sample_timestep)


def visualize_frame(frame: Frame):
    """visualize the frame using matplotlib"""
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle("frame at timestamp {}".format(frame.timestamp))
    axs[0].imshow(frame.color, aspect="equal")  # bgr to rgb
    axs[0].set_title("color")
    im = axs[1].imshow(frame.depth, cmap="coolwarm", aspect="equal")
    axs[1].set_title("depth")
    fig.colorbar(im)
    plt.tight_layout()
    plt.show()
