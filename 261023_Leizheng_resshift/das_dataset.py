import os
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

VALID_SPLITS = ("train", "val", "test")


class PairDataset(Dataset):
    """
    半合成 DAS 配对波场数据集读取类。
    基于只读内存映射 (mmap_mode='r')，实现低内存开销、确定性配对读取。
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        expected_hw: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        if split not in VALID_SPLITS:
            raise ValueError(
                f"Invalid split: {split!r}. Must be one of {VALID_SPLITS}"
            )
        self.data_dir = str(data_dir)
        self.split = split
        self.expected_hw = tuple(expected_hw) if expected_hw is not None else None

        x_path = os.path.join(self.data_dir, f"{split}_x.npy")
        y_path = os.path.join(self.data_dir, f"{split}_y.npy")

        if not os.path.exists(x_path):
            raise FileNotFoundError(f"Input file not found: {x_path}")
        if not os.path.exists(y_path):
            raise FileNotFoundError(f"Target file not found: {y_path}")

        # 使用只读 mmap 加载，禁止 pickle
        self.x_mmap = np.load(x_path, mmap_mode="r", allow_pickle=False)
        self.y_mmap = np.load(y_path, mmap_mode="r", allow_pickle=False)

        # 检查 dtype 必须为 float16
        if self.x_mmap.dtype != np.float16:
            raise ValueError(
                f"Expected float16 dtype for {x_path}, got {self.x_mmap.dtype}"
            )
        if self.y_mmap.dtype != np.float16:
            raise ValueError(
                f"Expected float16 dtype for {y_path}, got {self.y_mmap.dtype}"
            )

        # 检查维度与形状
        if self.x_mmap.ndim != 3:
            raise ValueError(
                f"Expected 3D array [N, H, W] for {x_path}, got ndim={self.x_mmap.ndim} shape={self.x_mmap.shape}"
            )
        if self.y_mmap.ndim != 3:
            raise ValueError(
                f"Expected 3D array [N, H, W] for {y_path}, got ndim={self.y_mmap.ndim} shape={self.y_mmap.shape}"
            )

        if self.x_mmap.shape != self.y_mmap.shape:
            raise ValueError(
                f"Shape mismatch between x ({self.x_mmap.shape}) and y ({self.y_mmap.shape}) in split {split!r}"
            )

        n, h, w = self.x_mmap.shape
        if n <= 0 or h <= 0 or w <= 0:
            raise ValueError(
                f"Invalid zero or negative dimension in shape {self.x_mmap.shape} for split {split!r}"
            )

        if self.expected_hw is not None:
            if (h, w) != self.expected_hw:
                raise ValueError(
                    f"Spatial dimensions {(h, w)} do not match expected_hw {self.expected_hw} for split {split!r}"
                )

        self._length = n
        self._h = h
        self._w = w

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> Dict[str, Union[torch.Tensor, int]]:
        if index < -self._length or index >= self._length:
            raise IndexError(
                f"Index {index} out of range for dataset of size {self._length} in split {self.split!r}"
            )
        if index < 0:
            index += self._length

        # 独立拷贝并转换为 float32，增加单通道维 [1, H, W]
        x_arr = np.array(self.x_mmap[index], dtype=np.float32, copy=True)
        y_arr = np.array(self.y_mmap[index], dtype=np.float32, copy=True)

        x_tensor = torch.from_numpy(x_arr).unsqueeze(0)  # [1, H, W]
        y_tensor = torch.from_numpy(y_arr).unsqueeze(0)  # [1, H, W]

        # 有限性检验
        if not torch.isfinite(x_tensor).all():
            raise ValueError(
                f"Non-finite values (NaN/Inf) encountered in noisy input x at index {index} (split {self.split!r})"
            )
        if not torch.isfinite(y_tensor).all():
            raise ValueError(
                f"Non-finite values (NaN/Inf) encountered in clean target y at index {index} (split {self.split!r})"
            )

        return {
            "noisy": x_tensor,
            "clean": y_tensor,
            "local_index": index,
        }
