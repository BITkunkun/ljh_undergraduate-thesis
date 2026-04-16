#!/usr/bin/env python3
"""
MATLAB 动态中心点 + 连续编队变换参考模型（与 main.m 注释块同源）。

来源: ``复现论文+code/code_ljh/main.m`` 中 “虚拟中心点动态轨迹与编队连续变换仿真”
（约第 132–199 行）。全局期望位置为::

    P_des = c(t) + R_z(theta(t)) @ (s(t) * base_shape)

其中 ``base_shape`` 为消除平移后的纯形状矩阵（各列为一架机的相对坐标），
与 MATLAB::

    base_desired_matrix = [1 1 1; 2 1 2; 1 2 2; 2 2 1].';
    base_center = mean(base_desired_matrix, 2);
    base_shape = base_desired_matrix - base_center;

一致。时间轴为三段连续变换，每段时长 ``MATLAB_SEGMENT_DURATION`` 秒（默认 5s），
总时长 15s，无停留间隙（与当前 Gazebo final.py 中带 gap 的时间轴可并存，由调用方映射 t）。
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

# --- 与 MATLAB 完全一致的模板顶点 (3 x N_AGENT)，列为智能体 ---
MATLAB_BASE_DESIRED = np.array(
    [
        [1.0, 2.0, 1.0, 2.0],
        [1.0, 1.0, 2.0, 2.0],
        [1.0, 2.0, 2.0, 1.0],
    ],
    dtype=float,
)

# 虚拟中心点路径节点 (m)
MATLAB_P0 = np.array([0.0, 0.0, 0.0], dtype=float)
MATLAB_P1 = np.array([10.0, 0.0, 5.0], dtype=float)
MATLAB_P2 = np.array([20.0, 0.0, -5.0], dtype=float)
MATLAB_P3 = np.array([30.0, 0.0, 0.0], dtype=float)

MATLAB_SEGMENT_DURATION = 5.0
MATLAB_DYNAMIC_DURATION = 3.0 * MATLAB_SEGMENT_DURATION  # 15 s


def base_shape_zero_mean(
    base_desired: np.ndarray = MATLAB_BASE_DESIRED,
) -> np.ndarray:
    """与 MATLAB ``base_desired_matrix - mean(...,2)`` 相同的零均值形状矩阵。"""
    c = np.mean(base_desired, axis=1, keepdims=True)
    return base_desired - c


def rot_z(theta: float) -> np.ndarray:
    """绕 Z 轴旋转矩阵 R_z(theta)，与 MATLAB 构造一致。"""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def compose_desired_positions(
    c: np.ndarray,
    s: float,
    theta: float,
    base_shape: np.ndarray,
) -> np.ndarray:
    """
    P_des = c + R_z(theta) @ (s * base_shape)，返回形状 (3, N) 的期望位置矩阵。

    ``c`` 为长度 3 的中心点；``base_shape`` 为 (3, N) 且列和为 0（每轴）。
    """
    c = np.asarray(c, dtype=float).reshape(3)
    R = rot_z(theta)
    return c.reshape(3, 1) + R @ (s * base_shape)


def matlab_dynamic_c_s_theta(
    t: float,
    p0: np.ndarray = MATLAB_P0,
    p1: np.ndarray = MATLAB_P1,
    p2: np.ndarray = MATLAB_P2,
    p3: np.ndarray = MATLAB_P3,
    segment_duration: float = MATLAB_SEGMENT_DURATION,
) -> Tuple[np.ndarray, float, float]:
    """
    MATLAB 动态场景中 ``t`` 时刻的 (c, s, theta)。

    - 阶段1 [0, T): p0→p1，s: 1 → 1.5，theta = 0
    - 阶段2 [T, 2T): p1→p2，s = 1.5，theta: 0 → π/2
    - 阶段3 [2T, 3T]: p2→p3，s: 1.5 → 1，theta = π/2

    ``t < 0`` 按 0 处理；``t > 3T`` 钳位到终点 (p3, s=1, theta=π/2)。
    """
    T = float(segment_duration)
    if T <= 0.0:
        raise ValueError("segment_duration must be positive")

    t = max(0.0, float(t))
    p0 = np.asarray(p0, dtype=float).reshape(3)
    p1 = np.asarray(p1, dtype=float).reshape(3)
    p2 = np.asarray(p2, dtype=float).reshape(3)
    p3 = np.asarray(p3, dtype=float).reshape(3)

    t_end = 3.0 * T
    if t >= t_end:
        return p3.copy(), 1.0, math.pi / 2.0

    if t <= T:
        u = t / T
        c = p0 + (p1 - p0) * u
        s = 1.0 + 0.5 * u
        theta = 0.0
    elif t <= 2.0 * T:
        u = (t - T) / T
        c = p1 + (p2 - p1) * u
        s = 1.5
        theta = (math.pi / 2.0) * u
    else:
        u = (t - 2.0 * T) / T
        c = p2 + (p3 - p2) * u
        s = 1.5 - 0.5 * u
        theta = math.pi / 2.0

    return c, s, theta


def matlab_dynamic_desired_positions(
    t: float,
    base_shape: np.ndarray,
    p0: np.ndarray = MATLAB_P0,
    p1: np.ndarray = MATLAB_P1,
    p2: np.ndarray = MATLAB_P2,
    p3: np.ndarray = MATLAB_P3,
    segment_duration: float = MATLAB_SEGMENT_DURATION,
) -> np.ndarray:
    """给定 ``t`` 与形状矩阵，返回 (3, N) 的 MATLAB 动态参考期望位置。"""
    c, s, th = matlab_dynamic_c_s_theta(
        t, p0=p0, p1=p1, p2=p2, p3=p3, segment_duration=segment_duration
    )
    return compose_desired_positions(c, s, th, base_shape)


if __name__ == "__main__":
    bs = base_shape_zero_mean()
    assert bs.shape == (3, 4)
    assert np.allclose(np.sum(bs, axis=1), 0.0)
    # 段边界连续性抽检
    c5, s5, th5 = matlab_dynamic_c_s_theta(5.0 - 1e-9)
    c5b, s5b, th5b = matlab_dynamic_c_s_theta(5.0 + 1e-9)
    assert np.allclose(c5, c5b) and abs(s5 - s5b) < 1e-6 and abs(th5 - th5b) < 1e-6
    P0 = matlab_dynamic_desired_positions(0.0, bs)
    P15 = matlab_dynamic_desired_positions(15.0, bs)
    print("base_shape column sums (should be ~0):", np.sum(bs, axis=1))
    print("t=0 center:", np.mean(P0, axis=1))
    print("t=15 center:", np.mean(P15, axis=1))
