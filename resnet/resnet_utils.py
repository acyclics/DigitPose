from pypcd import pypcd
import numpy as np
import os

def get_pc(pcd_root, pcd_name, downsample_factor=1, p_2_keep=40000):
    pcd_path = os.path.join(pcd_root, pcd_name)
    pcd = pypcd.PointCloud.from_path(pcd_path)
    points = np.ones((pcd.points // downsample_factor, 4), dtype=np.float32)
    points[:, :3] = pcd.pc_data.view((np.float32, 3))[downsample_factor-1::downsample_factor, :]
    all_points = points.T
    return all_points[:, 0:p_2_keep]
