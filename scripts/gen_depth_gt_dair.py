import os
import cv2

import mmcv
import numpy as np
from pypcd import pypcd
from tqdm import tqdm

from scripts.vis_utils import *

def read_pcd(pcd_path):
    pcd = pypcd.PointCloud.from_path(pcd_path)
    pcd_np_points = np.zeros((pcd.points, 4), dtype=np.float32)
    pcd_np_points[:, 0] = np.transpose(pcd.pc_data["x"])
    pcd_np_points[:, 1] = np.transpose(pcd.pc_data["y"])
    pcd_np_points[:, 2] = np.transpose(pcd.pc_data["z"])
    pcd_np_points[:, 3] = np.transpose(pcd.pc_data["intensity"]) / 256.0
    del_index = np.where(np.isnan(pcd_np_points))[0]
    pcd_np_points = np.delete(pcd_np_points, del_index, axis=0)
    return pcd_np_points

def to_gt_boxes(ann_infos):
    gt_boxes = []
    for anno in ann_infos:
        x, y ,z = anno["translation"]
        yaw_lidar = anno["yaw_lidar"]
        l, w, h = anno["size"]
        gt_boxes.append([x, y, z, l, w, h, yaw_lidar])    
    gt_boxes = np.array(gt_boxes)
    return gt_boxes

if __name__ == '__main__':
    data_root = 'data/dair-v2x'
    info_path = 'data/dair-v2x/dair_12hz_infos_train.pkl'
    mmcv.mkdir_or_exist(os.path.join(data_root, 'depth_gt'))
    infos = mmcv.load(info_path)
    for info in tqdm(infos):
        sample_id = info["sample_token"].split('/')[1].split('.')[0]
        lidar_info = info["lidar_infos"]["LIDAR_TOP"]
        lidar_path = lidar_info["filename"]

        lidar_file_path = os.path.join(data_root, lidar_path)        
        camera_info = info["cam_infos"]["CAM_FRONT"]
        calibrated_sensor = camera_info["calibrated_sensor"]
        camera_intrinsic = calibrated_sensor["camera_intrinsic"]
        rotation_matrix = calibrated_sensor["rotation_matrix"]
        translation = calibrated_sensor["translation"]
        lidar2cam = np.eye(4)
        lidar2cam[:3,:3] = rotation_matrix
        lidar2cam[:3, 3] = translation.flatten()
        
        points = read_pcd(lidar_file_path)
        points[:, 3] = 1.0
        camera_points = np.matmul(lidar2cam, points.T).T
        depths = camera_points[:, 2]
        P = np.eye(4)
        P[:3, :3] = camera_intrinsic        
        img_poins = np.matmul(P, camera_points.T)
        img_poins = img_poins[:2, :] / img_poins[2, :]
        
        img_shape, min_dist = (1080, 1920), 5
        mask = np.ones(depths.shape[0], dtype=bool)
        mask = np.logical_and(mask, depths > min_dist)
        mask = np.logical_and(mask, img_poins[0, :] > 1)
        mask = np.logical_and(mask, img_poins[0, :] < img_shape[1] - 1)
        mask = np.logical_and(mask, img_poins[1, :] > 1)
        mask = np.logical_and(mask, img_poins[1, :] < img_shape[0] - 1)
        img_poins = img_poins[:, mask].astype(np.int32)
        depths = depths[mask]
        img_path = os.path.join(data_root, info["sample_token"])
        img = cv2.imread(img_path)
        img[img_poins[1,:], img_poins[0,:]] = (255, 255, 0)
        
        gt_boxes = to_gt_boxes(info["ann_infos"])
        demo(img_path, gt_boxes, rotation_matrix, translation, camera_intrinsic)

        # cv2.imwrite("demo.jpg", img)
        np.concatenate([img_poins[:2, :].T, depths[:, None]],
                       axis=1).astype(np.float32).flatten().tofile(
                           os.path.join(data_root, 'depth_gt',
                                        f'{sample_id}.jpg.bin'))
