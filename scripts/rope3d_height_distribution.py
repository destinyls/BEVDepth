import os
import pickle
import math
import mmcv

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

from dataset.nusc_mv_det_dataset import map_name_from_general_to_detection
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box
from exps.bev_depth_lss_r50_256x704_128x128_24e import CLASSES

def get_lidar_3d_8points(obj_size, yaw_lidar, center_lidar):
    center_lidar = [center_lidar[0], center_lidar[1], center_lidar[2]]
    lidar_r = np.matrix(
        [[math.cos(yaw_lidar), -math.sin(yaw_lidar), 0], [math.sin(yaw_lidar), math.cos(yaw_lidar), 0], [0, 0, 1]]
    )
    l, w, h = obj_size
    center_lidar[2] = center_lidar[2] - h / 2
    corners_3d_lidar = np.matrix(
        [
            [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2],
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
            [0, 0, 0, 0, h, h, h, h],
        ]
    )
    corners_3d_lidar = lidar_r * corners_3d_lidar + np.matrix(center_lidar).T
    return corners_3d_lidar

if __name__ == "__main__":
    rope3d_data_infos = "data/rope3d/rope3d_12hz_infos_train.pkl"
    cams = [
        'CAM_FRONT'
    ]
    data_infos = mmcv.load(rope3d_data_infos)
    
    ego_locs_list = list()
    corners_list = list()
    valid_ego_locs_list = list()
    valid_corners_list = list()
    gt_boxes = list()
    for info in data_infos:
        ego2global_rotation = np.mean(
            [info['cam_infos'][cam]['ego_pose']['rotation'] for cam in cams],
            0)
        ego2global_translation = np.mean([
            info['cam_infos'][cam]['ego_pose']['translation'] for cam in cams
        ], 0)
        trans = -np.array(ego2global_translation)
        rot = Quaternion(ego2global_rotation).inverse
        for ann_info in info['ann_infos']:
            # Use ego coordinate.
            if (map_name_from_general_to_detection[ann_info['category_name']]
                    not in CLASSES
                    or ann_info['num_lidar_pts'] + ann_info['num_radar_pts'] <=
                    0):
                continue
            box = Box(
                ann_info['translation'],
                ann_info['size'],
                Quaternion(ann_info['rotation']),
                velocity=ann_info['velocity'],
            )
            box.translate(trans)
            box.rotate(rot)
            box_xyz = np.array(box.center)
            box_dxdydz = np.array(box.wlh)[[1, 0, 2]]
            box_yaw = np.array([box.orientation.yaw_pitch_roll[0]])
            box_velo = np.array(box.velocity[:2])
            gt_box = np.concatenate([box_xyz, box_dxdydz, box_yaw, box_velo])
            gt_boxes.append(gt_box)
      
            corners3d = get_lidar_3d_8points(box_dxdydz, box_yaw, box_xyz)
            
            corners_list.append(corners3d)
            ego_locs_list.append(box_xyz[:,np.newaxis])
            min_height, max_height = -1.0, 3.0
            if box_xyz[2] > min_height and box_xyz[2] < max_height:
                valid_ego_locs_list.append(box_xyz[:,np.newaxis])
            
            corners3d = corners3d.A
            mask = corners3d[2, :] > min_height 
            corners3d = corners3d[:, mask]
            mask = corners3d[2, :] < max_height 
            corners3d = corners3d[:, mask]
            if corners3d.shape[1] > 0:
                valid_corners_list.append(corners3d)
                
    corners_array = np.concatenate(corners_list, axis=1)
    ego_locs_array = np.concatenate(ego_locs_list, axis=1)
    valid_ego_locs_array = np.concatenate(valid_ego_locs_list, axis=1)
    valid_corners_array = np.concatenate(valid_corners_list, axis=1)

    print("--->: ", np.min(ego_locs_array[2]), np.max(ego_locs_array[2]), np.min(corners_array[2]), np.max(corners_array[2]))
    print("The precentage of location from", min_height, "to", max_height, "is: ", valid_ego_locs_array.shape[1] / ego_locs_array.shape[1])
    print("The precent of corners from", min_height, "to", max_height, "is: ", valid_corners_array.shape[1] / corners_array.shape[1])

    plt.figure(figsize=(16, 7.5))
    sns.set_palette("hls")
    sns.histplot(valid_ego_locs_array[2], color="r",bins=60, kde=True, legend=True)    
    plt.xlabel('Height', fontdict={'weight': 'normal', 'size': 25})
    plt.ylabel('Count', fontdict={'weight': 'normal', 'size': 25})
    plt.tick_params(labelsize=25)
    ax = plt.gca()
    bwith = 3
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)   
    plt.savefig('valid_ego_locs_array_hist.png')
    
    plt.figure(figsize=(16, 7.5))
    sns.set_palette("hls")
    sns.histplot(valid_corners_array[2], color="g",bins=45, kde=True, legend=True)    
    plt.xlabel('Height', fontdict={'weight': 'normal', 'size': 25})
    plt.ylabel('Count', fontdict={'weight': 'normal', 'size': 25})
    plt.tick_params(labelsize=25)
    ax = plt.gca()
    bwith = 3
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)   
    plt.savefig('valid_corners_array_hist.png')
