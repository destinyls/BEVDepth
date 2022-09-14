import os
import csv
import math
import random
import cv2

import mmcv
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from pyquaternion import Quaternion
from tqdm import tqdm

from scipy.spatial.transform import Rotation as R

name2nuscenceclass = {
    "car": "vehicle.car",
    "van": "vehicle.car",
    "truck": "vehicle.truck",
    "bus": "vehicle.bus.rigid",
    "cyclist": "vehicle.bicycle",
    "bicycle": "vehicle.bicycle",
    "tricyclist": "vehicle.trailer",
    "motorcycle": "vehicle.motorcycle",
    "barrow": "vehicle.bicycle",
    "pedestrian": "human.pedestrian.adult",
    "traffic_cone": "movable_object.trafficcone",
}

def equation_plane(points): 
    x1, y1, z1 = points[0, 0], points[0, 1], points[0, 2]
    x2, y2, z2 = points[1, 0], points[1, 1], points[1, 2]
    x3, y3, z3 = points[2, 0], points[2, 1], points[2, 2]
    a1 = x2 - x1
    b1 = y2 - y1
    c1 = z2 - z1
    a2 = x3 - x1
    b2 = y3 - y1
    c2 = z3 - z1
    a = b1 * c2 - b2 * c1
    b = a2 * c1 - a1 * c2
    c = a1 * b2 - b1 * a2
    d = (- a * x1 - b * y1 - c * z1)
    return np.array([a, b, c, d])

def get_denorm(rotation_matrix, translation):
    lidar2cam = np.eye(4)
    lidar2cam[:3, :3] = rotation_matrix
    lidar2cam[:3, 3] = translation
    ground_points_lidar = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    ground_points_lidar = np.concatenate((ground_points_lidar, np.ones((ground_points_lidar.shape[0], 1))), axis=1)
    ground_points_cam = np.matmul(lidar2cam, ground_points_lidar.T).T
    denorm = equation_plane(ground_points_cam)
    return denorm

def generate_info_dair(dair_root, split):
    dair_pkl = os.path.join(dair_root, "dair_v2x_i_infos_temporal_{}.pkl".format(split))
    dair_infos = mmcv.load(dair_pkl)
    
    infos = list()
    for dair_info in tqdm(dair_infos["infos"]):
        info = dict()
        cam_info = dict()
        info['sample_token'] = dair_info["token"]
        info['timestamp'] = dair_info["timestamp"]
        info['scene_token'] = dair_info["scene_token"]
        
        cam_names = ['CAM_FRONT']
        lidar_names = ['LIDAR_TOP']
        cam_infos, lidar_infos = dict(), dict()
        token = dair_info["token"]
        for cam_name in cam_names:
            cam_info = dict()
            token = dair_info["cams"][cam_name]["sample_data_token"]
            cam_info['sample_token'] = token
            cam_info['timestamp'] = 1000000
            cam_info['is_key_frame'] = True
            cam_info['height'] = 1080
            cam_info['width'] = 1920
            cam_info['filename'] = token
            ego_pose = {"translation": [0.0, 0.0, 0.0], "rotation": [1.0, 0.0, 0.0, 0.0], "token": token, "timestamp": 1000000}
            cam_info['ego_pose'] = ego_pose
            
            camera_intrinsic = dair_info["cams"][cam_name]["cam_intrinsic"]
            translation = dair_info["cams"][cam_name]["sensor2lidar_translation"]
            rotation_matrix = dair_info["cams"][cam_name]["sensor2lidar_rotation"]
            denorm = get_denorm(rotation_matrix, translation)
            calibrated_sensor = {"token": token, "sensor_token": token, "translation": translation, "rotation_matrix": rotation_matrix, "camera_intrinsic": camera_intrinsic}
            cam_info['calibrated_sensor'] = calibrated_sensor
            cam_info['denorm'] = denorm
            cam_infos[cam_name] = cam_info
            
        for lidar_name in lidar_names:
            lidar_info = dict()
            lidar_info['sample_token'] = token
            ego_pose = {"translation": [0.0, 0.0, 0.0], "rotation": [1.0, 0.0, 0.0, 0.0], "token": token, "timestamp": 1000000}
            lidar_info['ego_pose'] = ego_pose
            lidar_info['timestamp'] = 1000000
            lidar_info['filename'] = "velodyne/" + dair_info["lidar_path"].split('/')[-1]
            lidar_info['calibrated_sensor'] = calibrated_sensor
            lidar_infos[lidar_name] = lidar_info            
        info['cam_infos'] = cam_infos
        info['lidar_infos'] = lidar_infos
        info['sweeps'] = list()
        gt_boxes = dair_info["gt_boxes"]
        gt_names = dair_info["gt_names"]
        
        ann_infos = list()
        for idx in range(gt_boxes.shape[0]):
            category_name = gt_names[idx]
            if category_name not in name2nuscenceclass.keys():
                continue
            gt_box = gt_boxes[idx]
            lwh = gt_box[3:6]
            loc = gt_box[:3]    # need to certify
            yaw_lidar = gt_box[6]
            rot_mat = np.array([[math.cos(yaw_lidar), -math.sin(yaw_lidar), 0], 
                                [math.sin(yaw_lidar), math.cos(yaw_lidar), 0], 
                                [0, 0, 1]])    
            rotation = Quaternion(matrix=rot_mat)
            ann_info = dict()
            ann_info["category_name"] = name2nuscenceclass[category_name]
            ann_info["translation"] = loc
            ann_info["rotation"] = rotation
            ann_info["size"] = lwh
            ann_info["prev"] = ""
            ann_info["next"] = ""
            ann_info["sample_token"] = token
            ann_info["instance_token"] = token
            ann_info["token"] = token
            ann_info["visibility_token"] = "0"
            ann_info["num_lidar_pts"] = 3
            ann_info["num_radar_pts"] = 0            
            ann_info['velocity'] = np.zeros(3)
            ann_infos.append(ann_info)
        info['ann_infos'] = ann_infos
        infos.append(info)
    return infos

def main():
    dair_root = "data/dair-v2x"
    train_infos = generate_info_dair(dair_root, split='train')
    val_infos = generate_info_dair(dair_root, split='val')
    
    mmcv.dump(train_infos, './data/dair-v2x/dair_12hz_infos_train.pkl')
    mmcv.dump(val_infos, './data/dair-v2x/dair_12hz_infos_val.pkl')

if __name__ == '__main__':
    main()
