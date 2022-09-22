from operator import matmul
import os
import math
import json

import mmcv
import numpy as np
from pyquaternion import Quaternion
from tqdm import tqdm

from scripts.vis_utils import *
from scripts.gen_info_rope3d import get_cam2lidar
from evaluators.result2kitti import get_camera_3d_8points

name2nuscenceclass = {
    "car": "vehicle.car",
    "van": "vehicle.car",
    "truck": "vehicle.truck",
    "bus": "vehicle.bus.rigid",
    "cyclist": "vehicle.bicycle",
    "bicycle": "vehicle.bicycle",
    "tricyclist": "vehicle.bicycle",
    "motorcycle": "vehicle.bicycle",
    "motorcyclist": "vehicle.bicycle",
    "barrowlist": "vehicle.bicycle",
    "barrow": "vehicle.bicycle",
    "pedestrian": "human.pedestrian.adult",
    "traffic_cone": "movable_object.trafficcone",
}

def read_json(path_json):
    with open(path_json, "r") as load_f:
        my_json = json.load(load_f)
    return my_json

def get_velo2cam(path):
    my_json = read_json(path)
    t_velo2cam = np.array(my_json["translation"])
    r_velo2cam = np.array(my_json["rotation"])
    return r_velo2cam, t_velo2cam

def get_P(path):
    my_json = read_json(path)
    P = np.array(my_json["cam_K"]).reshape(3,3)
    return P

def get_annos(path):
    my_json = read_json(path)
    gt_names = []
    gt_boxes = []
    for item in my_json:
        gt_names.append(item["type"].lower())
        x, y, z = float(item["3d_location"]["x"]), float(item["3d_location"]["y"]), float(item["3d_location"]["z"])
        h, w, l = float(item["3d_dimensions"]["h"]), float(item["3d_dimensions"]["w"]), float(item["3d_dimensions"]["l"])                                                            
        lidar_yaw = float(item["rotation"])
        gt_boxes.append([x, y, z, l, w, h, lidar_yaw])
    gt_boxes = np.array(gt_boxes)
    return gt_names, gt_boxes

def load_data(dair_root, token):
    sample_id = token.split('/')[1].split('.')[0]
    camera_intrinsic_path = os.path.join(dair_root, "calib", "camera_intrinsic", sample_id + ".json")
    virtuallidar_to_camera_path = os.path.join(dair_root, "calib", "virtuallidar_to_camera", sample_id + ".json")
    label_path = os.path.join(dair_root, "label", "camera", sample_id + ".json")
    r_velo2cam, t_velo2cam = get_velo2cam(virtuallidar_to_camera_path)
    Tr_velo2cam = np.eye(4)
    Tr_velo2cam[:3, :3] = r_velo2cam
    Tr_velo2cam[:3, 3] = t_velo2cam.flatten()
     
    P = get_P(camera_intrinsic_path)
    gt_names, gt_boxes = get_annos(label_path)
    return r_velo2cam, t_velo2cam, Tr_velo2cam, P, gt_names, gt_boxes

def to_corners3d(obj_size, yaw_lidar, center_lidar):
    liadr_r = np.matrix([[math.cos(yaw_lidar), -math.sin(yaw_lidar), 0], [math.sin(yaw_lidar), math.cos(yaw_lidar), 0], [0, 0, 1]])
    l, w, h = obj_size
    corners_3d = np.matrix(
        [
            [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2],
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
            [0, 0, 0, 0, h, h, h, h],
        ]
    )
    corners_3d = liadr_r * corners_3d + np.matrix(center_lidar).T
    return corners_3d

def corners3d_yaw(corners_3d):
    x0, y0 = corners_3d[0, 0], corners_3d[1, 0]
    x3, y3 = corners_3d[0, 3], corners_3d[1, 3]
    dx, dy = x0 - x3, y0 - y3
    yaw = math.atan2(dy, dx)
    if yaw > math.pi:
        yaw = yaw - 2.0 * math.pi
    if yaw <= (-1 * math.pi):
        yaw = yaw + 2.0 * math.pi
    return yaw

def generate_info_dair(dair_root, split):    
    infos = mmcv.load("scripts/single-infrastructure-split-data.json")
    split_list = infos[split]
    infos = list()
    for sample_id in tqdm(split_list):
        token = "image/" + sample_id + ".jpg"
        _, _, Tr_velo2cam, camera_intrinsic, gt_names, gt_boxes = load_data(dair_root, token)
        r_cam2lidar, t_cam2lidar, Tr_cam2lidar, denorm = get_cam2lidar(os.path.join(dair_root, "denorm", sample_id + ".txt"))
        
        info = dict()
        cam_info = dict()
        info['sample_token'] = token
        info['timestamp'] = 1000000
        info['scene_token'] = token
        cam_names = ['CAM_FRONT']
        lidar_names = ['LIDAR_TOP']
        cam_infos, lidar_infos = dict(), dict()
        for cam_name in cam_names:
            cam_info = dict()
            cam_info['sample_token'] = token
            cam_info['timestamp'] = 1000000
            cam_info['is_key_frame'] = True
            cam_info['height'] = 1080
            cam_info['width'] = 1920
            cam_info['filename'] = token
            ego_pose = {"translation": [0.0, 0.0, 0.0], "rotation": [1.0, 0.0, 0.0, 0.0], "token": token, "timestamp": 1000000}
            cam_info['ego_pose'] = ego_pose
                        
            calibrated_sensor = {"token": token, "sensor_token": token, "translation": t_cam2lidar, "rotation_matrix": r_cam2lidar, "camera_intrinsic": camera_intrinsic}
            cam_info['calibrated_sensor'] = calibrated_sensor
            cam_info['denorm'] = denorm
            cam_infos[cam_name] = cam_info                  
        for lidar_name in lidar_names:
            lidar_info = dict()
            lidar_info['sample_token'] = token
            ego_pose = {"translation": [0.0, 0.0, 0.0], "rotation": [1.0, 0.0, 0.0, 0.0], "token": token, "timestamp": 1000000}
            lidar_info['ego_pose'] = ego_pose
            lidar_info['timestamp'] = 1000000
            lidar_info['filename'] = "velodyne/" + sample_id + ".pcd"
            lidar_info['calibrated_sensor'] = calibrated_sensor
            lidar_info['Tr_velo2cam'] = Tr_velo2cam
            lidar_infos[lidar_name] = lidar_info            
        info['cam_infos'] = cam_infos
        info['lidar_infos'] = lidar_infos
        info['sweeps'] = list()
        
        # demo(img_pth, gt_boxes, r_velo2cam, t_velo2cam, camera_intrinsic)   
        ann_infos = list()
        for idx in range(gt_boxes.shape[0]):
            category_name = gt_names[idx]
            if category_name not in name2nuscenceclass.keys():
                continue
            gt_box = gt_boxes[idx]
            lwh = gt_box[3:6]
            loc = gt_box[:3]    # need to certify
            yaw_lidar = gt_box[6]  # need to be confirmed
            
            corners_3d = to_corners3d(lwh, yaw_lidar, loc)
            corners_3d_extend = np.concatenate((corners_3d, np.ones((1, corners_3d.shape[1]))), axis=0)
            corners_3d_cam = np.matmul(Tr_velo2cam, corners_3d_extend)
            corners_3d_lidar = np.matmul(Tr_cam2lidar, corners_3d_cam)
            corners_3d_lidar = corners_3d_lidar[:3, :]
            yaw_lidar = corners3d_yaw(corners_3d_lidar)
                        
            loc = np.array([loc[0], loc[1], loc[2], 1])[:, np.newaxis]
            loc = np.matmul(Tr_velo2cam, loc)
            loc = np.matmul(Tr_cam2lidar, loc).squeeze(-1)[:3]
              
            rot_mat = np.array([[math.cos(yaw_lidar), -math.sin(yaw_lidar), 0], 
                                [math.sin(yaw_lidar), math.cos(yaw_lidar), 0], 
                                [0, 0, 1]])    
            rotation = Quaternion(matrix=rot_mat)
            ann_info = dict()
            ann_info["category_name"] = name2nuscenceclass[category_name]
            ann_info["translation"] = loc
            ann_info["rotation"] = rotation
            ann_info["yaw_lidar"] = yaw_lidar
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

