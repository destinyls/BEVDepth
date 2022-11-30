import os
import math
import json
import cv2

import mmcv
import numpy as np
import pandas as pd
from pyquaternion import Quaternion
from tqdm import tqdm
from scipy.stats import norm

from scripts.vis_utils import *
from scipy.stats import *

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import seaborn as sns 
import numpy as np

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
    gt_bbox2d = []
    for item in my_json:
        gt_names.append(item["type"].lower())
        x, y, z = float(item["3d_location"]["x"]), float(item["3d_location"]["y"]), float(item["3d_location"]["z"])
        h, w, l = float(item["3d_dimensions"]["h"]), float(item["3d_dimensions"]["w"]), float(item["3d_dimensions"]["l"])                                                            
        lidar_yaw = float(item["rotation"])
        gt_boxes.append([x, y, z, l, w, h, lidar_yaw])
        gt_bbox2d.append([float(item["2d_box"]["xmin"]), float(item["2d_box"]["ymin"]), float(item["2d_box"]["xmax"]), float(item["2d_box"]["ymax"])])
    gt_boxes = np.array(gt_boxes)
    gt_bbox2d = np.array(gt_bbox2d)
    return gt_names, gt_boxes, gt_bbox2d

def load_data(dair_root, token):
    sample_id = token.split('/')[1].split('.')[0]
    img_pth = os.path.join(dair_root, token)
    camera_intrinsic_path = os.path.join(dair_root, "calib", "camera_intrinsic", sample_id + ".json")
    virtuallidar_to_camera_path = os.path.join(dair_root, "calib", "virtuallidar_to_camera", sample_id + ".json")
    label_path = os.path.join(dair_root, "label", "camera", sample_id + ".json")
    r_velo2cam, t_velo2cam = get_velo2cam(virtuallidar_to_camera_path)
    P = get_P(camera_intrinsic_path)
    gt_names, gt_boxes, gt_bbox2d = get_annos(label_path)
    return r_velo2cam, t_velo2cam, P, gt_names, gt_boxes, gt_bbox2d, img_pth

def cam2velo(r_velo2cam, t_velo2cam):
    Tr_velo2cam = np.eye(4)
    Tr_velo2cam[:3, :3] = r_velo2cam
    Tr_velo2cam[:3 ,3] = t_velo2cam.flatten()
    Tr_cam2velo = np.linalg.inv(Tr_velo2cam)
    r_cam2velo = Tr_cam2velo[:3, :3]
    t_cam2velo = Tr_cam2velo[:3, 3]
    return r_cam2velo, t_cam2velo
    
    
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
    lidar2cam[:3, 3] = translation.flatten()
    ground_points_lidar = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    ground_points_lidar = np.concatenate((ground_points_lidar, np.ones((ground_points_lidar.shape[0], 1))), axis=1)
    ground_points_cam = np.matmul(lidar2cam, ground_points_lidar.T).T
    denorm = -1 * equation_plane(ground_points_cam)
    
    return denorm

def degree2rad(degree):
    return degree * np.pi / 180

def sample_intrin_extrin_augmentation(sweepego2sweepsensor, roll_range=[0.0, 2.67], pitch_range=[0.0, 2.67]):
        # rectify sweepego2sweepsensor by roll
        # roll = np.random.normal(roll_range[0], roll_range[1])
        # roll = np.random.normal(roll_range[0], roll_range[1])
        roll = np.random.uniform(-5.0, 5.0)
        roll_rad = degree2rad(roll)
        rectify_roll = np.array([[math.cos(roll_rad), -math.sin(roll_rad), 0, 0], 
                                 [math.sin(roll_rad), math.cos(roll_rad), 0, 0], 
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 1]])
        sweepego2sweepsensor_rectify_roll = np.matmul(rectify_roll, sweepego2sweepsensor)
        
        # rectify sweepego2sweepsensor by pitch
        # pitch = np.random.normal(pitch_range[0], pitch_range[1])
        pitch = np.random.uniform(-5.0, 5.0)
        pitch_rad = degree2rad(pitch)
        rectify_pitch = np.array([[1, 0, 0, 0],
                                  [0,math.cos(pitch_rad), -math.sin(pitch_rad), 0], 
                                  [0,math.sin(pitch_rad), math.cos(pitch_rad), 0],
                                  [0, 0, 0, 1]])
        sweepego2sweepsensor_rectify_pitch = np.matmul(rectify_pitch, sweepego2sweepsensor_rectify_roll)
        return sweepego2sweepsensor_rectify_pitch

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

def generate_info_dair(dair_root, split):    
    infos = mmcv.load("scripts/single-infrastructure-split-data.json")
    split_list = infos[split]
    infos = list()
    img_locs_list = list()
    ego_locs_list = list()
    corners_list = list()
    valid_ego_locs_list = list()
    valid_corners_list = list()
    
    bbox_depth, bbox_height = dict(), dict()
    bbox_depth_list = []
    for sample_id in tqdm(split_list):
        token = "image/" + sample_id + ".jpg"
        r_velo2cam, t_velo2cam, camera_intrinsic, gt_names, gt_boxes, gt_bbox2d, img_pth = load_data(dair_root, token)
        Tr_velo2cam = np.eye(4)
        Tr_velo2cam[:3,:3] = r_velo2cam
        Tr_velo2cam[:3,3] = t_velo2cam[:,0]
        
        cam_key = str(round(t_velo2cam[0, 0], 4))
        if cam_key not in bbox_depth.keys():
            bbox_depth[cam_key] = []
            bbox_height[cam_key] = []
        
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
            
            denorm = get_denorm(r_velo2cam, t_velo2cam)
            r_cam2velo, t_cam2velo = cam2velo(r_velo2cam, t_velo2cam)
            calibrated_sensor = {"token": token, "sensor_token": token, "translation": t_cam2velo.flatten(), "rotation_matrix": r_cam2velo, "camera_intrinsic": camera_intrinsic}
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
            yaw_lidar = gt_box[6]            
            corners3d = get_lidar_3d_8points(lwh, yaw_lidar, loc)
            
            img_loc = np.matmul(Tr_velo2cam, np.array([[loc[0],loc[1],loc[2], 1]]).T)            
            image_loc = np.matmul(camera_intrinsic, img_loc[:3,:])
            image_loc = image_loc[:2,0] / image_loc[2,0]
            
            Tr_velo2cam_rectify = sample_intrin_extrin_augmentation(Tr_velo2cam)
            img_loc_rectify = np.matmul(Tr_velo2cam_rectify, np.array([[loc[0],loc[1],loc[2], 1]]).T)   
            image_loc_rectify = np.matmul(camera_intrinsic, img_loc_rectify[:3,:])
            image_loc_rectify = image_loc_rectify[:2,0] / image_loc_rectify[2,0]
            
            img_locs_list.append(img_loc[:3])
            ego_locs_list.append(loc[:,np.newaxis])
            corners_list.append(corners3d)
            
            min_height, max_height = -2.5, 1.5
            if loc[2] > min_height and loc[2] < max_height:
                valid_ego_locs_list.append(loc[:,np.newaxis])
                   
            corners3d = corners3d.A
            mask = corners3d[2, :] > min_height 
            corners3d = corners3d[:, mask]
            mask = corners3d[2, :] < max_height 
            corners3d = corners3d[:, mask]
            if corners3d.shape[1] > 0:
                valid_corners_list.append(corners3d)
            if category_name == "car":
                xmin, ymin, xmax, ymax = gt_bbox2d[idx]
                area = (ymax - ymin) * (xmax - xmin)
                if area > 15000 or loc[2] < -2.0 or len(bbox_depth[cam_key]) > 20000:
                    continue
                bbox_depth[cam_key].append([[area, img_loc[2,0], image_loc[1], image_loc_rectify[1]]])
                bbox_height[cam_key].append([[area, loc[2], image_loc[1], image_loc_rectify[1]]])
                bbox_depth_list.append([[area, img_loc[2,0]]])
                
            yaw_lidar = gt_box[6]
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
        
    img_locs_array = np.concatenate(img_locs_list, axis=1)
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

    plt.figure(figsize=(8, 8))
    plt.scatter(img_locs_array[2, :], 
                ego_locs_array[2, :],
                c='blue')
    
    plt.xlabel('depth', fontdict={'weight': 'normal', 'size': 25})
    plt.ylabel('height', fontdict={'weight': 'normal', 'size': 25})
    plt.tick_params(labelsize=20)
    plt.savefig('distribution.png')
        
    print("depth mean & var: ", np.mean(img_locs_array[2, :]), np.var(img_locs_array[2, :]))
    print("height mean & var: ", np.mean(ego_locs_array[2, :]), np.var(ego_locs_array[2, :]))

    plt.figure(figsize=(16, 7.5))
    x = img_locs_array[2,:]
    sns.set_palette("hls")
    sns.histplot(x, color="r",bins=31, kde=True, legend=True)    
    # plt.title(r'$\mu=75.45$, $\sigma=1258.95$', fontsize=25)
    plt.xlabel('Height', fontdict={'weight': 'normal', 'size': 25})
    plt.ylabel('Count', fontdict={'weight': 'normal', 'size': 25})
    plt.tick_params(labelsize=25)
    ax = plt.gca()
    bwith = 3
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)   
    plt.savefig('depth_hist.png')
    
    plt.figure(figsize=(16, 7.5))
    x = ego_locs_array[2,:]
    sns.set_palette("hls") 
    sns.histplot(x, color="g", bins=31, kde=True, legend=True)
    # plt.title(r'$\mu=-0.87$, $\sigma=0.09$', fontsize=25)
    plt.xlabel('Height', fontdict={'weight': 'normal', 'size': 25})
    plt.ylabel('Count', fontdict={'weight': 'normal', 'size': 25})
    plt.tick_params(labelsize=25)
    ax = plt.gca()
    bwith = 3
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)   
    plt.savefig('height_hist.png')
    
    scene1 = bbox_depth['-2.2854']
    scene1_array = np.concatenate(scene1, axis=0)
    
    plt.figure(figsize=(13.5, 11.5))
    plt.scatter(scene1_array[:, 0], 
                scene1_array[:, 1],
                c='lightcoral',
                s=50,
                marker='x',
                label = 'origin focal')

    plt.scatter(scene1_array[:, 0] * 0.5625,
                scene1_array[:, 1],
                c='royalblue',
                s=50,
                alpha = 1.0,
                label = '0.75 x focal')
    plt.legend(fontsize=45, markerscale=2.0)
    plt.xlabel('box area', fontdict={'weight': 'normal', 'size': 45})
    plt.ylabel('depth (m)', fontdict={'weight': 'normal', 'size': 45})
    plt.tick_params(labelsize=35)
    
    ax = plt.gca()
    bwith = 3
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)   
    plt.savefig('depth_area.png')
    
    scene1 = bbox_height['-2.2854']
    scene1_array = np.concatenate(scene1, axis=0)
    plt.figure(figsize=(13.5, 11.5))
    plt.scatter(scene1_array[:, 0], 
                -1 * scene1_array[:, 1],
                c='lightcoral',
                s=50,
                marker='x',
                label = 'origin focal')

    plt.scatter(scene1_array[:, 0] * 0.5625,
                -1 * scene1_array[:, 1],
                c='royalblue',
                s=50,
                alpha = 1.0,
                label = '0.75 x focal')
    plt.legend(fontsize=45, markerscale=2.0)
    plt.xlabel('box area', fontdict={'weight': 'normal', 'size': 45})
    plt.ylabel('height (m)', fontdict={'weight': 'normal', 'size': 45})
    plt.tick_params(labelsize=35)
    
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)  
    
    plt.savefig('height_area.png')
    
    ##################### depth ###########################
    scene1 = bbox_depth['-2.2854']
    scene1_array = np.concatenate(scene1, axis=0)
    plt.figure(figsize=(13.5, 11.5))
    plt.scatter(scene1_array[:, 2], 
                scene1_array[:, 1],
                c='lightcoral',
                s=50,
                marker='x',
                label = 'Original')
    
    plt.scatter(scene1_array[:, 3],
                scene1_array[:, 1],
                c='royalblue',
                s=50,
                alpha = 1.0,
                label = 'Noisy')
    plt.legend(fontsize=45, markerscale=2.5)
    plt.xlabel('v-coordinate of image', fontdict={'weight': 'normal', 'size': 45})
    plt.ylabel('depth (m)', fontdict={'weight': 'normal', 'size': 45})
    plt.tick_params(labelsize=35)
    
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)  
    plt.savefig('depth_v.png')
    
    ##################### height ###########################
    scene1 = bbox_height['-2.2854']
    scene1_array = np.concatenate(scene1, axis=0)
    plt.figure(figsize=(13.5, 11.5))
    plt.scatter(scene1_array[:, 2], 
                -1 * scene1_array[:, 1],
                c='lightcoral',
                s=50,
                marker='x',
                label = 'Original')
    plt.scatter(scene1_array[:, 3],
                -1 * scene1_array[:, 1],
                c='royalblue',
                s=50,
                alpha = 1.0,
                label = 'Noisy')
    plt.legend(fontsize=45, markerscale=2.5)
    plt.xlabel('v-coordinate of image', fontdict={'weight': 'normal', 'size': 45})
    plt.ylabel('height (m)', fontdict={'weight': 'normal', 'size': 45})
    plt.tick_params(labelsize=35)
    
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)  
    
    plt.savefig('height_v.png')
    
    return infos

def main():
    dair_root = "data/dair-v2x"
    # train_infos = generate_info_dair(dair_root, split='train')
    val_infos = generate_info_dair(dair_root, split='val')
    
    # mmcv.dump(train_infos, './data/dair-v2x/dair_12hz_infos_train.pkl')
    # mmcv.dump(val_infos, './data/dair-v2x/dair_12hz_infos_val.pkl')

if __name__ == '__main__':
    main()
