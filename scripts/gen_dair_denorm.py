
import os
import json

import random
import numpy as np

from scripts.vis_utils import *
from scripts.gen_info_dair import get_annos, get_velo2cam

def augment(xyzs):
    axyz = np.ones((len(xyzs), 4))
    axyz[:, :3] = xyzs
    return axyz

def estimate(xyzs):
    axyz = augment(xyzs[:3])
    return np.linalg.svd(axyz)[-1][-1, :]

def is_inlier(coeffs, xyz, threshold):
    return np.abs(coeffs.dot(augment([xyz]).T)) < threshold

def run_ransac(data, estimate, is_inlier, sample_size, goal_inliers, max_iterations, stop_at_goal=True, random_seed=None):
    best_ic = 0
    best_model = None
    random.seed(random_seed)
    data = list(data)
    for i in range(max_iterations):
        s = random.sample(data, int(sample_size))
        m = estimate(s)
        ic = 0
        for j in range(len(data)):
            if is_inlier(m, data[j]):
                ic += 1
        if ic > best_ic:
            best_ic = ic
            best_model = m
            if ic > goal_inliers and stop_at_goal:
                break
    return best_model, best_ic

def plane_distance(locs, denorm):
    distance_list = []
    for idx in range(locs.shape[0]):
        loc = locs[idx]
        # dis = abs(np.sum(loc * np.array(denorm[:3])) + denorm[3]) / np.sqrt(denorm[0]**2 + denorm[1]**2 + denorm[2]**2)
        dis = (np.sum(loc * np.array(denorm[:3])) + denorm[3]) / np.sqrt(denorm[0]**2 + denorm[1]**2 + denorm[2]**2)
        distance_list.append(dis)
    return distance_list

if __name__ == '__main__':
    data_root = 'data/dair-v2x'
    info_path = 'data/dair-v2x/data_info.json'
    
    label_path = os.path.join(data_root, "label", "camera")
    calib_path = os.path.join(data_root, "calib", "virtuallidar_to_camera")
    denorm_path = os.path.join(data_root, "denorm")
    os.makedirs(denorm_path, exist_ok=True)

    with open(info_path, "r") as json_f:
        data_infos = json.load(json_f)
    
    intersection_locs = {}
    for info in data_infos:
        sample_id = info["image_path"].split('/')[1].split('.')[0]
        if info['intersection_loc'] not in intersection_locs.keys():
            intersection_locs[info['intersection_loc']] = [sample_id]
        else:
            intersection_locs[info['intersection_loc']].append(sample_id)
    
    denorm_dict = {}
    num = 0
    for loc_name, sample_list in intersection_locs.items():
        bottom_points = []
        for sample_id in sample_list:
            label_file = os.path.join(label_path, sample_id + ".json")
            virtuallidar_to_camera_file = os.path.join(calib_path, sample_id + ".json")
            gt_names, gt_boxes = get_annos(label_file)
            r_velo2cam, t_velo2cam = get_velo2cam(virtuallidar_to_camera_file)
            Tr_velo2cam = np.eye(4)
            Tr_velo2cam[:3, :3] = r_velo2cam
            Tr_velo2cam[:3, 3] = t_velo2cam.flatten()
            xyz = gt_boxes[:, :3]
            lwh = gt_boxes[:, 3:6]
            xyz_extend = np.concatenate((xyz, np.ones((xyz.shape[0], 1))), axis=-1)
            xyz_cam = np.matmul(Tr_velo2cam, xyz_extend.T).T
            xyz_cam[:, 1] = xyz_cam[:, 1] + 0.5 * lwh[:,2]
            bottom_points.append(xyz_cam)
            num += gt_boxes.shape[0]
            if num > 10000:
                break
            
        bottom_points = np.vstack(bottom_points)
        n = bottom_points.shape[0]
        max_iterations = 100
        goal_inliers = n * 0.8
        xyzs = bottom_points[:, :3]
        
        denorm, b = run_ransac(xyzs, estimate, lambda x, y: is_inlier(x, y, 0.01), 3, goal_inliers, max_iterations)     
        ref_height = np.abs(denorm[3]) / np.sqrt(denorm[0]**2 + denorm[1]**2 + denorm[2]**2)
        print(ref_height)
        denorm_dict[loc_name] = denorm
    
    for loc_name, sample_list in intersection_locs.items():
        for sample_id in sample_list:
            denorm_file = os.path.join(denorm_path, sample_id + ".txt")
            denorm = denorm_dict[loc_name]
            denorm_str = str(denorm[0]) + ' ' + str(denorm[1]) + ' ' + str(denorm[2]) + ' ' + str(denorm[3])
            with open(denorm_file, 'w') as f:
                f.write(denorm_str)