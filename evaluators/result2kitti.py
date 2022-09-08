import os
import json
import mmcv
import math

import numpy as np

from tqdm import tqdm

from evaluators.kitti_utils import *
from evaluators.kitti_utils import kitti_common as kitti
from evaluators.kitti_utils.eval import kitti_eval

from scripts.gen_info_rope3d import *
from scipy.spatial.transform import Rotation as R

category_map = {"car": "Car", "van": "Car", "truck": "Truck", "bus": "Truck", "pedestrian": "Pedestrian", "bicycle": "Cyclist", "trailer": "Cyclist", "motorcycle": "Cyclist"}

def get_lidar_3d_8points(obj_size, yaw_lidar, center_lidar):
    center_lidar = [float(center_lidar[0]), float(center_lidar[1]), float(center_lidar[2])]
    lidar_r = np.matrix([[math.cos(yaw_lidar), -math.sin(yaw_lidar), 0], [math.sin(yaw_lidar), math.cos(yaw_lidar), 0], [0, 0, 1]])
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
    return corners_3d_lidar.T

def read_label_bboxes(label_path):
    with open(label_path, "r") as load_f:
        labels = json.load(load_f)
    boxes = []
    for label in labels:
        obj_size = [label["3d_dimensions"]["l"], label["3d_dimensions"]["w"], label["3d_dimensions"]["h"]]
        yaw_lidar = label["rotation"]
        center_lidar = [label["3d_location"]["x"], label["3d_location"]["y"], label["3d_location"]["z"]]
        box = get_lidar_3d_8points(obj_size, yaw_lidar, center_lidar)
        boxes.append(box)
    return boxes

def kitti_evaluation(pred_label_path, gt_label_path, metric_path="metric"):
    pred_annos, image_ids = kitti.get_label_annos(pred_label_path, return_ids=True)
    gt_annos = kitti.get_label_annos(gt_label_path, image_ids=image_ids)
    result, ret_dict = kitti_eval(gt_annos, pred_annos, ["Car", "Pedestrian", "Cyclist"], metric="R40")
    mAP_3d_moderate = ret_dict["KITTI/Car_3D_moderate_strict"]
    os.makedirs(os.path.join(metric_path, "R40"), exist_ok=True)
    with open(os.path.join(metric_path, "R40", 'epoch_result_{}.txt'.format(round(mAP_3d_moderate, 2))), "w") as f:
        f.write(result)
    print(result)
    return mAP_3d_moderate

def write_kitti_in_txt(pred_lines, path_txt):
    wf = open(path_txt, "w")
    for line in pred_lines:
        line_string = " ".join(line) + "\n"
        wf.write(line_string)
    wf.close()

def get_velo2cam(src_denorm_file):
    _, _, Tr_cam2lidar = get_cam2lidar(src_denorm_file)
    Tr_velo_to_cam = np.linalg.inv(Tr_cam2lidar) 
    r_velo2cam, t_velo2cam = Tr_velo_to_cam[:3, :3], Tr_velo_to_cam[:3, 3]
    t_velo2cam = t_velo2cam.reshape(3, 1)
    return Tr_velo_to_cam, r_velo2cam, t_velo2cam

def convert_point(point, matrix):
    pos =  matrix @ point
    return pos[0], pos[1], pos[2]

def normalize_angle(angle):
    alpha_tan = np.tan(angle)
    alpha_arctan = np.arctan(alpha_tan)
    if np.cos(angle) < 0:
        alpha_arctan = alpha_arctan + math.pi
    return alpha_arctan

def get_camera_3d_8points(obj_size, yaw_lidar, center_lidar, center_in_cam, r_velo2cam, t_velo2cam):
    liadr_r = np.matrix([[math.cos(yaw_lidar), -math.sin(yaw_lidar), 0], [math.sin(yaw_lidar), math.cos(yaw_lidar), 0], [0, 0, 1]])
    l, w, h = obj_size
    corners_3d_lidar = np.matrix(
        [
            [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2],
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
            [0, 0, 0, 0, h, h, h, h],
        ]
    )
    corners_3d_lidar = liadr_r * corners_3d_lidar + np.matrix(center_lidar).T
    corners_3d_cam = r_velo2cam * corners_3d_lidar + t_velo2cam
    
    x0, z0 = corners_3d_cam[0, 0], corners_3d_cam[2, 0]
    x3, z3 = corners_3d_cam[0, 3], corners_3d_cam[2, 3]
    dx, dz = x0 - x3, z0 - z3
    yaw = math.atan2(-dz, dx)
    alpha = yaw - math.atan2(center_in_cam[0], center_in_cam[2])
    if alpha > math.pi:
        alpha = alpha - 2.0 * math.pi
    if alpha <= (-1 * math.pi):
        alpha = alpha + 2.0 * math.pi
    alpha_arctan = normalize_angle(alpha)
    return alpha_arctan, yaw

def pcd_vis(pcd_path, boxes, save_file="demo.jpg", label_path=None):    
    range_list = [(-60, 60), (0, 100), (-2., -2.), 0.1]
    points_filter = PointCloudFilter(side_range=range_list[0], fwd_range=range_list[1], res=range_list[-1])
    bev_image = points_filter.get_bev_image(pcd_path)
    for n in range(len(boxes)):
        corner_points = boxes[n]
        x_img, y_img = points_filter.pcl2xy_plane(corner_points[:, 0], corner_points[:, 1])
        x_img = x_img[:, 0]
        y_img = y_img[:, 0]
        for i in np.arange(4):
            cv2.line(bev_image, (int(x_img[0]), int(y_img[0])), (int(x_img[1]), int(y_img[1])), (255,255,0), 2)
            cv2.line(bev_image, (int(x_img[0]), int(y_img[0])), (int(x_img[3]), int(y_img[3])), (255,255,0), 2)
            cv2.line(bev_image, (int(x_img[1]), int(y_img[1])), (int(x_img[2]), int(y_img[2])), (255,255,0), 2)
            cv2.line(bev_image, (int(x_img[2]), int(y_img[2])), (int(x_img[3]), int(y_img[3])), (255,255,0), 2)
    if label_path is not None:
        boxes = read_label_bboxes(label_path)
        for n in range(len(boxes)):
            corner_points = boxes[n]
            x_img, y_img = points_filter.pcl2xy_plane(corner_points[:, 0], corner_points[:, 1])
            x_img = x_img[:, 0]
            y_img = y_img[:, 0]
            for i in np.arange(4):
                cv2.line(bev_image, (int(x_img[0]), int(y_img[0])), (int(x_img[1]), int(y_img[1])), (0,0,255), 2)
                cv2.line(bev_image, (int(x_img[0]), int(y_img[0])), (int(x_img[3]), int(y_img[3])), (0,0,255), 2)
                cv2.line(bev_image, (int(x_img[1]), int(y_img[1])), (int(x_img[2]), int(y_img[2])), (0,0,255), 2)
                cv2.line(bev_image, (int(x_img[2]), int(y_img[2])), (int(x_img[3]), int(y_img[3])), (0,0,255), 2)
    cv2.imwrite(save_file, bev_image)
    
def bbbox2bbox(box3d, Tr_velo_to_cam, camera_intrinsic, img_size=[1920, 1080]):
    corners_3d = np.array(box3d)
    corners_3d_extend = np.concatenate(
        [corners_3d, np.ones((corners_3d.shape[0], 1), dtype=np.float32)], axis=1) 
    corners_3d_extend = np.matmul(Tr_velo_to_cam, corners_3d_extend.transpose(1, 0))
        
    corners_2d = np.matmul(camera_intrinsic, corners_3d_extend)
    corners_2d = corners_2d[:2] / corners_2d[2]
    box2d = np.array([min(corners_2d[0]), min(corners_2d[1]),
                      max(corners_2d[0]), max(corners_2d[1])])
    
    # [xmin, ymin, xmax, ymax]
    box2d[0] = max(box2d[0], 0.0)
    box2d[1] = max(box2d[1], 0.0)
    box2d[2] = min(box2d[2], img_size[0])
    box2d[3] = min(box2d[3], img_size[1])
    return box2d
    
def result2kitti(results_file, results_path, dair_root, demo=False):
    with open(results_file,'r',encoding='utf8')as fp:
        results = json.load(fp)["results"]
    with open("data/rope3d-kitti/ImageSets/train.json") as fp:
        token2sample = json.load(fp)
    with open("data/rope3d-kitti/ImageSets/val.json") as fp:
        token2sample_1 = json.load(fp)
    token2sample.update(token2sample_1)
    for sample_token in tqdm(results.keys()):
        sample_id = int(token2sample[sample_token])
        src_denorm_file = os.path.join(dair_root, "training/denorm", sample_token + ".txt")
        src_calib_file = os.path.join(dair_root, "training/calib", sample_token + ".txt")
        if not os.path.exists(src_denorm_file):
            src_denorm_file = os.path.join(dair_root, "validation/denorm", sample_token + ".txt")
            src_calib_file = os.path.join(dair_root, "validation/calib", sample_token + ".txt")
  
        Tr_velo_to_cam, r_velo2cam, t_velo2cam = get_velo2cam(src_denorm_file)
        camera_intrinsic = load_calib(src_calib_file)
        camera_intrinsic = np.concatenate([camera_intrinsic, np.zeros((camera_intrinsic.shape[0], 1))], axis=1)
        preds = results[sample_token]
        pred_lines = []
        bboxes = []
        for pred in preds:
            loc = pred["translation"]
            dim = pred["size"]
            rotation = pred["rotation"]
            Rq = [rotation[1], rotation[2], rotation[3], rotation[0]]
            Rm = R.from_quat(Rq)
            Rm_matrix = Rm.as_matrix()
            yaw_lidar = math.acos(Rm_matrix[0,0])
            if Rm_matrix[1,0] < 0:
                yaw_lidar = -1 * yaw_lidar
            detection_score = pred["detection_score"]
            class_name = pred["detection_name"]
            
            # w, h, l = dim[0], dim[1], dim[2]
            l, w, h = dim[0], dim[1], dim[2]
            x, y, z = loc[0], loc[1], loc[2]            
            bottom_center = [x, y, z]
            obj_size = [l, w, h]
            bottom_center_in_cam = r_velo2cam * np.matrix(bottom_center).T + t_velo2cam
            alpha, yaw = get_camera_3d_8points(
                obj_size, yaw_lidar, bottom_center, bottom_center_in_cam, r_velo2cam, t_velo2cam
            )
            cam_x, cam_y, cam_z = convert_point(np.array([x, y, z-h/2, 1]).T, Tr_velo_to_cam)
            box = get_lidar_3d_8points([w, l, h], -yaw_lidar, bottom_center)
            box2d = bbbox2bbox(box, Tr_velo_to_cam, camera_intrinsic)
            if detection_score > 0.45 and class_name in category_map.keys():
                i1 = category_map[class_name]
                i2 = str(0)
                i3 = str(0)
                i4 = str(round(alpha, 4))
                i5, i6, i7, i8 = (
                    str(round(box2d[0], 4)),
                    str(round(box2d[1], 4)),
                    str(round(box2d[2], 4)),
                    str(round(box2d[3], 4)),
                )
                i9, i11, i10 = str(round(h, 4)), str(round(w, 4)), str(round(l, 4))
                i12, i13, i14 = str(round(cam_x, 4)), str(round(cam_y, 4)), str(round(cam_z, 4))
                i15 = str(round(yaw, 4))
                line = [i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, str(round(detection_score, 4))]
                pred_lines.append(line)
                bboxes.append(box)
        os.makedirs(os.path.join(results_path, "data"), exist_ok=True)
        write_kitti_in_txt(pred_lines, os.path.join(results_path, "data", "{:06d}".format(sample_id) + ".txt"))       
        if demo:
            os.makedirs(os.path.join(results_path, "demo"), exist_ok=True)
            pcd_path = os.path.join(dair_root, "velodyne", "{:06d}".format(sample_id) + ".pcd")
            label_path = os.path.join(dair_root, "label/camera", "{:06d}".format(sample_id) + ".json")
            demo_file = os.path.join(results_path, "demo", "{:06d}".format(sample_id) + ".jpg")
            pcd_vis(pcd_path, bboxes, demo_file, label_path)
    return os.path.join(results_path, "data")

if __name__ == "__main__":
    
    root = "/root"
    dair_root = os.path.join(root, "DataSets/DAIR-V2X/cooperative-vehicle-infrastructure/infrastructure-side")    
    results_path = os.path.join(root, "BEVFormer/test/bevformer_small_dair_v2x")
    results_file = os.path.join(results_path, "Mon_Jul__4_21_44_10_2022/pts_bbox/results_nusc.json")
    # result2kitti(results_file, results_path, dair_root, demo=True)
    # pred_label_path = os.path.join(results_path, "data")
    # gt_label_path = os.path.join(root, "DataSets/DAIR-V2X/cooperative-vehicle-infrastructure/infrastructure-side-kitti/training/label_2")
    # kitti_evaluation(pred_label_path, gt_label_path)