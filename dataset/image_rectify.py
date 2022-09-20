import math
import numpy as np
from numpy import random

class ImageRectify(object):
    def __init__(self, target_roll, pitch_abs):
        self.target_roll = target_roll
        self.pitch_abs = pitch_abs
        self.uvd = self.init_uvd(image_shape=[1080, 1920])

    def equation_plane(self, points): 
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
        return [a, b, c, d]
    
    def parse_roll_pitch(self, lidar2cam):
        ground_points_lidar = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
        ground_points_lidar = np.concatenate((ground_points_lidar, np.ones((ground_points_lidar.shape[0], 1))), axis=1)
        ground_points_cam = np.matmul(lidar2cam, ground_points_lidar.T).T
        denorm = -1 * self.equation_plane(ground_points_cam)
        
        origin_vector = np.array([0, 1.0, 0])
        target_vector_xy = np.array([denorm[0], denorm[1], 0.0])
        target_vector_yz = np.array([0.0, denorm[1], denorm[2]])
        target_vector_xy = target_vector_xy / np.sqrt(target_vector_xy[0]**2 + target_vector_xy[1]**2 + target_vector_xy[2]**2)       
        target_vector_yz = target_vector_yz / np.sqrt(target_vector_yz[0]**2 + target_vector_yz[1]**2 + target_vector_yz[2]**2)       
        roll = math.acos(np.inner(origin_vector, target_vector_xy))
        pitch = math.acos(np.inner(origin_vector, target_vector_yz))
        roll = -1 * self.rad2degree(roll) if target_vector_xy[0] > 0 else self.rad2degree(roll)
        pitch = -1 * self.rad2degree(pitch) if target_vector_yz[1] > 0 else self.rad2degree(pitch)
        return roll, pitch
    
    def rectify_cam_intrinsic(self, cam_intrinsic, ratio):
        cam_intrinsic_rectify = cam_intrinsic * ratio
        return cam_intrinsic_rectify
    
    def rectify_roll_params(self, lidar2cam, roll_status, target_roll=[-0.48,]):
        if len(target_roll) > 1:
            target_roll_status = np.random.uniform(target_roll[0], target_roll[1])
        else:
            target_roll_status = target_roll[0]
        roll = target_roll_status - roll_status   
        roll_rad = self.degree2rad(roll)
        rectify_roll = np.array([[math.cos(roll_rad), -math.sin(roll_rad), 0, 0], 
                                 [math.sin(roll_rad), math.cos(roll_rad), 0, 0], 
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 1]])
        lidar2cam_rectify = np.matmul(rectify_roll, lidar2cam)
        return lidar2cam_rectify
    
    def rectify_pitch_params(self, lidar2cam, pitch, pitch_abs=2.0):
        if len(pitch_abs) > 1:
            target_pitch_status = np.random.uniform(pitch + pitch_abs[0], pitch + pitch_abs[1])
        else:
            target_pitch_status = pitch_abs[0]
            
        pitch = -1 * (target_pitch_status - pitch)
        pitch = self.degree2rad(pitch)
        rectify_pitch = np.array([[1, 0, 0, 0],
                                  [0,math.cos(pitch), -math.sin(pitch), 0], 
                                  [0,math.sin(pitch), math.cos(pitch), 0],
                                  [0, 0, 0, 1]])
        lidar2cam_rectify = np.matmul(rectify_pitch, lidar2cam)
        return lidar2cam_rectify

    def rad2degree(self, radian):
        return radian * 180 / np.pi
    
    def degree2rad(self, degree):
        return degree * np.pi / 180
    
    def get_M(self, R, K, R_r, K_r):
        R_inv = np.linalg.inv(R)
        K_inv = np.linalg.inv(K)
        M = np.matmul(K_r, R_r)
        M = np.matmul(M, R_inv)
        M = np.matmul(M, K_inv)
        return M
    
    def init_uvd(self, image_shape=[1080, 1920]):
        u = range(image_shape[1])
        v = range(image_shape[0])
        xu, yv = np.meshgrid(u, v)
        uv = np.concatenate((xu[:,:,np.newaxis], yv[:,:,np.newaxis]), axis=2)
        uvd = np.concatenate((uv, np.ones((uv.shape[0], uv.shape[1], 1))), axis=-1) * 10
        uvd = uvd.reshape(-1, 3)
        return uvd
    
    def transform_with_M_bilinear(self, image, M):
        M = np.linalg.inv(M)
        uvd_new = np.matmul(M, self.uvd.T).T
        uv_new = uvd_new[:,:2] / (uvd_new[:,2][:, np.newaxis])
        uv_new_mask = uv_new.copy()
        uv_new_mask = uv_new_mask.reshape(image.shape[0], image.shape[1], 2)
        
        uv_new[:,0] = np.clip(uv_new[:,0], 0, image.shape[1]-2)
        uv_new[:,1] = np.clip(uv_new[:,1], 0, image.shape[0]-2)
        uv_new = uv_new.reshape(image.shape[0], image.shape[1], 2)
        
        image_new = np.zeros_like(image)
        corr_x, corr_y = uv_new[:,:,1], uv_new[:,:,0]
        point1 = np.concatenate((np.floor(corr_x)[:,:,np.newaxis].astype(np.int32), np.floor(corr_y)[:,:,np.newaxis].astype(np.int32)), axis=2)
        point2 = np.concatenate((point1[:,:,0][:,:,np.newaxis], (point1[:,:,1]+1)[:,:,np.newaxis]), axis=2)
        point3 = np.concatenate(((point1[:,:,0]+1)[:,:,np.newaxis], point1[:,:,1][:,:,np.newaxis]), axis=2)
        point4 = np.concatenate(((point1[:,:,0]+1)[:,:,np.newaxis], (point1[:,:,1]+1)[:,:,np.newaxis]), axis=2)

        fr1 = (point2[:,:,1]-corr_y)[:,:,np.newaxis] * image[point1[:,:,0], point1[:,:,1], :] + (corr_y-point1[:,:,1])[:,:,np.newaxis] * image[point2[:,:,0], point2[:,:,1], :]
        fr2 = (point2[:,:,1]-corr_y)[:,:,np.newaxis] * image[point3[:,:,0], point3[:,:,1], :] + (corr_y-point1[:,:,1])[:,:,np.newaxis] * image[point4[:,:,0], point4[:,:,1], :]
        image_new = (point3[:,:,0] - corr_x)[:,:,np.newaxis] * fr1 + (corr_x - point1[:,:,0])[:,:,np.newaxis] * fr2
        
        mask_1 = np.logical_or(uv_new_mask[:,:,0] < 0, uv_new_mask[:,:,0] > image.shape[1] -2)
        mask_2 = np.logical_or(uv_new_mask[:,:,1] < 0, uv_new_mask[:,:,1] > image.shape[0] -2)
        mask = np.logical_or(mask_1, mask_2)
        image_new[mask] = [0,0,0]
        image_new = image_new.astype(np.float32)
        return image_new
    
    def __call__(self, image, lidar2cam, cam_intrinsic):
        roll_init, pitch_init = self.parse_roll_pitch(lidar2cam)
        lidar2cam_roll_rectify = self.rectify_roll_params(lidar2cam, roll_init, target_roll=self.target_roll)
        lidar2cam_rectify = lidar2cam_roll_rectify
        if self.pitch_abs is not None:
            lidar2cam_pitch_rectify = self.rectify_pitch_params(lidar2cam_roll_rectify, pitch_init, pitch_abs=self.pitch_abs)            
            lidar2cam_rectify = lidar2cam_pitch_rectify
        cam_intrinsic_rectify = self.rectify_cam_intrinsic(cam_intrinsic, 1.2)
        
        M = self.get_M(lidar2cam[:3,:3], cam_intrinsic[:3,:3], lidar2cam_rectify[:3,:3], cam_intrinsic_rectify[:3,:3])
        image = self.transform_with_M_bilinear(image, M)
        return image, lidar2cam, cam_intrinsic