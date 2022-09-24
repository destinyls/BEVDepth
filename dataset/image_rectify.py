import math
import numpy as np

class ImageRectify(object):
    def __init__(self, image_shape, roll_range=[-2.0, 2.0], pitch_range=[-1.0, 1.0], ratio=0.95):
        self.roll_range = roll_range
        self.pitch_range = pitch_range
        self.ratio = ratio
        self.uvd = self.init_uvd(image_shape=image_shape)

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
        return np.array([a, b, c, d])
    
    def parse_roll_pitch(self, lidar2cam):
        ground_points_lidar = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
        ground_points_lidar = np.concatenate((ground_points_lidar, np.ones((ground_points_lidar.shape[0], 1))), axis=1)
        ground_points_cam = np.matmul(lidar2cam, ground_points_lidar.T).T
        denorm = self.equation_plane(ground_points_cam)
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
    
    def rectify_cam_intrinsic(self, cam_intrinsic):
        ratio = np.random.uniform(1.0, self.ratio)
        cam_intrinsic_rectify = cam_intrinsic.copy()
        cam_intrinsic_rectify[:2,:2] = cam_intrinsic[:2,:2] * ratio
        return cam_intrinsic_rectify
    
    def rectify_roll_params(self, lidar2cam, roll_status):
        target_roll_status = np.random.uniform(self.roll_range[0], self.roll_range[1])
        roll = target_roll_status - roll_status
        roll_rad = self.degree2rad(roll)
        rectify_roll = np.array([[math.cos(roll_rad), -math.sin(roll_rad), 0, 0], 
                                 [math.sin(roll_rad), math.cos(roll_rad), 0, 0], 
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 1]])
        lidar2cam_rectify = np.matmul(rectify_roll, lidar2cam)
        return lidar2cam_rectify
    
    def rectify_pitch_params(self, lidar2cam, pitch_status):
        target_pitch_status = np.random.uniform(pitch_status + self.pitch_range[0], pitch_status + self.pitch_range[1])            
        pitch = -1 * (target_pitch_status - pitch_status)
        pitch_rad = self.degree2rad(pitch)
        rectify_pitch = np.array([[1, 0, 0, 0],
                                  [0,math.cos(pitch_rad), -math.sin(pitch_rad), 0], 
                                  [0,math.sin(pitch_rad), math.cos(pitch_rad), 0],
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
        roll_status, pitch_status = self.parse_roll_pitch(lidar2cam)
        lidar2cam_roll_rectify = self.rectify_roll_params(lidar2cam, roll_status)
        lidar2cam_rectify = lidar2cam_roll_rectify
        lidar2cam_pitch_rectify = self.rectify_pitch_params(lidar2cam_roll_rectify, pitch_status)            
        lidar2cam_rectify = lidar2cam_pitch_rectify
        cam_intrinsic_rectify = self.rectify_cam_intrinsic(cam_intrinsic)
        
        M = self.get_M(lidar2cam[:3,:3], cam_intrinsic[:3,:3], lidar2cam_rectify[:3,:3], cam_intrinsic_rectify[:3,:3])
        image = self.transform_with_M_bilinear(image, M)
        return image.astype(np.uint8), lidar2cam_rectify, cam_intrinsic_rectify

class ProduceHeightMap(object):
    def __init__(self, resolution=0.001):
        self.resolution = resolution
        
    def __call__(self, image, gt_bboxes, lidar2cam, cam_intrinsic):
        surface_points_list = []
        for obj_id in range(gt_bboxes.shape[0]):
            gt_bbox = gt_bboxes[obj_id]
            lwh = gt_bbox[3:6]
            # center_lidar = gt_bbox[:3] + [0.0, 0.0, 0.5 * lwh[2]]
            center_lidar = gt_bbox[:3]
            yaw_lidar = gt_bbox[6]
            surface_points = self.box3d_surface(lwh, center_lidar, yaw_lidar, lidar2cam)   
            surface_points_list.append(surface_points)
        surface_points = np.vstack(surface_points_list)   
        
        surface_points_cam = np.matmul(lidar2cam, np.concatenate((surface_points, np.ones((surface_points.shape[0],1))), axis=1).T)
        depths = surface_points_cam[2, :]
        mask = np.ones(depths.shape[0], dtype=bool)
        mask = np.logical_and(mask, depths > 3.0)
        surface_points_cam = surface_points_cam[:, mask]
        
        surface_points_img = np.matmul(cam_intrinsic, surface_points_cam).T
        surface_points_img = surface_points_img[:,:2] / (surface_points_img[:,2] + 10e-6)[:, np.newaxis]
        surface_points_img = surface_points_img.astype(np.int32)
        surface_points_img[:,0] = np.clip(surface_points_img[:,0], 0, image.shape[1]-1)
        surface_points_img[:,1] = np.clip(surface_points_img[:,1], 0, image.shape[0]-1)
        
        image[surface_points_img[:,1], surface_points_img[:,0]] = (255,0,0)
        return image
    
    def local2global(self, points, center_lidar, yaw_lidar):
        points_3d_lidar = points.reshape(-1, 3)
        rot_mat = np.array([[math.cos(yaw_lidar), -math.sin(yaw_lidar), 0], 
                            [math.sin(yaw_lidar), math.cos(yaw_lidar), 0], 
                            [0, 0, 1]])
        points_3d_lidar = np.matmul(rot_mat, points_3d_lidar.T).T + center_lidar
        return points_3d_lidar
    
    def global2cam(self, points, lidar2cam):
        points = np.concatenate((points[:, :3], np.ones((points.shape[0], 1))), axis=1)
        points = np.matmul(lidar2cam, points.T).T
        return points[:, :3]
    
    def distance(self, point):
        return np.sqrt(point[0]**2 + point[1]**2 + point[2]**2)
    
    def box3d_surface(self, lwh, center_lidar, yaw_lidar, lidar2cam):
        l, w, h = lwh[0], lwh[1], lwh[2]
        surface_points = []
        # top
        shape_top = np.array([w / self.resolution, l / self.resolution]).astype(np.int32)
        n, m = [(ss - 1.) / 2. for ss in shape_top]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        xv, yv = np.meshgrid(x, y, sparse=False)
        xyz = np.concatenate((xv[:,:,np.newaxis], yv[:,:,np.newaxis],  0.5 * np.ones_like(xv)[:,:,np.newaxis] * h / self.resolution), axis=-1)
        points_top = self.local2global(xyz * self.resolution, center_lidar, yaw_lidar)
        # left
        shape_left = np.array([h / self.resolution, l / self.resolution]).astype(np.int32)
        n, m = [(ss - 1.) / 2. for ss in shape_left]
        x, z = np.ogrid[-m:m + 1, -n:n + 1]
        xv, zv = np.meshgrid(x, z, sparse=False)    
        xyz = np.concatenate((0.5 * np.ones_like(xv)[:,:,np.newaxis] * w / self.resolution, xv[:,:,np.newaxis], zv[:,:,np.newaxis]), axis=-1)
        points_left = self.local2global(xyz * self.resolution, center_lidar, yaw_lidar)
        points_left_mean = self.local2global(np.array([0.5 * w, 0.0, 0.0]), center_lidar, yaw_lidar)
        points_left_mean = self.global2cam(points_left_mean, lidar2cam)[0]
        # right
        xyz = np.concatenate((-0.5 * np.ones_like(xv)[:,:,np.newaxis] * w / self.resolution, xv[:,:,np.newaxis], zv[:,:,np.newaxis]), axis=-1)
        points_right = self.local2global(xyz * self.resolution, center_lidar, yaw_lidar)
        points_right_mean = self.local2global(np.array([-0.5 * w, 0.0, 0.0]), center_lidar, yaw_lidar)
        points_right_mean = self.global2cam(points_right_mean, lidar2cam)[0]
        # front
        shape_front = np.array([h / self.resolution, w / self.resolution]).astype(np.int32)
        n, m = [(ss - 1.) / 2. for ss in shape_front]
        y, z = np.ogrid[-m:m + 1, -n:n + 1]
        yv, zv = np.meshgrid(y, z, sparse=False)
        xyz = np.concatenate((yv[:,:,np.newaxis], -0.5 * np.ones_like(yv)[:,:,np.newaxis] * l / self.resolution, zv[:,:,np.newaxis]), axis=-1)
        points_front = self.local2global(xyz * self.resolution, center_lidar, yaw_lidar)
        points_front_mean = self.local2global(np.array([0.0, -0.5 * l, 0.0]), center_lidar, yaw_lidar)
        points_front_mean = self.global2cam(points_front_mean, lidar2cam)[0]
        # rear
        xyz = np.concatenate((yv[:,:,np.newaxis], 0.5 * np.ones_like(yv)[:,:,np.newaxis] * l / self.resolution, zv[:,:,np.newaxis]), axis=-1)
        points_rear = self.local2global(xyz * self.resolution, center_lidar, yaw_lidar)
        points_rear_mean = self.local2global(np.array([0.0, 0.5 * l, 0.0]), center_lidar, yaw_lidar)
        points_rear_mean = self.global2cam(points_rear_mean, lidar2cam)[0]
        surface_points.append(points_top)
        if self.distance(points_left_mean) < self.distance(points_right_mean):
            surface_points.append(points_left)
        else:
            surface_points.append(points_right)
            
        if self.distance(points_front_mean) < self.distance(points_rear_mean):
            surface_points.append(points_front)
        else:
            surface_points.append(points_rear)                
        surface_points = np.vstack(surface_points)
        return surface_points
    