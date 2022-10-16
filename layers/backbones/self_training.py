import math
from termios import BS1
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops import roi_align
from mmdet3d.models.builder import NECKS

def D(p, z, version='simplified'): # negative cosine similarity
    if version == 'original':
        z = z.detach() # stop gradient
        p = F.normalize(p, dim=1) # l2-normalize 
        z = F.normalize(z, dim=1) # l2-normalize 
        return -(p*z).sum(dim=1).mean()

    elif version == 'simplified':# same thing, much faster. Scroll down, speed test in __main__
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
    else:
        raise Exception
    
def MSE(p, z, reduction="mean"):
    return F.mse_loss(p, z.detach(), reduction=reduction) 

@NECKS.register_module()
class SelfTraining(nn.Module):
    def __init__(self,                 
                 pc_range=[0, -51.2, -5, 102.4, 51.2, 3],
                 bev_h=128,
                 bev_w=128
                 ):
        super().__init__()
        self.pc_range = pc_range
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.grid_length = [self.real_h / self.bev_h, self.real_w / self.bev_w]

    def forward(self, feature_map_list, gt_boxes=None):
        feature_map, feature_map_warped = feature_map_list[0], feature_map_list[1]
        ids1 = np.arange(0, feature_map.shape[0], 2)
        src_feature_map = feature_map[ids1]
        gt_boxes = [gt_boxes[ids] for ids in ids1.tolist()] 
        
        bbox_mask = self.get_bbox_mask(gt_boxes)
        bbox_mask = torch.from_numpy(bbox_mask).to(device=feature_map.device)
        bbox_mask = bbox_mask.float()  # [B, H, W]
        
        # feature level 
        ref_feature_map_1, ref_feature_map_2 = feature_map_warped[:feature_map_warped.shape[0]//2], feature_map_warped[feature_map_warped.shape[0]//2:]
        x1, x2, x3 = src_feature_map, ref_feature_map_1, ref_feature_map_2
        x1, x2, x3 = x1.permute(0, 2, 3, 1).contiguous(), x2.permute(0, 2, 3, 1).contiguous(), x3.permute(0, 2, 3, 1).contiguous()
        x1, x2, x3 = x1 * bbox_mask.unsqueeze(-1), x2 * bbox_mask.unsqueeze(-1), x3 * bbox_mask.unsqueeze(-1)
        x1, x2, x3 = x1.view(-1, x1.shape[-1]), x2.view(-1, x2.shape[-1]), x3.view(-1, x3.shape[-1])
        loss_map = MSE(x1, x2) / 2 + MSE(x1, x3) / 2
        
        # object level
        feature_map = torch.cat([src_feature_map, feature_map_warped], dim=0)  # [bs, c, h w]
        bs = feature_map.shape[0]
        
        max_objs, num_keys = 200, 1
        bbox_locs = np.zeros((bs//3, num_keys * max_objs, 2), dtype=np.float32)
        bbox_mask = np.zeros((bs//3, max_objs), dtype=np.bool)
        for batch_id in range(len(gt_boxes)):
            gt_bbox = gt_boxes[batch_id].cpu().numpy()
            if gt_bbox.shape[0] == 0:
                continue
            for obj_id in range(gt_bbox.shape[0]):
                loc, lwh, rot_y = gt_bbox[obj_id, :3], gt_bbox[obj_id, 3:6], gt_bbox[obj_id, 6]
                corners = self.get_object_axes(lwh, loc, rot_y)
                pixels = self.point2bevpixel(corners)
                bbox_locs[batch_id, (num_keys * obj_id):(num_keys * (obj_id+1)), :] = pixels
                bbox_mask[batch_id, obj_id] = True
        bbox_mask = torch.from_numpy(bbox_mask).to(device=feature_map.device)
        bbox_locs = torch.from_numpy(bbox_locs).to(device=feature_map.device).unsqueeze(0).repeat(3, 1, 1, 1).view(bs, num_keys * max_objs, 2) # [bs, max_obj, 2]
        bbox_rois = torch.cat([bbox_locs - 2, bbox_locs + 2], dim=-1) # [bs, max_obj, 4]
        batch_id = torch.arange(bs, dtype=torch.float, device=feature_map.device).unsqueeze(1) # [bs, 1]
        batch_id = batch_id.repeat(1, bbox_rois.shape[1]).view(-1, 1) # [bs, max_obj] --> [bs * max_obj, 1]
        bbox_rois = torch.cat([batch_id, bbox_rois.view(-1, 4)], dim=-1)
                
        mask = bbox_mask.flatten()
        features_bbox_rois = roi_align(feature_map, bbox_rois, output_size=[1,1], spatial_scale=1, sampling_ratio=1)
        features_bbox_rois = features_bbox_rois.view(bs, -1, num_keys * features_bbox_rois.shape[1])
        x1, x2, x3 = features_bbox_rois[:(bs//3)], features_bbox_rois[(bs//3):(2*bs//3)], features_bbox_rois[(2*bs//3):]
        x1 = x1.view(-1, x1.shape[-1])[mask]
        x2 = x2.view(-1, x2.shape[-1])[mask]
        x3 = x3.view(-1, x3.shape[-1])[mask]
        if x1.shape[0] == 1:
            x1 = x1.repeat(2, 1)
            x2 = x2.repeat(2, 1)
            x3 = x3.repeat(2, 1)
        loss_bbox = MSE(x1, x2) / 2 + MSE(x1, x3) / 2
        return loss_bbox + loss_map
         
    def bev_voxels(self, num_voxels):
        u, v = np.ogrid[0:num_voxels[0], 0:num_voxels[1]]
        uu, vv = np.meshgrid(u, v, sparse=False)
        voxel_size = np.array([self.bev_h / num_voxels[0], self.bev_w / num_voxels[1]])
        uv = np.concatenate((uu[:,:,np.newaxis], vv[:,:,np.newaxis]), axis=-1)
        uv = uv * voxel_size + 0.5 * voxel_size
        return uv.astype(np.float32)
    
    def point2bevpixel(self, points):
        pixels_w = (points[:, 0] - self.pc_range[0]) / self.grid_length[0]  # xs
        pixels_h = (points[:, 1] - self.pc_range[1]) / self.grid_length[1]  # ys

        pixels = np.concatenate((pixels_w[:, np.newaxis], pixels_h[:, np.newaxis]), axis=-1)
        pixels = pixels.astype(np.int32)
        pixels[:, 0] = np.clip(pixels[:, 0], 0, self.bev_w-1)
        pixels[:, 1] = np.clip(pixels[:, 1], 0, self.bev_h-1)
        return pixels
    
    def get_object_corners(self, lwh, loc, rot_y):
        tr_matrix = np.zeros((2, 3)) 
        tr_matrix[:2, :2] = np.array([np.cos(rot_y), -np.sin(rot_y), np.sin(rot_y), np.cos(rot_y)]).astype(float).reshape(2,2)
        tr_matrix[:2, 2] = np.array([loc[0], loc[1]]).astype(float).reshape(1,2)
        lwh = 0.5 * lwh
        corner_points = np.array([lwh[1], lwh[0], 1.0, lwh[1], -lwh[0], 1.0, -lwh[1], lwh[0], 1.0, -lwh[1], -lwh[0], 1.0]).astype(float).reshape(4,3).T
        corner_points = np.dot(tr_matrix, corner_points).T
        return corner_points
    
    def get_object_axes(self, lwh, loc, rot_y):
        tr_matrix = np.zeros((2, 3)) 
        tr_matrix[:2, :2] = np.array([np.cos(rot_y), -np.sin(rot_y), np.sin(rot_y), np.cos(rot_y)]).astype(float).reshape(2,2)
        tr_matrix[:2, 2] = np.array([loc[0], loc[1]]).astype(float).reshape(1,2)
        lwh = 0.5 * lwh
        # corner_points = np.array([0.0, lwh[0], 1.0, 0.0, -lwh[0], 1.0, 0.0, -lwh[0], 1.0]).astype(float).reshape(3,3).T
        # corner_points = np.array([lwh[1], 0.0, 1.0, 0.0, 0.0, 1.0, -lwh[1], 0.0, 1.0]).astype(float).reshape(3,3).T
        corner_points = np.array([0.0, 0.0, 1.0]).astype(float).reshape(1,3).T
        corner_points = np.dot(tr_matrix, corner_points).T
        return corner_points
    
    def local2global(self, points, center_lidar, yaw_lidar):
        points_3d_lidar = points.reshape(-1, 3)
        rot_mat = np.array([[math.cos(yaw_lidar), -math.sin(yaw_lidar), 0], 
                            [math.sin(yaw_lidar), math.cos(yaw_lidar), 0], 
                            [0, 0, 1]])
        points_3d_lidar = np.matmul(rot_mat, points_3d_lidar.T).T + center_lidar
        return points_3d_lidar
    
    def get_bbox_mask(self, gt_boxes, resolution=0.5):
        bbox_mask = np.zeros((len(gt_boxes), self.bev_h, self.bev_w), dtype=np.bool)
        for batch_id in range(len(gt_boxes)):
            gt_bbox = gt_boxes[batch_id].cpu().numpy()
            if gt_bbox.shape[0] == 0:
                continue
            for obj_id in range(gt_bbox.shape[0]):
                loc, lwh, rot_y = gt_bbox[obj_id, :3], gt_bbox[obj_id, 3:6], gt_bbox[obj_id, 6]
                lwh = lwh * 1.5
                shape = np.array([lwh[0] / resolution, lwh[1] / resolution]).astype(np.int32)
                n, m = [(ss - 1.) / 2. for ss in shape]
                x, y = np.ogrid[-m:m + 1, -n:n + 1]
                xv, yv = np.meshgrid(x, y, sparse=False)
                xyz = np.concatenate((xv[:,:,np.newaxis], yv[:,:,np.newaxis], np.ones_like(xv)[:,:,np.newaxis]), axis=-1)
                obj_points = self.local2global(xyz * resolution, loc, rot_y)
                obj_pixels = self.point2bevpixel(obj_points)
                bbox_mask[batch_id, obj_pixels[:, 1], obj_pixels[:, 0]] = True
        return bbox_mask
