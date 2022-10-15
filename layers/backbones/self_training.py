import math
from termios import BS1
import cv2
import numpy as np

from einops import rearrange, repeat
from torch import einsum

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

def compute_imitation_loss(feature_pred, feature_target, weights, reduction="mean"):
    # input:  [B, C, H, W]
    # target: [B, C, H, W]
    # weight: [B, H, W]
    feature_target = feature_target.detach()
    diff = feature_pred - feature_target
    loss = 0.5 * diff ** 2
    loss = loss * weights.unsqueeze(1).repeat(1, feature_pred.shape[1], 1, 1)
    if reduction == "mean":
        return (loss.sum() / weights.sum())
    else:
        return loss
    
def compute_resize_affinity_loss(feature_pred, feature_target):
    feature_target = feature_target.detach()
    B, C, H, W = feature_pred.shape
    
    resize_shape = [feature_target.shape[-2] // 16, feature_target.shape[-1] // 16]
    feature_pred_down = F.interpolate(feature_pred, size=resize_shape, mode="bilinear")
    feature_target_down = F.interpolate(feature_target, size=resize_shape, mode="bilinear")
    
    feature_target_down = feature_target_down.reshape(B, C, -1)
    depth_affinity = torch.bmm(feature_target_down.permute(0, 2, 1), feature_target_down)
    feature_pred_down = feature_pred_down.reshape(B, C, -1)
    rgb_affinity = torch.bmm(feature_pred_down.permute(0, 2, 1), feature_pred_down)
    
    loss = F.l1_loss(rgb_affinity, depth_affinity, reduction='mean') / B
    return loss

def compute_local_affinity_loss(feature_pred, feature_target):
    local_shape = [feature_target.shape[-2] // 16, feature_target.shape[-1] // 16]
    feature_target = feature_target.detach()
    B, _, H, W = feature_pred.shape
    
    feature_pred_q = rearrange(feature_pred, 'b c (h p1) (w p2) -> (b h w) (p1 p2) c', p1=local_shape[0], p2=local_shape[1])
    feature_pred_k = rearrange(feature_pred, 'b c (h p1) (w p2) -> (b h w) (p1 p2) c', p1=local_shape[0], p2=local_shape[1])
    rgb_affinity = einsum('b i d, b j d -> b i j', feature_pred_q, feature_pred_k)
    feature_target_q = rearrange(feature_target, 'b c (h p1) (w p2) -> (b h w) (p1 p2) c', p1=local_shape[0], p2=local_shape[1])
    feature_target_k = rearrange(feature_target, 'b c (h p1) (w p2) -> (b h w) (p1 p2) c', p1=local_shape[0], p2=local_shape[1])
    depth_affinity = einsum('b i d, b j d -> b i j', feature_target_q, feature_target_k)

    loss = F.l1_loss(rgb_affinity, depth_affinity, reduction='mean') / B
    return loss

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
        weights = bbox_mask.float()

        # feature level
        ref_feature_map_1, ref_feature_map_2 = feature_map_warped[:feature_map_warped.shape[0]//2], feature_map_warped[feature_map_warped.shape[0]//2:]
        x1, x2, x3 = src_feature_map, ref_feature_map_1, ref_feature_map_2
        feature_imitation_loss = compute_imitation_loss(x2, x1, weights) / 2 + compute_imitation_loss(x3, x1, weights) / 2
        feature_affinity_loss = compute_resize_affinity_loss(x2, x1) / 2 + compute_resize_affinity_loss(x3, x1) / 2
        print(feature_affinity_loss + feature_imitation_loss)
        return feature_imitation_loss + feature_affinity_loss
    
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
                shape = np.array([lwh[0] / resolution, lwh[1] / resolution]).astype(np.int32)
                n, m = [(ss - 1.) / 2. for ss in shape]
                x, y = np.ogrid[-m:m + 1, -n:n + 1]
                xv, yv = np.meshgrid(x, y, sparse=False)
                xyz = np.concatenate((xv[:,:,np.newaxis], yv[:,:,np.newaxis], np.ones_like(xv)[:,:,np.newaxis]), axis=-1)
                obj_points = self.local2global(xyz * resolution, loc, rot_y)
                obj_pixels = self.point2bevpixel(obj_points)
                bbox_mask[batch_id, obj_pixels[:, 1], obj_pixels[:, 0]] = True
        return bbox_mask
