# Copyright (c) Megvii Inc. All rights reserved.
import math
import numpy as np

import torch
import torch.nn.functional as F

from mmcv.cnn import build_conv_layer
from mmdet3d.models import build_neck
from mmdet.models import build_backbone
from mmdet.models.backbones.resnet import BasicBlock
from mmcv.cnn.bricks.transformer import build_positional_encoding
from torch import nn
from layers.backbones.temporal_self_attention import TemporalSelfAttention

from ops.voxel_pooling import voxel_pooling

__all__ = ['LSSFPN']


class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation,
                 BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes,
                                     planes,
                                     kernel_size=kernel_size,
                                     stride=1,
                                     padding=padding,
                                     dilation=dilation,
                                     bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):
    def __init__(self, inplanes, mid_channels=256, BatchNorm=nn.BatchNorm2d):
        super(ASPP, self).__init__()

        dilations = [1, 6, 12, 18]

        self.aspp1 = _ASPPModule(inplanes,
                                 mid_channels,
                                 1,
                                 padding=0,
                                 dilation=dilations[0],
                                 BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[1],
                                 dilation=dilations[1],
                                 BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[2],
                                 dilation=dilations[2],
                                 BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[3],
                                 dilation=dilations[3],
                                 BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, mid_channels, 1, stride=1, bias=False),
            BatchNorm(mid_channels),
            nn.ReLU(),
        )
        self.conv1 = nn.Conv2d(int(mid_channels * 5),
                               mid_channels,
                               1,
                               bias=False)
        self.bn1 = BatchNorm(mid_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5,
                           size=x4.size()[2:],
                           mode='bilinear',
                           align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.ReLU,
                 drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class SELayer(nn.Module):
    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.conv_reduce = nn.Conv2d(channels, channels, 1, bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2d(channels, channels, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


class DepthNet(nn.Module):
    def __init__(self, in_channels, mid_channels, context_channels,
                 depth_channels, height_channels):
        super(DepthNet, self).__init__()
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.context_conv = nn.Conv2d(mid_channels,
                                      context_channels,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0)
        self.bn = nn.BatchNorm1d(27)
        self.depth_mlp = Mlp(27, mid_channels, mid_channels)
        self.depth_se = SELayer(mid_channels)  # NOTE: add camera-aware
        self.context_mlp = Mlp(27, mid_channels, mid_channels)
        self.context_se = SELayer(mid_channels)  # NOTE: add camera-aware
        self.depth_conv = nn.Sequential(
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            ASPP(mid_channels, mid_channels),
            build_conv_layer(cfg=dict(
                type='DCN',
                in_channels=mid_channels,
                out_channels=mid_channels,
                kernel_size=3,
                padding=1,
                groups=4,
                im2col_step=128,
            )),
            nn.Conv2d(mid_channels,
                      depth_channels+height_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0),
        )

    def forward(self, x, mats_dict):
        intrins = mats_dict['intrin_mats'][:, 0:1, ..., :3, :3]
        batch_size = intrins.shape[0]
        num_cams = intrins.shape[2]
        ida = mats_dict['ida_mats'][:, 0:1, ...]
        sensor2ego = mats_dict['sensor2ego_mats'][:, 0:1, ..., :3, :]
        bda = mats_dict['bda_mat'].view(batch_size, 1, 1, 4,
                                        4).repeat(1, 1, num_cams, 1, 1)
        mlp_input = torch.cat(
            [
                torch.stack(
                    [
                        intrins[:, 0:1, ..., 0, 0],
                        intrins[:, 0:1, ..., 1, 1],
                        intrins[:, 0:1, ..., 0, 2],
                        intrins[:, 0:1, ..., 1, 2],
                        ida[:, 0:1, ..., 0, 0],
                        ida[:, 0:1, ..., 0, 1],
                        ida[:, 0:1, ..., 0, 3],
                        ida[:, 0:1, ..., 1, 0],
                        ida[:, 0:1, ..., 1, 1],
                        ida[:, 0:1, ..., 1, 3],
                        bda[:, 0:1, ..., 0, 0],
                        bda[:, 0:1, ..., 0, 1],
                        bda[:, 0:1, ..., 1, 0],
                        bda[:, 0:1, ..., 1, 1],
                        bda[:, 0:1, ..., 2, 2],
                    ],
                    dim=-1,
                ),
                sensor2ego.view(batch_size, 1, num_cams, -1),
            ],
            -1,
        )
        mlp_input = self.bn(mlp_input.reshape(-1, mlp_input.shape[-1]))
        x = self.reduce_conv(x)
        context_se = self.context_mlp(mlp_input)[..., None, None]
        context = self.context_se(x, context_se)
        context = self.context_conv(context)
        depth_se = self.depth_mlp(mlp_input)[..., None, None]
        depth = self.depth_se(x, depth_se)
        depth = self.depth_conv(depth)
        return torch.cat([depth, context], dim=1)


class LSSFPN(nn.Module):
    def __init__(self, x_bound, y_bound, z_bound, d_bound, h_bound, final_dim,
                 downsample_factor, output_channels, img_backbone_conf,
                 img_neck_conf, depth_net_conf):
        """Modified from `https://github.com/nv-tlabs/lift-splat-shoot`.

        Args:
            x_bound (list): Boundaries for x.
            y_bound (list): Boundaries for y.
            z_bound (list): Boundaries for z.
            d_bound (list): Boundaries for d.
            final_dim (list): Dimension for input images.
            downsample_factor (int): Downsample factor between feature map
                and input image.
            output_channels (int): Number of channels for the output
                feature map.
            img_backbone_conf (dict): Config for image backbone.
            img_neck_conf (dict): Config for image neck.
            depth_net_conf (dict): Config for depth net.
        """

        super(LSSFPN, self).__init__()
        self.downsample_factor = downsample_factor
        self.d_bound = d_bound
        self.h_bound = h_bound
        self.final_dim = final_dim
        self.output_channels = output_channels
        self.is_fusion = True

        if self.is_fusion:
            self.depth_fusion = TemporalSelfAttention(embed_dims=output_channels, num_heads=8, num_levels=1)
            positional_encoding=dict(type='SinePositionalEncoding', num_feats=40, normalize=True)
            self.positional_encoding = build_positional_encoding(positional_encoding)

            self.cross_attn = nn.MultiheadAttention(output_channels, 8, dropout=0.01, batch_first=True)
            self.dropout = nn.Dropout(0.01)
            self.norm_depth = nn.LayerNorm(output_channels)

        self.register_buffer(
            'voxel_size',
            torch.Tensor([row[2] for row in [x_bound, y_bound, z_bound]]))
        self.register_buffer(
            'voxel_coord',
            torch.Tensor([
                row[0] + row[2] / 2.0 for row in [x_bound, y_bound, z_bound]
            ]))
        self.register_buffer(
            'voxel_num',
            torch.LongTensor([(row[1] - row[0]) / row[2]
                              for row in [x_bound, y_bound, z_bound]]))
        
        self.register_buffer('frustum_depth', self.create_frustum(is_depth=True))
        self.register_buffer('frustum_height', self.create_frustum(is_depth=False))
        self.depth_channels, _, _, _ = self.frustum_depth.shape
        self.height_channels, _, _, _ = self.frustum_height.shape

        self.img_backbone = build_backbone(img_backbone_conf)
        self.img_neck = build_neck(img_neck_conf)
        self.depth_net = self._configure_depth_net(depth_net_conf)

        self.img_neck.init_weights()
        self.img_backbone.init_weights()

    def _configure_depth_net(self, depth_net_conf):
        return DepthNet(
            depth_net_conf['in_channels'],
            depth_net_conf['mid_channels'],
            self.output_channels,
            self.depth_channels,
            self.height_channels,
        )
    
    @staticmethod
    def get_reference_points(H, W, Z=8, num_points_in_pillar=4, dim='3d', bs=1, device='cuda', dtype=torch.float):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """
        # reference points in 3D space, used in spatial cross-attention (SCA)
        if dim == '3d':
            zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,
                                device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
            xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                                device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
            ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                                device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
            ref_3d = torch.stack((xs, ys, zs), -1)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
            return ref_3d
        # reference points on 2D bev plane, used in temporal self-attention (TSA).
        elif dim == '2d':
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=dtype, device=device)
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d

    def create_frustum(self, is_depth=True):
        """Generate frustum"""
        # make grid in image plane
        ogfH, ogfW = self.final_dim
        fH, fW = ogfH // self.downsample_factor, ogfW // self.downsample_factor
        # depth
        if is_depth:
            d_coords = torch.arange(*self.d_bound,
                                    dtype=torch.float).view(-1, 1,
                                                            1).expand(-1, fH, fW)
        else:
            # DID
            alpha = 1.0
            num_bins = self.h_bound[2]
            hmean, hlen = (self.h_bound[0] + self.h_bound[1]) / 2.0, (self.h_bound[1] - self.h_bound[0]) / 2.0
            d_coords = np.arange(-1 * num_bins//2, num_bins//2, 1) / (num_bins//2)    
            flag = np.sign(d_coords)
            d_coords = np.power(np.abs(d_coords), alpha) * flag
            d_coords = d_coords * hlen + hmean
            d_coords = torch.tensor(d_coords, dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)

        # height LID style
        '''
        min_height, min_num = 0.5, 40
        lid_num = self.d_bound[2] - 2 * min_num
        range_num1 = int(lid_num * abs(self.d_bound[0]) / (self.d_bound[1] - self.d_bound[0]))
        range_num2 = lid_num - range_num1

        delta = 2 * (abs(self.d_bound[0]) - min_height) / (range_num1 * (1 + range_num1))
        d_coords1 = ((np.arange(range_num1) + 0.5) * 2)**2
        d_coords1 = (d_coords1 - 1) * delta / 8 + min_height
        d_coords1 = -1 * np.flipud(d_coords1)

        delta = 2 * (abs(self.d_bound[1]) - min_height) / (range_num2 * (1 + range_num2))
        d_coords2 = ((np.arange(range_num2) + 0.5) * 2)**2
        d_coords2 = (d_coords2 - 1) * delta / 8 + min_height

        mid_coords1 = (np.arange(min_num) * min_height / min_num)
        mid_coords1 = -1 * np.flipud(mid_coords1)
        mid_coords2 = (np.arange(min_num) * min_height / min_num)
        mid_coords1[-1], mid_coords2[0] = mid_coords1[-2] / 2, mid_coords2[1] / 2
        d_coords = np.concatenate([d_coords1, mid_coords1, mid_coords2, d_coords2], axis=0)
        d_coords = torch.tensor(d_coords, dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        '''
        D, _, _ = d_coords.shape
        x_coords = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(
            1, 1, fW).expand(D, fH, fW)
        y_coords = torch.linspace(0, ogfH - 1, fH,
                                  dtype=torch.float).view(1, fH,
                                                          1).expand(D, fH, fW)
        paddings = torch.ones_like(d_coords)

        # D x H x W x 3
        frustum = torch.stack((x_coords, y_coords, d_coords, paddings), -1)
        return frustum
    
    def height2localtion(self, points, sensor2ego_mat, sensor2virtual_mat, intrin_mat, reference_heights):
        batch_size, num_cams, _, _ = sensor2ego_mat.shape
        reference_heights = reference_heights.view(batch_size, num_cams, 1, 1, 1, 1,
                                                   1).repeat(1, 1, points.shape[2], points.shape[3], points.shape[4], 1, 1)
        height = -1 * points[:, :, :, :, :, 2, :] + reference_heights[:, :, :, :, :, 0, :]
        points_const = points.clone()
        points_const[:, :, :, :, :, 2, :] = 10
        points_const = torch.cat(
            (points_const[:, :, :, :, :, :2] * points_const[:, :, :, :, :, 2:3],
             points_const[:, :, :, :, :, 2:]), 5)
        combine_virtual = sensor2virtual_mat.matmul(torch.inverse(intrin_mat))
        points_virtual = combine_virtual.view(batch_size, num_cams, 1, 1, 1, 4, 4).matmul(points_const)
        ratio = height[:, :, :, :, :, 0] / points_virtual[:, :, :, :, :, 1, 0]
        ratio = ratio.view(batch_size, num_cams, ratio.shape[2], ratio.shape[3], ratio.shape[4], 1, 1).repeat(1, 1, 1, 1, 1, 4, 1)
        ratio = torch.maximum(ratio, ratio.new_zeros(ratio.shape[0], ratio.shape[1], ratio.shape[2], ratio.shape[3], ratio.shape[4], ratio.shape[5], ratio.shape[6]))
        points = points_virtual * ratio
        points[:, :, :, :, :, 3, :] = 1
        combine_ego = sensor2ego_mat.matmul(torch.inverse(sensor2virtual_mat))
        points = combine_ego.view(batch_size, num_cams, 1, 1, 1, 4,
                              4).matmul(points)
        return points
        
    def depth2location(self, points, sensor2ego_mat, intrin_mat):
        batch_size, num_cams, _, _ = sensor2ego_mat.shape
        points = torch.cat(
            (points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
             points[:, :, :, :, :, 2:]), 5)
        combine = sensor2ego_mat.matmul(torch.inverse(intrin_mat))
        points = combine.view(batch_size, num_cams, 1, 1, 1, 4,
                              4).matmul(points)
        return points
    
    def get_geometry(self, sensor2ego_mat, sensor2virtual_mat, intrin_mat, ida_mat, reference_heights, bda_mat, is_depth=True):
        """Transfer points from camera coord to ego coord.
        Args:
            rots(Tensor): Rotation matrix from camera to ego.
            trans(Tensor): Translation matrix from camera to ego.
            intrins(Tensor): Intrinsic matrix.
            post_rots_ida(Tensor): Rotation matrix for ida.
            post_trans_ida(Tensor): Translation matrix for ida
            post_rot_bda(Tensor): Rotation matrix for bda.

        Returns:
            Tensors: points ego coord.
        """
        batch_size, num_cams, _, _ = sensor2ego_mat.shape
        ida_mat = ida_mat.view(batch_size, num_cams, 1, 1, 1, 4, 4)
        # undo post-transformation
        # B x N x D x H x W x 3\
        if is_depth:
            points = self.frustum_depth
            points = ida_mat.inverse().matmul(points.unsqueeze(-1))
            points = self.depth2location(points, sensor2ego_mat, intrin_mat)
        else:
            points = self.frustum_height
            points = ida_mat.inverse().matmul(points.unsqueeze(-1))
            points = self.height2localtion(points, sensor2ego_mat, sensor2virtual_mat, intrin_mat, reference_heights)
        
        if bda_mat is not None:
            bda_mat = bda_mat.unsqueeze(1).repeat(1, num_cams, 1, 1).view(
                batch_size, num_cams, 1, 1, 1, 4, 4)
            points = (bda_mat @ points).squeeze(-1)
        else:
            points = points.squeeze(-1)
        return points[..., :3]

    def get_cam_feats(self, imgs):
        """Get feature maps from images."""
        batch_size, num_sweeps, num_cams, num_channels, imH, imW = imgs.shape

        imgs = imgs.flatten().view(batch_size * num_sweeps * num_cams,
                                   num_channels, imH, imW)
        img_feats = self.img_neck(self.img_backbone(imgs))[0]
        img_feats = img_feats.reshape(batch_size, num_sweeps, num_cams,
                                      img_feats.shape[1], img_feats.shape[2],
                                      img_feats.shape[3])
        return img_feats

    def _forward_depth_net(self, feat, mats_dict):
        return self.depth_net(feat, mats_dict)

    def _forward_voxel_net(self, img_feat_with_depth):
        return img_feat_with_depth

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def _forward_single_sweep(self,
                              sweep_index,
                              sweep_imgs,
                              mats_dict,
                              is_return_depth=False):
        """Forward function for single sweep.

        Args:
            sweep_index (int): Index of sweeps.
            sweep_imgs (Tensor): Input images.
            mats_dict (dict):
                sensor2ego_mats(Tensor): Transformation matrix from
                    camera to ego with shape of (B, num_sweeps,
                    num_cameras, 4, 4).
                intrin_mats(Tensor): Intrinsic matrix with shape
                    of (B, num_sweeps, num_cameras, 4, 4).
                ida_mats(Tensor): Transformation matrix for ida with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                sensor2sensor_mats(Tensor): Transformation matrix
                    from key frame camera to sweep frame camera with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                bda_mat(Tensor): Rotation matrix for bda with shape
                    of (B, 4, 4).
            is_return_depth (bool, optional): Whether to return depth.
                Default: False.

        Returns:
            Tensor: BEV feature map.
        """
        batch_size, num_sweeps, num_cams, num_channels, img_height, \
            img_width = sweep_imgs.shape
        img_feats = self.get_cam_feats(sweep_imgs)
        source_features = img_feats[:, 0, ...]
        depth_feature = self._forward_depth_net(
            source_features.reshape(batch_size * num_cams,
                                    source_features.shape[2],
                                    source_features.shape[3],
                                    source_features.shape[4]),
            mats_dict,
        )
        geo_channels = self.depth_channels + self.height_channels
        depth = depth_feature[:, :self.depth_channels].softmax(1)
        height = depth_feature[:, self.depth_channels:geo_channels].softmax(1)

        img_feat_with_depth = depth.unsqueeze(
            1) * depth_feature[:, geo_channels:(
                geo_channels + self.output_channels)].unsqueeze(2)
        img_feat_with_depth = self._forward_voxel_net(img_feat_with_depth)
        img_feat_with_depth = img_feat_with_depth.reshape(
            batch_size,
            num_cams,
            img_feat_with_depth.shape[1],
            img_feat_with_depth.shape[2],
            img_feat_with_depth.shape[3],
            img_feat_with_depth.shape[4],
        )

        img_feat_with_height = height.unsqueeze(
            1) * depth_feature[:, geo_channels:(
                geo_channels + self.output_channels)].unsqueeze(2)
        img_feat_with_height = self._forward_voxel_net(img_feat_with_height)
        img_feat_with_height = img_feat_with_height.reshape(
            batch_size,
            num_cams,
            img_feat_with_height.shape[1],
            img_feat_with_height.shape[2],
            img_feat_with_height.shape[3],
            img_feat_with_height.shape[4],
        )

        geom_xyz_depth = self.get_geometry(
            mats_dict['sensor2ego_mats'][:, sweep_index, ...],
            mats_dict['sensor2virtual_mats'][:, sweep_index, ...],
            mats_dict['intrin_mats'][:, sweep_index, ...],
            mats_dict['ida_mats'][:, sweep_index, ...],
            mats_dict['reference_heights'][:, sweep_index, ...],
            mats_dict.get('bda_mat', None),
            is_depth=True,
        )
        geom_xyz_height = self.get_geometry(
            mats_dict['sensor2ego_mats'][:, sweep_index, ...],
            mats_dict['sensor2virtual_mats'][:, sweep_index, ...],
            mats_dict['intrin_mats'][:, sweep_index, ...],
            mats_dict['ida_mats'][:, sweep_index, ...],
            mats_dict['reference_heights'][:, sweep_index, ...],
            mats_dict.get('bda_mat', None),
            is_depth=False
        )

        img_feat_depth = img_feat_with_depth.permute(0, 1, 4, 5, 3, 2).contiguous()
        img_feat_height = img_feat_with_height.permute(0, 1, 4, 5, 3, 2).contiguous()
        f_h, f_w = img_feat_depth.shape[2], img_feat_depth.shape[3]
        img_feat_depth = img_feat_depth.view(-1, img_feat_depth.shape[-2], img_feat_depth.shape[-1])
        img_feat_height = img_feat_height.view(-1, img_feat_height.shape[-2], img_feat_height.shape[-1])
        # q = k = self.with_pos_embed(tgt, query_pos)
        q = img_feat_depth
        k = v = img_feat_height
        output, _ = self.cross_attn(q, k, v)
        output = img_feat_depth + self.dropout(output)        
        output = self.norm_depth(output)
        output = output.view(batch_size, num_cams, f_h, f_w, img_feat_depth.shape[-2], img_feat_depth.shape[-1])
        img_feat_with_depth = output.permute(0, 1, 4, 2, 3, 5)

        # img_feat_with_depth = img_feat_with_depth.permute(0, 1, 3, 4, 5, 2)
        # img_feat_with_height = img_feat_with_height.permute(0, 1, 3, 4, 5, 2)

        geom_xyz_depth = ((geom_xyz_depth - (self.voxel_coord - self.voxel_size / 2.0)) /
                    self.voxel_size).int()
        geom_xyz_height = ((geom_xyz_height - (self.voxel_coord - self.voxel_size / 2.0)) /
                    self.voxel_size).int()
        feature_map_depth = voxel_pooling(geom_xyz_depth, img_feat_with_depth.contiguous(),
                                    self.voxel_num.cuda())
        '''
        feature_map_height = voxel_pooling(geom_xyz_height, img_feat_with_height.contiguous(),
                                    self.voxel_num.cuda())
        '''
        depth_pred = depth.view(batch_size, num_cams, depth.shape[1], depth.shape[2], depth.shape[3]).permute(0,1,3,4,2)
        height_pred = height.view(batch_size, num_cams, height.shape[1], height.shape[2], height.shape[3]).permute(0,1,3,4,2)
        heigth_template = geom_xyz_height[:,:,:,:,:,2].permute(0,1,3,4,2)
        depth_template = geom_xyz_depth[:,:,:,:,:,2].permute(0,1,3,4,2)
        depth_pred = torch.sum(depth_pred * depth_template, dim=-1)
        height_pred = torch.sum(height_pred * heigth_template, dim=-1)

        if self.is_fusion and False:
            device, dtype = feature_map_depth.device, feature_map_depth.dtype
            channels, bev_h, bev_w = feature_map_depth.shape[1], feature_map_depth.shape[2], feature_map_depth.shape[3]
            bev_mask = torch.zeros((batch_size, bev_h, bev_w), device=device).to(dtype)
            bev_pos = self.positional_encoding(bev_mask).to(dtype)
            bev_pos = bev_pos.permute(0, 2, 3, 1).reshape(batch_size, -1, channels).contiguous()
            depth_embed = feature_map_depth.permute(0, 2, 3, 1).reshape(batch_size, -1, channels).contiguous()
            height_embed = feature_map_height.permute(0, 2, 3, 1).reshape(batch_size, -1, channels).contiguous()
            ref_2d = self.get_reference_points(
                bev_h, bev_w, dim='2d', bs=batch_size, device=device, dtype=dtype)
            
            _, len_bev, _, _ = ref_2d.shape
            shift_ref_2d = ref_2d
            hybird_ref_2d = torch.stack([shift_ref_2d, ref_2d], 1).reshape(
                    batch_size*2, len_bev, 1, 2)
            
            output = self.depth_fusion(depth_embed, height_embed, height_embed, 
                                       query_pos=bev_pos, 
                                       key_pos=bev_pos,
                                       reference_points=hybird_ref_2d,
                                       spatial_shapes=torch.tensor(
                                            [[bev_h, bev_w]], device=depth_embed.device),
                                       level_start_index=torch.tensor([0], device=depth_embed.device)
            )
            
            feature_map = output.permute(0, 2, 1).contiguous().view(batch_size, channels, bev_h, bev_w)
        else:
            feature_map = feature_map_depth            
        if is_return_depth:
            return [feature_map.contiguous(), depth_pred, height_pred], depth
        return [feature_map.contiguous(), depth_pred, height_pred]

    def forward(self,
                sweep_imgs,
                mats_dict,
                timestamps=None,
                is_return_depth=False):
        """Forward function.

        Args:
            sweep_imgs(Tensor): Input images with shape of (B, num_sweeps,
                num_cameras, 3, H, W).
            mats_dict(dict):
                sensor2ego_mats(Tensor): Transformation matrix from
                    camera to ego with shape of (B, num_sweeps,
                    num_cameras, 4, 4).
                intrin_mats(Tensor): Intrinsic matrix with shape
                    of (B, num_sweeps, num_cameras, 4, 4).
                ida_mats(Tensor): Transformation matrix for ida with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                sensor2sensor_mats(Tensor): Transformation matrix
                    from key frame camera to sweep frame camera with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                bda_mat(Tensor): Rotation matrix for bda with shape
                    of (B, 4, 4).
            timestamps(Tensor): Timestamp for all images with the shape of(B,
                num_sweeps, num_cameras).

        Return:
            Tensor: bev feature map.
        """
        batch_size, num_sweeps, num_cams, num_channels, img_height, \
            img_width = sweep_imgs.shape

        key_frame_res = self._forward_single_sweep(
            0,
            sweep_imgs[:, 0:1, ...],
            mats_dict,
            is_return_depth=is_return_depth)
        if num_sweeps == 1:
            return key_frame_res

        key_frame_feature = key_frame_res[
            0] if is_return_depth else key_frame_res

        ret_feature_list = [key_frame_feature]
        for sweep_index in range(1, num_sweeps):
            with torch.no_grad():
                feature_map = self._forward_single_sweep(
                    sweep_index,
                    sweep_imgs[:, sweep_index:sweep_index + 1, ...],
                    mats_dict,
                    is_return_depth=False)
                ret_feature_list.append(feature_map)

        if is_return_depth:
            return torch.cat(ret_feature_list, 1), key_frame_res[1]
        else:
            return torch.cat(ret_feature_list, 1)
