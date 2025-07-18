import numpy as np
import cv2
import torch
import torch.nn.functional as F
from core_gm.gmflow.gmflow.gmflow import GMFlow
from torch import nn

class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()
        self.batch_size = batch_size
        self.height = height
        self.width = width
        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)
        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)
        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False) # B, 3, H, W
    def forward(self, depth, inv_K):
        batch_size = depth.size(0)
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords[:batch_size])
        cam_points = depth.view(batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones[:batch_size]], 1)
        return cam_points

class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """
    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps
    def forward(self, points, K, T):
        batch_size = points.size(0)
        P = torch.matmul(K, T)[:, :3, :]
        cam_points = torch.matmul(P, points)
        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(batch_size, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        _pix_coords_ = torch.clone(pix_coords)
        _pix_coords_[..., 0] /= self.width - 1
        _pix_coords_[..., 1] /= self.height - 1
        _pix_coords_ = (_pix_coords_ - 0.5) * 2
        return _pix_coords_, pix_coords

class OpticFlowMask(nn.Module):
    def __init__(self,feature_channels = 128,num_scales = 1, upsample_factor = 8,
                 num_head = 1,attention_type = 'swin',ffn_dim_expansion = 4, num_transformer_layers = 6,
                 batch_size=12,matching_height=192,matching_width=640):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_gmflow = GMFlow(feature_channels=feature_channels,
                                   num_scales=num_scales,
                                   upsample_factor=upsample_factor,
                                   num_head=num_head,
                                   attention_type=attention_type,
                                   ffn_dim_expansion=ffn_dim_expansion,
                                   num_transformer_layers=num_transformer_layers).to(self.device)
        self.attn_splits_list = [2]
        self.corr_radius_list = [-1]
        self.prop_radius_list = [-1]
        self.frame_ids=[0,-1,1]
        address = '/home/jsw/pretrained/gmflow_sintel-0c07dcb3.pth'
        checkpoint = torch.load(address)
        weights = checkpoint['model'] if 'model' in checkpoint else checkpoint
        self.model_gmflow.load_state_dict(weights)
        self.model_gmflow.eval()
        for param in self.model_gmflow.parameters():
            param.requires_grad = False
        self.batch_size = batch_size
        self.matching_height = matching_height
        self.matching_width = matching_width
        # ^^^^^^^^^^^^^^^^^^^^^^ Modifying here ^^^^^^^^^^^^^^^^^^^^^^
        self.backprojector_batch = BackprojectDepth(batch_size=self.batch_size,
                                              height=self.matching_height,
                                              width=self.matching_width)
        self.projector_batch = Project3D(batch_size=self.batch_size,
                                   height=self.matching_height,
                                   width=self.matching_width)
    def invert_flow(self, fwd_flow, segmentation):
        # Get the backward optical flow.
        B, _, H, W = fwd_flow.shape
        grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        grid = torch.stack((grid_x, grid_y), dim=0).float().to(fwd_flow.device)  # Shape (2, H, W)
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)
        coords = grid + fwd_flow  # Shape (B, 2, H, W)
        bwd_flow = torch.zeros_like(fwd_flow)
        seg_ref = torch.zeros_like(segmentation)
        coords = torch.round(coords).long()
        grid = grid.long()
        coords[:, 0].clamp_(0, W - 1)
        coords[:, 1].clamp_(0, H - 1)
        for b in range(B):
            bwd_flow[b, :, coords[b, 1], coords[b, 0]] = - fwd_flow[b, :, grid[b,1], grid[b,0]]
            seg_ref[b, :, coords[b, 1], coords[b, 0]] = segmentation[b, :, grid[b,1], grid[b,0]]
        return bwd_flow, seg_ref
    def warping(self, lookup_images, current_image, flow_bwd, seg_ref):
        ref_image = F.interpolate(lookup_images[:, 0], scale_factor=1/4, mode='bilinear', align_corners=False)
        cur_image = F.interpolate(current_image, scale_factor=1/4, mode='bilinear', align_corners=False)
        B, _, H, W = flow_bwd.shape
        grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        grid = torch.stack((grid_x, grid_y), dim=0).float().to(flow_bwd.device)
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)
        new_coords = grid + flow_bwd
        new_coords[:, 0, :, :] = (new_coords[:, 0, :, :] / (W - 1)) * 2 - 1
        new_coords[:, 1, :, :] = (new_coords[:, 1, :, :] / (H - 1)) * 2 - 1
        new_coords = new_coords.permute(0, 2, 3, 1)  # Shape (1, H, W, 2)
        static_reference_dyn = F.grid_sample(cur_image, new_coords, mode='bilinear', padding_mode='zeros', align_corners=True)
        new_static_reference = ref_image*(~seg_ref) + static_reference_dyn*seg_ref
        return new_static_reference
    def normalize_check_flow(self, flow):
        min_val = torch.min(flow)
        max_val = torch.max(flow)
        normalized_flow = (flow - min_val) / (max_val - min_val)
        return normalized_flow

    def normalize_flow(self, flow):
        min_val = torch.min(flow)
        max_val = torch.max(flow)
        normalized_flow = 2 * (flow - min_val) / (max_val - min_val) - 1
        return normalized_flow

    def compute_dynamic_flow(self, lookup_pose, _flow, depth, _K, _invK, current_image, lookup_images,epoch):
        flow = _flow[:depth.size(0)]
        # flow=flow/10
        flow=self.normalize_flow(flow)
        world_points_depth = self.backprojector_batch(depth, _invK)
        _, pix_locs_depth = self.projector_batch(world_points_depth, _K, lookup_pose)
        pix_locs_depth = pix_locs_depth.permute(0, 3, 1, 2)
        # --------------- backprojector_batch
        pix_coords = self.backprojector_batch.pix_coords.view(self.batch_size, 3, \
                                                              self.matching_height, self.matching_width)[:, :2, :, :]
        pix_coords = pix_coords[:flow.size(0)]
        normal_static_flow = (pix_locs_depth - pix_coords)
        normal_static_flow = self.normalize_flow(normal_static_flow )
        dynamic_flow = flow - normal_static_flow
        check_flow = torch.norm(dynamic_flow, dim=1, keepdim=True)
        check_flow = self.normalize_check_flow(check_flow)
        threshold = 0.98
        motion_mask = check_flow < threshold
        # flow_bwd, seg_ref = self.invert_flow(normal_static_flow, segmentation)
        # static_reference = self.warping(lookup_images, current_image, flow_bwd, seg_ref)
        return motion_mask

    def predict_gmflow(self, inputs, features=None):
        outputs = {}
        pose_feats = {f_i: inputs["color", f_i, 0] for f_i in self.frame_ids}
        for f_i in [-1, 1]:
            # ----------------- FOWARD FLOW -----------------
            results_dict = self.model_gmflow(pose_feats[0] * 255., pose_feats[f_i] * 255.,
                                             attn_splits_list=self.attn_splits_list,
                                             corr_radius_list=self.corr_radius_list,
                                             prop_radius_list=self.prop_radius_list,
                                             pred_bidir_flow=True)
            flow_preds = results_dict['flow_preds'][-1]
            outputs[("flow", f_i)] = flow_preds
        return outputs

    def forward(self, inputs, outputs, epoch):
        motion_masks = {}
        with torch.no_grad():
            # if epoch > 5:
            #     depth=pseudo_depth
            # else:
                depth= outputs["depth", 0, 0]
                flow_pred = self.predict_gmflow(inputs, None)
                outputs.update(flow_pred)
                # this one is usually set as half of freeze epoch
                for fi in [-1, 1]:
                    scale = 0
                    poses = outputs["cam_T_cam", 0, fi]
                    flow = outputs["flow", fi]
                    # depth = outputs["depth", 0, scale]
                    K = inputs["K", 0]
                    invK = inputs["inv_K", 0]
                    current_image = inputs["color", fi, 0]
                    lookup_images = inputs["color", 0, 0]
                    motion_masks[("motion_mask", fi, scale)]  = \
                        self.compute_dynamic_flow(poses, flow, depth, K, invK, current_image, lookup_images,epoch)

        return motion_masks
