# %%
import torch
import torch.nn as nn
import numpy as np
import sys
from typing import List
sys.path.append('/oliver/SAM2')
import pdb
from PIL import Image
from torchvision.transforms import ToTensor

from sam2.build_sam import build_sam2
from sam2.utils.transforms import SAM2Transforms
import torch.nn.functional as F

DEVICE = "cuda:1"

def build_all_layer_point_grids(
    n_per_side: int, n_layers: int, scale_per_layer: int
) -> List[np.ndarray]:
    """Generates point grids for all crop layers."""
    points_by_layer = []
    for i in range(n_layers + 1):
        n_points = int(n_per_side / (scale_per_layer**i))
        points_by_layer.append(build_point_grid(n_points))
    return points_by_layer

def build_point_grid(n_per_side: int) -> np.ndarray:
    """Generates a 2D grid of points evenly spaced in [0,1]x[0,1]."""
    offset = 1 / (2 * n_per_side)
    points_one_side = np.linspace(offset, 1 - offset, n_per_side)
    points_x = np.tile(points_one_side[None, :], (n_per_side, 1))
    points_y = np.tile(points_one_side[:, None], (1, n_per_side))
    points = np.stack([points_x, points_y], axis=-1).reshape(-1, 2)
    return points

class SAM2_MODEL(nn.Module):
    def __init__(self, model_cfg, ckpt_path, device):
        super(SAM2_MODEL, self).__init__()
        self.model = build_sam2(model_cfg, ckpt_path, device=device)
        self.device = device
        self._transforms = SAM2Transforms(
            resolution=1024,
            mask_threshold=0.0,
            max_hole_area=0.0,
            max_sprinkle_area=0.0,
        )
        self.se, self.de = self.create_point_embeddings()
        # self._bb_feat_sizes = [
        #     (256, 256),
        #     (128, 128),
        #     (64, 64),
        # ]
        self.image_size = 1024

    def forward(self, x, memory: tuple = (None, None)):
        """
        Args:
            x: image tensor
            memory: tuple of (maskmem_features, maskmem_pos_enc)
        """
        backbone_out = self.model.forward_image(x)
        _, vision_feats, vision_pos_embeds, feat_sizes = self.model._prepare_backbone_features(backbone_out)
        B, C, H, W = self.get_embedding_dims(vision_feats, feat_sizes)
        pixel_features = None
        if all(memory) is not None:
            pix_feat_with_mem = self.model.memory_attention(
                curr=vision_feats[-1:],
                curr_pos=vision_pos_embeds[-1:],
                memory=memory[0],
                memory_pos=memory[1],
                num_obj_ptr_tokens=0
            )
            pixel_features = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
        else:
            vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed
            feats = [
                feat.permute(1, 2, 0).view(1, -1, *feat_size)
                for feat, feat_size in zip(vision_feats[::-1], feat_sizes[::-1])
            ]
            pixel_features = feats[-1][-1].unsqueeze(0)
        if len(vision_feats) > 1:
            high_res_features = [
                feat_level[-1].unsqueeze(0)
                for feat_level in feats[:-1]
            ]
        else:
            high_res_features = None
        
        pred, _, _, _ = self.model.sam_mask_decoder(
            image_embeddings=pixel_features,
            image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=self.se,
            dense_prompt_embeddings=self.de,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res_features,
        )
        # create memory
        high_res_masks = F.interpolate(
            pred,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        
        mem = self.model.memory_encoder(
            pix_feat=pixel_features,
            masks=high_res_masks,
            skip_mask_sigmoid=False,
        )
        maskmem_features = mem.get('vision_features')
        maskmem_pos_enc = mem.get('vision_pos_enc')
        memory = maskmem_features.flatten(2).permute(2, 0, 1)
        memory_pos_embed = maskmem_pos_enc[-1].flatten(2).permute(2, 0, 1) + self.model.maskmem_tpos_enc[0]
        return pred, (memory, memory_pos_embed)
        
    
    def get_embedding_dims(self, vision_feats, feat_sizes):
        B = vision_feats[-1].size(1)
        C = self.model.memory_attention.d_model
        H, W = feat_sizes[-1]
        return B, C, H, W

    def forwarddddd(self, x, memory=None):
        # Encode the current image
        backbone_out = self.model.forward_image(x)
        backbone_features, vision_feats, vision_pos_embeds, feat_sizes = self.model._prepare_backbone_features(backbone_out)

        if memory is not None:
            # Encode the current image with the previous memory
            B = vision_feats[-1].size(0)  # batch size on this frame
            H, W = feat_sizes[-1]
            C = self.model.memory_attention.d_model
            pix_feat_with_mem = self.model.memory_attention(
                curr=vision_feats[-1:],
                curr_pos=vision_pos_embeds[-1:],
                memory=memory["maskmem_features"],
                memory_pos=memory["maskmem_pos_enc"],
                num_obj_ptr_tokens=B,
            )
            image_embeddings = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
        else:
            # Encode the current image without memory
            vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed
            feats = [
                feat.permute(1, 2, 0).view(1, -1, *feat_size)
                for feat, feat_size in zip(vision_feats[::-1], feat_sizes[::-1])
            ]
            image_embeddings = feats[-1][-1].unsqueeze(0)  # low resolution features

        if len(vision_feats) > 1:
            high_res_features = [
                x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                for x, s in zip(vision_feats[:-1], feat_sizes[:-1])
            ]
        else:
            high_res_features = None

        pred, _, _, _ = self.model.sam_mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=self.se,
            dense_prompt_embeddings=self.de,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res_features,
        )

        high_res_masks = F.interpolate(
            pred,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        # Create new memory
        maskmem_features, maskmem_pos_enc = self.model._encode_new_memory(
            current_vision_feats=vision_feats,
            feat_sizes=feat_sizes,
            pred_masks_high_res=high_res_masks,
            is_mask_from_pts=False,
        )

        output = {
            "pred_masks": high_res_masks,
            "maskmem_features": maskmem_features,
            "maskmem_pos_enc": maskmem_pos_enc,
        }

        return output
        
    
    def create_point_embeddings(self):
        point_grids = build_all_layer_point_grids(
            n_per_side=32, n_layers=0, scale_per_layer=1
        )
        point_coords = point_grids[0] * 1024
        point_labels = np.array([1] * len(point_coords))
        point_coords = torch.as_tensor(point_coords, dtype=torch.float, device=self.device)
        point_labels = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
        unnorm_coords = self._transforms.transform_coords(
            point_coords, normalize=True, orig_hw=[1024, 1024]
        )
        if len(unnorm_coords.shape) == 2:
            unnorm_coords, labels = unnorm_coords[None, ...], point_labels[None, ...]
        concat_points = (unnorm_coords, labels)
        return self.model.sam_prompt_encoder(concat_points, boxes=None, masks=None)




sam2_cp = "/oliver/SAM2/checkpoints/sam2_hiera_tiny.pt"
model_cfg = "sam2_hiera_t.yaml"
model = SAM2_MODEL(model_cfg, sam2_cp, DEVICE)

image = Image.open('/oliver/SAM2/0100.png')
image = image.resize((1024, 1024))
image = np.array(image.convert("RGB"))
image = model._transforms(image)
image = image[None, ...].to(DEVICE)
# %%
