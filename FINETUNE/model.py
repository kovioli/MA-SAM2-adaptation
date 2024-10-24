import torch
import torch.nn as nn
import numpy as np
import sys

sys.path.append("..")
from sam2.build_sam import build_sam2
from sam2.utils.transforms import SAM2Transforms
from typing import List


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


class SAM2_finetune(nn.Module):
    def __init__(self, model_cfg, ckpt_path, device, use_point_grid=False):
        super(SAM2_finetune, self).__init__()
        self.model = build_sam2(model_cfg, ckpt_path, device, mode="train")
        self.device = device
        self._transforms = SAM2Transforms(
            resolution=1024,
            mask_threshold=0.0,
            max_hole_area=0.0,
            max_sprinkle_area=0.0,
        )
        if use_point_grid:
            self.se, self.de = self.create_point_embeddings()
        else:
            self.se, self.de = self.model.sam_prompt_encoder(None, None, None)
        self._bb_feat_sizes = [
            (256, 256),
            (128, 128),
            (64, 64),
        ]

        for name, param in self.model.named_parameters():
            param.requires_grad = False
        for name, param in self.model.named_parameters():
            if "sam_mask_decoder" in name or "image_encoder" in name:
                param.requires_grad = True

    def forward(self, x):
        backbone_out = self.model.forward_image(x)
        _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)
        vision_feats[-1] = (
            vision_feats[-1] + self.model.no_mem_embed
        )  # TODO: CHECK IF THIS IS NEEDED
        feats = [
            feat.permute(1, 2, 0).view(1, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1]
        _features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
        high_res_features = [
            feat_level[-1].unsqueeze(0) for feat_level in _features["high_res_feats"]
        ]
        pred, _, _, _ = self.model.sam_mask_decoder(
            image_embeddings=_features["image_embed"][-1].unsqueeze(0),
            image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=self.se,
            dense_prompt_embeddings=self.de,
            multimask_output=False,  # TODO CHECK IF NEEDED -> False!
            repeat_image=False,
            high_res_features=high_res_features,
        )
        return pred

    def get_embedding_dims(self, vision_feats, feat_sizes):
        B = vision_feats[-1].size(1)
        C = self.model.memory_attention.d_model
        H, W = feat_sizes[-1]
        return B, C, H, W

    def create_point_embeddings(self):
        point_grids = build_all_layer_point_grids(
            n_per_side=2, n_layers=0, scale_per_layer=1
        )
        point_coords = point_grids[0] * 1024
        point_labels = np.array([1] * len(point_coords))
        point_coords = torch.as_tensor(
            point_coords, dtype=torch.float, device=self.device
        )
        point_labels = torch.as_tensor(
            point_labels, dtype=torch.int, device=self.device
        )
        unnorm_coords = self._transforms.transform_coords(
            point_coords, normalize=True, orig_hw=[1024, 1024]
        )
        if len(unnorm_coords.shape) == 2:
            unnorm_coords, labels = unnorm_coords[None, ...], point_labels[None, ...]
        concat_points = (unnorm_coords, labels)
        return self.model.sam_prompt_encoder(concat_points, boxes=None, masks=None)
