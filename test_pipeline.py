# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2
from typing import List
from sam2.utils.transforms import SAM2Transforms
import torch.nn.functional as F
DEVICE = 'cuda:1'
# %%
sam2_cp = "/oliver/SAM2/checkpoints/sam2_hiera_tiny.pt"
model_cfg = "sam2_hiera_t.yaml"
model = build_sam2(model_cfg, sam2_cp, device=DEVICE)
_transforms = SAM2Transforms(
    resolution=1024,
    mask_threshold=0.0,
    max_hole_area=0.0,
    max_sprinkle_area=0.0,
)
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

def create_point_embeddings():
    point_grids = build_all_layer_point_grids(
        n_per_side=32, n_layers=0, scale_per_layer=1
    )
    point_coords = point_grids[0] * 1024
    point_labels = np.array([1] * len(point_coords))
    point_coords = torch.as_tensor(point_coords, dtype=torch.float, device=DEVICE)
    point_labels = torch.as_tensor(point_labels, dtype=torch.int, device=DEVICE)
    unnorm_coords = _transforms.transform_coords(
        point_coords, normalize=True, orig_hw=[1024, 1024]
    )
    if len(unnorm_coords.shape) == 2:
        unnorm_coords, labels = unnorm_coords[None, ...], point_labels[None, ...]
    concat_points = (unnorm_coords, labels)
    return model.sam_prompt_encoder(concat_points, boxes=None, masks=None)
# %%
# image = Image.open('/oliver/SAM2/0100.png')
image = Image.open('/oliver/SAM2/dog.jpg')
image = image.resize((1024, 1024))
image = np.array(image.convert("RGB"))
image = _transforms(image)
image = image[None, ...].to(DEVICE)
_bb_feat_sizes = [
    (256, 256),
    (128, 128),
    (64, 64),
]
se, de = create_point_embeddings()

# %%
with torch.no_grad():
    backbone_out = model.forward_image(image)
backbone_out, vision_feats, vision_pos_embeds, feat_sizes = model._prepare_backbone_features(backbone_out)
vision_feats[-1] = vision_feats[-1] + model.no_mem_embed
feats = [
    feat.permute(1, 2, 0).view(1, -1, *feat_size)
    for feat, feat_size in zip(vision_feats[::-1], feat_sizes[::-1])
][::-1]
image_embeddings = feats[-1][-1].unsqueeze(0)
if len(vision_feats) > 1:
    high_res_features = [
        feat_level[-1].unsqueeze(0)
        for feat_level in feats[:-1]
    ]
else:
    high_res_features = None
image_embeddings = feats[-1][-1].unsqueeze(0)
with torch.no_grad():
    pred, _, _, _ = model.sam_mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=model.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=se,
        dense_prompt_embeddings=de,
        multimask_output=False, # TODO CHECK IF NEEDED -> False!
        repeat_image=False,
        high_res_features=high_res_features,
    )
image_size = 1024
high_res_masks = F.interpolate(
    pred,
    size=(image_size, image_size),
    mode="bilinear",
    align_corners=False,
)
B = vision_feats[-1].size(1)
C = model.memory_attention.d_model
H, W = feat_sizes[-1]
pix_feat = vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
mem  = model.memory_encoder(
    pix_feat=pix_feat,
    masks=high_res_masks,
    skip_mask_sigmoid=False,
)
maskmem_features = mem.get('vision_features')
maskmem_pos_enc = mem.get('vision_pos_enc')

# create pixel features with memory
# to_cat_memory = [maskmem_features.flatten(2).permute(2, 0, 1)]
# to_cat_memory_pos_embed = [maskmem_pos_enc[-1].flatten(2).permute(2, 0, 1) + model.maskmem_tpos_enc[0]]
# memory = torch.cat(to_cat_memory, dim=0)
# memory_pos_embed = torch.cat(to_cat_memory_pos_embed, dim=0)

memory = maskmem_features.flatten(2).permute(2, 0, 1)
memory_pos_embed = maskmem_pos_enc[-1].flatten(2).permute(2, 0, 1) + model.maskmem_tpos_enc[0]

pix_feat_with_mem = model.memory_attention(
    curr=vision_feats[-1:],
    curr_pos=vision_pos_embeds[-1:],
    memory=memory,
    memory_pos=memory_pos_embed,
    num_obj_ptr_tokens=0
)
pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
# %%
sam2_cp = "/oliver/SAM2/checkpoints/sam2_hiera_tiny.pt"
model_cfg = "sam2_hiera_t.yaml"

sam2_model = build_sam2(model_cfg, sam2_cp, device='cuda:3')
predictor = SAM2ImagePredictor(sam2_model)
# %%
# SET IMAGE EMBEDDING
_orig_hw = [image.shape[:2]]
input_image = predictor._transforms(image)
input_image = input_image[None, ...].to(DEVICE)
backbone_out = sam2_model.forward_image(input_image) # -> dict_keys(['vision_features', 'vision_pos_enc', 'backbone_fpn'])
_, vision_feats, _, _ = sam2_model._prepare_backbone_features(backbone_out)
# -> vision_feats[0].shape: torch.Size([65536, 1, 32]); vision_feats[1].shape: torch.Size([16384, 1, 64]); vision_feats[2].shape: torch.Size([4096, 1, 256])

vision_feats[-1] = vision_feats[-1] + sam2_model.no_mem_embed
feats = [
    feat.permute(1, 2, 0).view(1, -1, *feat_size)
    for feat, feat_size in zip(vision_feats[::-1], predictor._bb_feat_sizes[::-1])
][::-1]
# feats[0].shape: torch.Size([1, 32, 256, 256])
# feats[1].shape: torch.Size([1, 64, 128, 128])
# feats[2].shape: torch.Size([1, 256, 64, 64])
_features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]} # == IMAGE EMBEDDING

# %%
# PREPARE PROMPTS
point_coords = torch.as_tensor(point_coords, dtype=torch.float, device=DEVICE)
point_labels = torch.as_tensor(point_labels, dtype=torch.long, device=DEVICE)
normalize_coords = True
unnorm_coords = predictor._transforms.transform_coords(
    point_coords, normalize=normalize_coords, orig_hw=_orig_hw[-1]
)
labels = torch.as_tensor(point_labels, dtype=torch.int, device=DEVICE)
if len(unnorm_coords.shape) == 2:
    unnorm_coords, labels = unnorm_coords[None, ...], labels[None, ...]


# %%
# PREDICT -> unnorm_coords, labels
concat_points = (unnorm_coords, labels)
se, de = sam2_model.sam_prompt_encoder(
    points = concat_points,
    boxes = None,
    masks = None,
)
# %%
high_res_features = [
    feat_level[-1].unsqueeze(0)
    for feat_level in _features["high_res_feats"]
]
low_res_masks, iou_predictions, _, _ = sam2_model.sam_mask_decoder(
    image_embeddings=_features["image_embed"][-1].unsqueeze(0),
    image_pe=sam2_model.sam_prompt_encoder.get_dense_pe(),
    sparse_prompt_embeddings=se,
    dense_prompt_embeddings=de,
    multimask_output=False, # TODO CHECK IF NEEDED -> False!
    repeat_image=False,
    high_res_features=high_res_features,
)
masks = predictor._transforms.postprocess_masks(
    low_res_masks, _orig_hw[-1]
)

low_res_masks = torch.clamp(low_res_masks, -32.0, 32.0)
# %%
