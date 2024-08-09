# %%
import torch
from PIL import Image
import numpy as np
from sam2.utils.transforms import SAM2Transforms
import matplotlib.pyplot as plt
from _3D_HEAD.model import SAM2_MODEL
import matplotlib.pyplot as plt
DEVICE = 'cuda:1'
sam2_cp = "/oliver/SAM2/checkpoints/sam2_hiera_tiny.pt"
model_cfg = "sam2_hiera_t.yaml"

_transforms = SAM2Transforms(
    resolution=1024,
    mask_threshold=0.0,
    max_hole_area=0.0,
    max_sprinkle_area=0.0,
)

image = Image.open('/oliver/SAM2/dog.jpg')
image = image.resize((1024, 1024))
image = np.array(image.convert("RGB"))
image = _transforms(image)
image = image[None, ...].to(DEVICE)

sam2_cp = "/oliver/SAM2/checkpoints/sam2_hiera_tiny.pt"
model_cfg = "sam2_hiera_t.yaml"

model = SAM2_MODEL(model_cfg, sam2_cp, device=DEVICE)
# %%
with torch.no_grad():
    pred, memory = model(image)
# %%

with torch.no_grad():
    pred_next, memory_next = model(image, memory)