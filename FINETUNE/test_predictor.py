# %%
import torch
import sys
sys.path.append('..')
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

checkpoint = "/media/hdd1/oliver/SAM2/checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint, device=torch.device("cuda:3")))

image = Image.open("./test_predictor.png")
image = np.array(image.convert("RGB"))

input_point = np.array([[250, 250]])
input_label = np.array([1])
input_box = np.array([[50, 130, 420, 520]])
input_box = np.array([[130, 50, 520, 420]])

with torch.inference_mode():
    predictor.set_image(image)
    masks, _, _ = predictor.predict(
        box=input_box,
        multimask_output=True,
    )

# %%
