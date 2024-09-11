import os
import numpy as np
import torch
import mrcfile
from _3D_HEAD.model_3D import HeadModel
from _3D_HEAD.dataset import PNGDataset
from _3D_HEAD.config import DEVICE, MODEL_TYPE
from skimage.transform import resize


def load_model_from_path(model_path: str):
    model = HeadModel(MODEL_TYPE, DEVICE, in_chan=3)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def predict(
        PRED_ID: str,
        TIMESTAMP: str,
        PREDICTION_DIR: str,
        target_shape: tuple,
        memory_used: bool
    ):
    """
    memory_used: whether the model used memory during training
    example:
        - PRED_ID: 0001
        - TIMESTAMP: 27082024_13:40
        - PREDICTION_MAIN_DIR: /oliver/SAM2/PREDICT
        - target_shape: (210, 927, 927)
    """
    
    pred_save_path = os.path.join(PREDICTION_DIR, f'TS_{PRED_ID}.mrc')
    # if pred_save_path already exists, skip
    if os.path.exists(pred_save_path):
        print(f"Prediction for {PRED_ID} already exists.")
        return
    
    if not os.path.exists(PREDICTION_DIR):
        os.makedirs(PREDICTION_DIR)
    
    model_path = os.path.join(
        '/media',
        'hdd1',
        'oliver',
        'SAM2',
        'checkpoints',
        TIMESTAMP,
        'best_model.pth'
    )
    model = load_model_from_path(model_path)
    ds = PNGDataset(main_folder='/oliver/EMPIAR_png', DS_ID=f"TS_{PRED_ID}", device=DEVICE)
    
    all_predictions = []
    memory = None
    with torch.no_grad():
        for i in range(len(ds)):
            if i % 20 == 0:
                print(f"Slice {i}")
            pred, memory = model(ds[i][0].unsqueeze(0), memory)
            if not memory_used:
                memory = None
            all_predictions.append(pred.cpu().numpy().squeeze(0))
    
    all_predictions = np.concatenate(all_predictions, axis=0)
    print("resizing...")
    resized_pred = resize(all_predictions, target_shape, mode='reflect', anti_aliasing=True)
    
    with mrcfile.new(pred_save_path, overwrite=True) as mrc:
        mrc.set_data(resized_pred)
    
    return