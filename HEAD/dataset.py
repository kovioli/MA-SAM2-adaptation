# %%
import os
from torch.utils.data import Dataset
import torch
from PIL import Image
import numpy as np
from torchvision.transforms import ToTensor

def create_train_val_datasets(main_folder, DS_ID, device='cuda', train_ratio=0.8):
    full_dataset = PNGDataset(
        main_folder=main_folder,
        DS_ID=DS_ID,
        device=device
    )

    dataset_size = len(full_dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = dataset_size - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_dataset.dataset.split = 'train'
    val_dataset.dataset.split = 'val'

    return train_dataset, val_dataset


class PNGDataset(Dataset):
    def __init__(self, main_folder, DS_ID, device='cuda'):
        """
        main_folder: str (/oliver/EMPIAR_png)
        DS_ID: str (e.g. 'TS_0001')
        """
        tomogram_dir = os.path.join(main_folder, 'tomograms', DS_ID)
        label_dir = os.path.join(main_folder, 'ribosomes', DS_ID)

        self.device = device
        self.data = []
        
        tomogram_files = sorted([f for f in os.listdir(tomogram_dir) if f.endswith(".png")])
        label_files = sorted([f for f in os.listdir(label_dir) if f.endswith(".png")])
        for t_file, l_file in zip(tomogram_files, label_files):
            assert t_file == l_file, f"Tomogram and label file mismatch: {t_file}, {l_file}"
            self.data.append((os.path.join(tomogram_dir, t_file), os.path.join(label_dir, l_file)))
            
        self.to_tensor = ToTensor()

    def __len__(self):
        # Adjust to the sum of valid slice positions across all volumes
        return len(self.data)

    def __getitem__(self, idx):
        tomogram_path, label_path = self.data[idx]
        
        t = Image.open(tomogram_path)
        t = t.resize((1024, 1024), Image.LANCZOS)
        t = np.array(t.convert('RGB'))
        t = self.to_tensor(t).to(self.device)
        
        l = Image.open(label_path).convert('L')
        l = l.resize((256, 256), Image.LANCZOS)
        l = np.array(l) / 255
        # l = (l > 0.025).astype(np.uint8) # TODO: CHECK IF NEEDED
        l = self.to_tensor(l).to(self.device)#.unsqueeze(0)
        
        return t, l

    

# %%
# ds = PNGDataset('/oliver/EMPIAR_png', 'TS_0001')
# %%
