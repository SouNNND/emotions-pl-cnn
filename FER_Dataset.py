import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class FERPlusDataset(Dataset):
    def __init__(self, csv_file, transform=None, train=True):
        self.data = pd.read_csv(csv_file)
        if(train):
            self.data = self.data[self.data['Usage'] != 'PrivateTest']
        else:
            self.data = self.data[self.data['Usage'] == 'PrivateTest']
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = np.fromstring(self.data.loc[idx, 'pixels'], dtype=int, sep=' ').reshape(48, 48)
        image = Image.fromarray(image.astype(np.uint8))

        label = self.data.loc[idx, 'emotion']

        if self.transform:
            image = self.transform(image)

        return image, label

# Define transformations for FER+ dataset
fer_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),  # Ensure the images have 3 channels
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.485, 0.485], std=[0.229, 0.229, 0.229])
    transforms.Normalize(mean=[0.5], std=[0.5])
])
