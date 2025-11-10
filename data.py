#!/usr/bin/env python3
""" This script is used to download and save the dataset into disk"""
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from datasets import load_dataset
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import io, os, sys


class ImageDataset(Dataset):
    def __init__(self, hugging_face_dataset):
        super().__init__()
        self.dataset = hugging_face_dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        sample = self.dataset["train"][index]
        image = sample['image']
        label = sample['label']

        return image, label
    
if __name__ == "__main__":
    os.makedirs("images", exist_ok=True)

    dataset  = load_dataset("Hemg/Brain-Tumor-MRI-Dataset")
    torch_dataset = ImageDataset(dataset)
    for idx, row in enumerate(dataset["train"]):
        img, label = torch_dataset[idx]
        img.save(f"images/_{idx}_{label}.png")

