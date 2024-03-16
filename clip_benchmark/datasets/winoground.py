import os
from torch.utils.data import Dataset
from PIL import Image
import torch
import json

class WinoGround(Dataset):

    def __init__(self, root=".", transform=None):
        from datasets import load_dataset
        self.ds = load_dataset("facebook/winoground", cache_dir=root)["test"]
        # uncomment for 80 split
        # indices = list(range(len(self.ds) - 380, len(self.ds)))
        # self.ds = self.ds.select(indices)
        # print(len(self.ds))

        # uncomment for Object filter
        filtered_ds = self.ds.filter(lambda x: x['collapsed_tag'] == 'Object')
        self.ds =  filtered_ds
        print(len(self.ds))
        
        self.transform = transform

    def __getitem__(self, idx):
        data = self.ds[idx]
        img0 = data["image_0"]
        img1 = data["image_1"]
        cap0 = data["caption_0"]
        cap1 = data["caption_1"]
        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
            imgs = torch.stack([img0, img1])
        else:
            imgs = [img0, img1]
        caps = [cap0, cap1]
        return imgs, caps

    def __len__(self):
        return len(self.ds)
