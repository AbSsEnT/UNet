import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, images_dir, masks_dir, transform=None, target_transform=None):
        super(CustomImageDataset, self).__init__()
        self.__annotations = pd.read_csv(annotations_file)
        self.__images_dir = images_dir
        self.__masks_dir = masks_dir
        self.__transform = transform
        self.__target_transform = target_transform

    def __len__(self):
        return len(self.__annotations)

    def __getitem__(self, item):
        image = Image.open(os.path.join(self.__images_dir, self.__annotations.iloc[item, 0])).convert("L")
        mask = Image.open(os.path.join(self.__masks_dir, self.__annotations.iloc[item, 1]))

        if self.__transform:
            image = self.__transform(image)

        if self.__target_transform:
            mask = self.__target_transform(mask)

        sample = {"image": image, "mask": mask}
        return sample

