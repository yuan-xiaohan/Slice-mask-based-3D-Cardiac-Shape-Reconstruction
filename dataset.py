import glob
import os

import cv2
import numpy as np
import torch
import random
from torch.utils.data.dataset import Dataset
from processing import *

PCA_NUM = 51

class DataTrain(Dataset):
    def __init__(self, file_path):
        file_list = []
        for filename in os.listdir(file_path):
            file_list.append(os.path.join(file_path, filename))
        self.file_list = file_list

        # Calculate len
        self.data_len = len(self.file_list)


    def __getitem__(self, index):
        """Get specific data corresponding to the index
        Args:
            index (int): index of the data
        Returns:
            Tensor: specific data on index which is converted to Tensor
        """
        name_list = ["2CH", "3CH", "4CH", "SAX"]
        img_list = []
        masks_list = []
        for name in name_list:
            if name == "SAX":
                for i in range(10):
                    file_name = os.path.join(self.file_list[index], "slice", name + "_%02d" % i + ".png")
                    img = cv2.imread(file_name, 0).astype(np.float32)
                    img_list.append(img)
                    file_name = os.path.join(self.file_list[index], "slice-mask", name + "_%02d" % i + ".png")
                    mask = cv2.imread(file_name, 0).astype(np.uint8)
                    masks_list.append(mask)
            else:
                file_name = os.path.join(self.file_list[index], "slice", name + ".png")
                img = cv2.imread(file_name, 0).astype(np.float32)
                img_list.append(img)
                file_name = os.path.join(self.file_list[index], "slice-mask", name + ".png")
                mask = cv2.imread(file_name, 0).astype(np.uint8)
                masks_list.append(mask)

        img_data = np.array(img_list)
        mask_data = np.array(masks_list)

        # Brightness
        alpha = random.uniform(0.0, 1.0)
        if (alpha > 0.5):
            pix_add = random.randint(-20, 20)
            for i in range(len(img_list)):
                img_as_np = change_brightness(img_data[i, :, :], pix_add)

        # Normalize the image
        img_data = normalization2(img_data, max=1, min=0)
        img_as_tensor = torch.from_numpy(img_data).float()  # Convert numpy array to tensor

        mask_data = mask_data / 255
        # msk_as_np = np.expand_dims(msk_as_np, axis=0)  # add additional dimension
        msk_as_tensor = torch.from_numpy(mask_data).long()  # Convert numpy array to tensor

        """
        # GET PCA
        """
        pca_data = np.loadtxt(os.path.join(self.file_list[index], 'theta_nor.txt'))[0:PCA_NUM].astype(np.float32)
        pca_tensor = torch.from_numpy(pca_data).float()

        return (img_as_tensor, (msk_as_tensor, pca_tensor))

    def __len__(self):
        """
        Returns:
            length (int): length of the data
        """
        return self.data_len

class OnlyTest(Dataset):
    def __init__(self, file_path):
        file_list = []
        for filename in os.listdir(file_path):
            file_list.append(os.path.join(file_path, filename))
        self.file_list = file_list

        # Calculate len
        self.data_len = len(self.file_list)

    def __getitem__(self, index):
        name_list = ["2CH", "3CH", "4CH", "SAX"]
        img_list = []
        for name in name_list:
            if name == "SAX":
                for i in range(10):
                    file_name = os.path.join(self.file_list[index], "slice", name + "_%02d" % i + ".png")
                    img = cv2.imread(file_name, 0).astype(np.float32)
                    img_list.append(img)
            else:
                file_name = os.path.join(self.file_list[index], "slice", name + ".png")
                img = cv2.imread(file_name, 0).astype(np.float32)
                img_list.append(img)

        img_data = np.array(img_list)
        # Normalize the image
        img_data = normalization2(img_data, max=1, min=0)
        img_as_tensor = torch.from_numpy(img_data).float()  # Convert numpy array to tensor

        return (img_as_tensor)

    def __len__(self):
        return self.data_len
