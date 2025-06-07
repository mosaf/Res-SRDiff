

import torch
from torch.utils.data import Dataset
import os
import h5py
from natsort import natsorted
from torchvision import transforms


class MotionCorruptedMRIDataset2D(Dataset):
    def __init__(self, gt_name: str, to_test=False):
        if to_test:
            files = natsorted(os.listdir(gt_name))
            self.gt_files = [os.path.join(gt_name, x) for x in files]
        else:
            self.gt_files = [os.path.join(gt_name, x) for x in os.listdir(gt_name)]

        self.resize_transform = transforms.Resize((256, 256))
    def __len__(self):
        return len(self.gt_files)

    @staticmethod
    def miniMax(X, max_value=1., min_value=-1.0):
        X_std = (X - X.min()) / (X.max() - X.min())
        return X_std * (max_value - min_value) + min_value


    def center_crop(self, data, shape = (256, 256)):
        """
        Apply a center crop to the input image or batch of complex images.

        Args:
            data: The complex input tensor to be center cropped. It should have at
                least 3 dimensions and the cropping is applied along dimensions -3
                and -2 and the last dimensions should have a size of 2.
            shape: The output shape. The shape should be smaller than the
                corresponding dimensions of data.

        Returns:
            The center cropped image
        """

        w_from = (data.shape[0] - shape[0]) // 2
        h_from = (data.shape[1] - shape[1]) // 2
        w_to = w_from + shape[0]
        h_to = h_from + shape[1]
        return data[w_from:w_to, h_from:h_to]


    def __getitem__(self, idx):
        with h5py.File(self.gt_files[idx], 'r') as h5_file:
            img_hr = torch.from_numpy(h5_file['high_resolution'][()])
            img_lr = torch.from_numpy(h5_file['low_resolution'][()])

        # print(f"before {img_hr.shape}")
        img_hr = torch.nn.functional.pad(img_hr, (80, 80, 80, 80), mode='constant', value=0)
        img_lr = torch.nn.functional.pad(img_lr, (80, 80, 80, 80), mode='constant', value=0)
        # Convert to torch tensors
        img_hr = self.center_crop(self.miniMax(img_hr), shape=(384, 384))
        img_lr = self.center_crop(self.miniMax(img_lr), shape=(384, 384))

        # Return the data
        return {
            'hq': img_hr.unsqueeze(dim=0),  # Shape: (1, H, W)
            'lq': img_lr.unsqueeze(dim=0),  # Shape: (1, H, W)
        }
