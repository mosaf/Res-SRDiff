import torch
from torch.utils.data import Dataset
import os
import h5py
from natsort import natsorted


class Res_SRDiff(Dataset):
    def __init__(self, gt_name: str, to_test=False):
        if to_test:
            files = natsorted(os.listdir(gt_name))
        else:
            files = os.listdir(gt_name)

        self.gt_files = [os.path.join(gt_name, x) for x in files]

        # Preload all images into RAM
        self.data = []
        for file in self.gt_files:
            with h5py.File(file, 'r') as h5_file:
                img_hr = torch.from_numpy(h5_file['high_resolution'][()])
                img_lr = torch.from_numpy(h5_file['low_resolution'][()])

                # Normalize
                img_hr = self.miniMax(img_hr)
                img_lr = self.miniMax(img_lr)

                self.data.append({
                    'hq': img_hr.unsqueeze(dim=0),  # Shape: (1, H, W)
                    'lq': img_lr.unsqueeze(dim=0),  # Shape: (1, H, W)
                })

    def __len__(self):
        return len(self.data)

    @staticmethod
    def miniMax(X, max_value=1., min_value=-1.0):
        X_std = (X - X.min()) / (X.max() - X.min())
        return X_std * (max_value - min_value) + min_value

    def __getitem__(self, idx):
        return self.data[idx]  # Directly return preloaded data


class Res_SRDiff_onFly(Dataset):
    def __init__(self, gt_name: str, to_test=False):
        if to_test:
            files = natsorted(os.listdir(gt_name))
            self.gt_files = [os.path.join(gt_name, x) for x in files]
        else:
            self.gt_files = [os.path.join(gt_name, x) for x in os.listdir(gt_name)]

    def __len__(self):
        return len(self.gt_files)

    @staticmethod
    def miniMax(X, max_value=1., min_value=-1.0):
        X_std = (X - X.min()) / (X.max() - X.min())
        return X_std * (max_value - min_value) + min_value

    def __getitem__(self, idx):
        with h5py.File(self.gt_files[idx], 'r') as h5_file:
            img_hr = torch.from_numpy(h5_file['high_resolution'][()])
            img_lr = torch.from_numpy(h5_file['low_resolution'][()])

        # Convert to torch tensors
        img_hr = self.miniMax(img_hr)
        img_lr = self.miniMax(img_lr)

        # Return the data
        return {
            'hq': img_hr.unsqueeze(dim=0),  # Shape: (1, H, W)
            'lq': img_lr.unsqueeze(dim=0),  # Shape: (1, H, W)
        }