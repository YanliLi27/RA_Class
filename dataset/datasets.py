import os
import sys
import SimpleITK as sitk
import numpy as np
from torch.utils import data
from scipy.ndimage import binary_fill_holes
import pickle
from skimage.transform import resize
from typing import Union


class ESMIRADataset2D(data.Dataset):
    def __init__(self, data_root:str, train_dict:dict, transform=None, mean_std:bool=False):
        # train_dict {'site_dirc':[LIST(Target+Atlas): subdir\names.mha:cs:label ], ...}
        self.root = data_root  # the root of data
        self.train_dict = train_dict
        self.transform = transform
        self.mean_std = mean_std
 
    def __len__(self):
        key = self.train_dict.keys()
        return len(self.train_dict[key])

    def __getitem__(self, idx):
        # if fuse 2 organs, use: self.load_datafile_2_organ(self.dataset_list, idx)
        # if fuse 3 organs, use: self.load_datafile_3_organ(self.dataset_list, idx)
        # data, label = self.load_datafile_2_organ(self.dataset_list, idx) # [organ*2*slices, 512, 512]
        data, label = self._load_file(idx)
        data = self.transform(data)
        return data, label  # [N*5, 512, 512], 1:int

    def _load_file(self, idx):  # item -- [5, 512, 512] * N
        data_matrix = []
        for key in self.train_dict.keys():
            com_info = self.train_dict[key][idx]  # 'subdir\names.mha:cs:label'
            path, cs, label = com_info.split(':')  # 'subdir\names.mha', 'cs', 'label'
            lower, upper = cs.split('to')
            lower, upper = int(lower), int(upper)
            label = int(label)
            abs_path = os.path.join(self.root, path)
            data_mha = sitk.ReadImage(abs_path)
            data_array = sitk.GetArrayFromImage(data_mha)
            data_array = self._itensity_normalize(data_array[lower:upper])  # [5, 512, 512]
            if data_array.shape != (5, 512, 512):
                if data_array.shape == (5, 256, 256):
                    data_array = resize(data_array, (5, 512, 512), preserve_range=True)  # preserve_range: no normalization
                else:
                    raise ValueError('the shape of input:{}, the id: {}, central_slice: {}'.format(data_array.shape, path, lower))
            data_matrix.append(data_array)
        return np.vstack(data_matrix), label  # [N*5, 512, 512], 1:int


    def _itensity_normalize(self, volume: np.array):
        """
        normalize the itensity of a volume based on the mean and std of nonzeor region
        inputs:
            volume: the input volume
        outputs:
            out: the normalized volume
        """
        min_value = volume.min()
        max_value = volume.max()
        if max_value > min_value:
            out = (volume - min_value) / (max_value - min_value)
        else:
            out = volume
        # out_random = np.random.normal(0, 1, size=volume.shape)
        # out[volume == 0] = out_random[volume == 0]
        return out

    def _itensity_normalize_ms(self, volume: np.array):
        """
        normalize the itensity of a volume based on the mean and std of nonzeor region
        inputs:
            volume: the input volume
        outputs:
            out: the normalized volume
        """
        mean = np.mean(volume[volume!=0])
        std = np.std(volume[volume!=0])
        out = (volume[volume!=0] - mean)/std
        return out