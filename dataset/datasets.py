import os
import SimpleITK as sitk
import numpy as np
from torch.utils import data
from skimage.transform import resize
import torch
from typing import Union, Literal


class ESMIRADataset2D(data.Dataset):
    def __init__(self, data_root:str, train_dict:dict, transform=None, mean_std:bool=False, full_img:Union[bool,int]=5,
                 path_flag:bool=False, dimension:Literal['2D', '3D']='2D'):
        # train_dict {'site_dirc':[LIST(Target+Atlas): subdir\names.mha:cs:label ], ...}
        self.root = data_root  # the root of data
        self.train_dict = train_dict
        self.transform = transform
        self.mean_std = mean_std
        if isinstance(full_img, int):
            self.slices = full_img
            assert self.slices in [5, 7]
            self.full_img = False
        else:
            self.full_img = full_img
            self.slices = 20
        self.path_flag = path_flag
        self.dimension = dimension

 
    def __len__(self):
        key = list(self.train_dict.keys())[0]
        return len(self.train_dict[key])

    def __getitem__(self, idx):
        data, label, abs_path = self._load_file(idx) if self.dimension=='2D' else self._load_3dfile(idx)
        data = torch.from_numpy(data)
        if self.dimension=='2D':
            data = self.transform(data) if self.transform else data
        else:
            data = self.transform(data) if self.transform else data
            data = torch.unsqueeze(data, dim=0)
        if self.path_flag:
            return data, label, abs_path
        else:
            return data, label  # [N*5, 512, 512], 1:int

    def _load_file(self, idx):  # item -- [5, 512, 512] * N
        data_matrix = []
        path_list = []
        for key in self.train_dict.keys():
            com_info = self.train_dict[key][idx]  # 'subdir\names.mha:cs:label'
            path, cs, label = com_info.split(':')  # 'subdir\names.mha', 'cs', 'label'
            five, ten = cs.split('plus')
            fivelower, fiveupper = five.split('to')
            tenlower, tenupper = ten.split('to')
            if self.slices == 5:
                lower, upper = fivelower, fiveupper
            else:
                lower, upper = tenlower, tenupper
            lower, upper = int(lower), int(upper)
            label = int(label)
            abs_path = os.path.join(self.root, path)
            data_mha = sitk.ReadImage(abs_path)
            data_array = sitk.GetArrayFromImage(data_mha)
            data_array = self._itensity_normalize(data_array[lower:upper])  # [5, 512, 512]
            if data_array.shape != (self.slices, 512, 512):
                if data_array.shape == (self.slices, 256, 256):
                    data_array = resize(data_array, (self.slices, 512, 512), preserve_range=True)  # preserve_range: no normalization
                else:
                    raise ValueError('the shape of input:{}, the id: {}, central_slice: {}'.format(data_array.shape, path, lower))
            data_matrix.append(data_array)
            path_list.append(abs_path)
        return np.vstack(data_matrix).astype(np.float32), label, path_list  # [N*5, 512, 512], 1:int


    def _load_3dfile(self, idx):  # item -- [5, 512, 512] * N
        data_matrix = []
        path_list = []
        for key in self.train_dict.keys():
            com_info = self.train_dict[key][idx]  # 'subdir\names.mha:cs:label'
            path, cs, label = com_info.split(':')  # 'subdir\names.mha', 'cs', 'label'

            
            label = int(label)
            abs_path = os.path.join(self.root, path)
            data_mha = sitk.ReadImage(abs_path)
            data_array = sitk.GetArrayFromImage(data_mha)
            data_array = self._itensity_normalize(data_array)  # [5, 512, 512]
            if data_array.shape != (20, 512, 512):
                if data_array.shape == (20, 256, 256):
                    data_array = resize(data_array, (20, 512, 512), preserve_range=True)  # preserve_range: no normalization
                else:
                    raise ValueError('the shape of input:{}, the id: {}, central_slice: {}'.format(data_array.shape, path, lower))
            data_matrix.append(data_array)
            path_list.append(abs_path)
        return np.vstack(data_matrix).astype(np.float32), label, path_list  # [channel, N*5, 512, 512], 1:int


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