from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision.transforms
import torchvision.transforms.functional as TF
import torch
from pytracking import TensorDict

# TIR dataset class
class TIR_dataset(Dataset):

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.video_dir = os.listdir(self.root_dir)
        img_ir_path = []
        for video in self.video_dir:
            current_path_ir = os.path.join(self.root_dir, video)
            img_ir_list = os.listdir(current_path_ir)
            img_ir_list_path = [current_path_ir + '/' + x for x in img_ir_list]
            img_ir_path = img_ir_path + img_ir_list_path
        self.img_ir_path = img_ir_path

    def transform(self, img_ir):
        # random crop
        # i, j, h, w = torchvision.transforms.RandomCrop.get_params(img_ir, output_size=(64, 64))
        # img_ir = TF.crop(img_ir, i, j, h, w)
        # img_vis = TF.crop(img_vis, i, j, h, w)
        # reize
        img_ir = TF.resize(img_ir, [128, 128])
        #img_vis = TF.resize(img_vis, [256, 256])
        # grayscale
        # img_ir = TF.to_grayscale(img_ir, num_output_channels=3)
        img_ir = TF.to_grayscale(img_ir, num_output_channels=3)
        #img_vis = TF.to_grayscale(img_vis)
        # to tensor
        img_ir = TF.to_tensor(img_ir)
        #img_vis = TF.to_tensor(img_vis)
        # Normalize [-1, 1]
        img_ir = TF.normalize(img_ir, 0.5, 0.5)
        #img_vis = TF.normalize(img_vis, 0.5, 0.5)
        # rotation
        # img_ir_90 = torch.rot90(img_ir, k=-1, dims=[1, 2])
        # img_ir_180 = torch.rot90(img_ir, k=2, dims=[1, 2])
        # img_ir_270 = torch.rot90(img_ir, k=1, dims=[1, 2])
        # data = TensorDict({'TIRImage': img_ir,
        #                    'TIR90': img_ir_90,
        #                    'TIR180': img_ir_180,
        #                    'TIR270': img_ir_270})

        data = TensorDict({'TIRImage': img_ir})
        return data

    def __getitem__(self, idx):
        img_ir_name = self.img_ir_path[idx]
        # img_vis_name = img_ir_name.replace('lwir', 'visible')
        img_ir = Image.open(img_ir_name)
        # img_vis = Image.open(img_vis_name)
        # self-defined transform
        data = self.transform(img_ir)
        return data

    def __len__(self):
        return len(self.img_ir_path)
