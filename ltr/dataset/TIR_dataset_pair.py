from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision.transforms
import torchvision.transforms.functional as TF
import torch
import cv2
from pytracking import TensorDict
import json
import numpy as np
import torch
from collections import namedtuple
Corner = namedtuple('Corner', 'x1 y1 x2 y2')
BBox = Corner
Center = namedtuple('Center', 'x y w h')

# TIR dataset class
class TIR_dataset_pair(Dataset):

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.video_dir = os.listdir(self.root_dir)
        print('准备读取json')
        with open('/data/liqiao/dataset/0.json', 'r') as f:
            meta_data = json.load(f)
            #meta_data = _filter_zero(meta_data)
        self.json = meta_data
        print('json读取完毕')
        img_ir_path = []
        print('开始读取图片了')
        i = 0
        for dirpath, dirnames, filenames in os.walk(self.root_dir):
            for filename in filenames:
                if filename.endswith('z.jpg'):
                    img_ir_path.append(os.path.join(dirpath,filename))
                    i += 1
                    print(i)
        self.img_ir_path = img_ir_path
        print('图片读取完成')
    def transform(self, img_ir, img_ir_s , search_image_anno):
        # random crop
        # i, j, h, w = torchvision.transforms.RandomCrop.get_params(img_ir, output_size=(64, 64))
        # img_ir = TF.crop(img_ir, i, j, h, w)
        # img_vis = TF.crop(img_vis, i, j, h, w)
        # reize
        # img_ir = TF.resize(img_ir, [128, 128])
        #img_vis = TF.resize(img_vis, [256, 256])
        # grayscale
        # img_ir = TF.to_grayscale(img_ir, num_output_channels=3)
        img_ir = TF.to_grayscale(img_ir, num_output_channels=3)
        img_ir_s = TF.to_grayscale(img_ir_s, num_output_channels=3)
        #img_vis = TF.to_grayscale(img_vis)
        # to tensor
        img_ir = TF.to_tensor(img_ir)
        img_ir_s = TF.to_tensor(img_ir_s)
        img_ir_s = img_ir_s.permute(1,2,0)

        search_box = _get_bbox(img_ir_s , search_image_anno)
        img_ir_s, bbox = crop(img_ir_s,search_box,size = 255)
        img_ir_s = torch.tensor(img_ir_s).permute(2,0,1)
        #img_vis = TF.to_tensor(img_vis)
        # Normalize [-1, 1]
        img_ir = TF.normalize(img_ir, 0.5, 0.5)
        img_ir_s = TF.normalize(img_ir_s, 0.5, 0.5)
        #img_vis = TF.normalize(img_vis, 0.5, 0.5)
        # rotation
        # img_ir_90 = torch.rot90(img_ir, k=-1, dims=[1, 2])
        # img_ir_180 = torch.rot90(img_ir, k=2, dims=[1, 2])
        # img_ir_270 = torch.rot90(img_ir, k=1, dims=[1, 2])
        # data = TensorDict({'TIRImage': img_ir,
        #                    'TIR90': img_ir_90,
        #                    'TIR180': img_ir_180,
        #                    'TIR270': img_ir_270})

        data = TensorDict({'TIRTemplate': img_ir,
                           'TIRSearch': img_ir_s})
        return data

    def __getitem__(self, idx):
        img_ir_name = self.img_ir_path[idx]
        #img_ir_sname = img_ir_name.replace('T.jpg', 'S.jpg')
        img_ir_sname = img_ir_name.replace('z.jpg', 'x.jpg')

        video = os.path.join(img_ir_name.split('/')[-3], img_ir_name.split('/')[-2])
        track = img_ir_name.split('/')[-1].split('.')[1]
        frame = img_ir_name.split('/')[-1].split('.')[0]
        search_image_anno = self.json[video][track][frame]
        img_ir = Image.open(img_ir_name)
        img_ir_s = Image.open(img_ir_sname)
        # self-defined transform
        data = self.transform(img_ir, img_ir_s ,search_image_anno)



        return data

    def __len__(self):
        return len(self.img_ir_path)


# def _filter_zero(meta_data):
#     meta_data_new = {}
#     for video, tracks in meta_data.items():
#         new_tracks = {}
#         if 'aver_vary' in tracks:
#             del tracks['aver_vary']
#             del tracks['bbox_found_freq']
#             del tracks['bbox_picked_freq']
#         for trk, frames in tracks.items():
#             new_frames = {}
#             for frm, bbox in frames.items():
#                 if not isinstance(bbox, dict):
#                     if len(bbox) == 4:
#                         # x1, y1, x2, y2 = bbox
#                         # w, h = x2 - x1, y2 - y1
#                         x,y,w,h = bbox
#                     else:
#                         w, h = bbox
#                     if w <= 0 or h <= 0:
#                         continue
#                 new_frames[frm] = bbox
#             if len(new_frames) > 0:
#                 new_tracks[trk] = new_frames
#         if len(new_tracks) > 0:
#             meta_data_new[video] = new_tracks
#     return meta_data_new

def _get_bbox( image, shape):
    imh, imw = image.shape[:2]
    if len(shape) == 4:
        w, h = shape[2], shape[3]
    else:
        w, h = shape
    context_amount = 0.5
    exemplar_size = 127 #127
    wc_z = w + context_amount * (w+h)
    hc_z = h + context_amount * (w+h)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = exemplar_size / s_z
    w = w*scale_z
    h = h*scale_z
    cx, cy = imw//2, imh//2
    bbox = center2corner(Center(cx, cy, w, h))
    return bbox


def center2corner(center):
    """ convert (cx, cy, w, h) to (x1, y1, x2, y2)
    Args:
        center: Center or np.array (4 * N)
    Return:
        center or np.array (4 * N)
    """
    if isinstance(center, Center):
        x, y, w, h = center
        return Corner(x - w * 0.5, y - h * 0.5, x + w * 0.5, y + h * 0.5)
    else:
        x, y, w, h = center[0], center[1], center[2], center[3]
        x1 = x - w * 0.5
        y1 = y - h * 0.5
        x2 = x + w * 0.5
        y2 = y + h * 0.5
        return x1, y1, x2, y2



def crop(image, bbox, size):
    shape = image.shape
    crop_bbox = center2corner(Center(shape[0]//2, shape[1]//2,
                                     size-1, size-1))

    # shift scale augmentation
    image, bbox = _shift_scale_aug(image, bbox, crop_bbox, size)

    return image, bbox

def _shift_scale_aug(image, bbox, crop_bbox, size):
    im_h, im_w = image.shape[:2]

    # adjust crop bounding box
    crop_bbox_center = corner2center(crop_bbox)

    scale_x = (1.0 + (np.random.random() * 2 - 1.0) * 0.18)#self.scale
    scale_y = (1.0 + (np.random.random() * 2 - 1.0) * 0.18)
    h, w = crop_bbox_center.h, crop_bbox_center.w
    scale_x = min(scale_x, float(im_w) / w)
    scale_y = min(scale_y, float(im_h) / h)
    crop_bbox_center = Center(crop_bbox_center.x,
                              crop_bbox_center.y,
                              crop_bbox_center.w * scale_x,
                              crop_bbox_center.h * scale_y)

    crop_bbox = center2corner(crop_bbox_center)

    sx = (np.random.random() * 2 - 1.0) * 64#self.shift
    sy = (np.random.random() * 2 - 1.0) * 64#self.shift

    x1, y1, x2, y2 = crop_bbox

    sx = max(-x1, min(im_w - 1 - x2, sx))
    sy = max(-y1, min(im_h - 1 - y2, sy))

    crop_bbox = Corner(x1 + sx, y1 + sy, x2 + sx, y2 + sy)

    # # adjust target bounding box
    # x1, y1 = crop_bbox.x1, crop_bbox.y1
    # bbox = Corner(bbox.x1 - x1, bbox.y1 - y1,
    #               bbox.x2 - x1, bbox.y2 - y1)
    #
    # if self.scale:
    #     bbox = Corner(bbox.x1 / scale_x, bbox.y1 / scale_y,
    #                   bbox.x2 / scale_x, bbox.y2 / scale_y)

    image = _crop_roi(image, crop_bbox, size)
    return image, bbox

def corner2center(corner):
    """ convert (x1, y1, x2, y2) to (cx, cy, w, h)
    Args:
        conrner: Corner or np.array (4*N)
    Return:
        Center or np.array (4 * N)
    """
    if isinstance(corner, Corner):
        x1, y1, x2, y2 = corner
        return Center((x1 + x2) * 0.5, (y1 + y2) * 0.5, (x2 - x1), (y2 - y1))
    else:
        x1, y1, x2, y2 = corner[0], corner[1], corner[2], corner[3]
        x = (x1 + x2) * 0.5
        y = (y1 + y2) * 0.5
        w = x2 - x1
        h = y2 - y1
        return x, y, w, h


def _crop_roi(image, bbox, out_sz, padding=(0, 0, 0)):
    bbox = [float(x) for x in bbox]
    a = (out_sz-1) / (bbox[2]-bbox[0])
    b = (out_sz-1) / (bbox[3]-bbox[1])
    c = -a * bbox[0]
    d = -b * bbox[1]
    mapping = np.array([[a, 0, c],
                        [0, b, d]]).astype(float)
    image = image.to(torch.float32).numpy()

    crop = cv2.warpAffine(image, mapping, (out_sz, out_sz),
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=padding)
    return crop