from . import BaseActor
import torch
import numpy as np
import matplotlib.pyplot as plt

class TranstActor(BaseActor):
    """ Actor for training the TransT"""
    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'search_images', 'template_images', 'search_anno'.

        returns:
            loss    - the training loss
            states  -  dict containing detailed losses
        """
        # this code is used to show input data to check the correction
        # import time
        # outputs = self.net(data['search_images'], data['template_images'])
        # data[1]['TIRImage'],  data[1]['TIR90'] , data[1]['TIR180'] , data[1]['TIR270'] TIR rotation images
        # show image
        # TIRimages = data[1]['TIRImage'].cpu().numpy()
        # se_img = data[0]['search_images'].cpu().numpy()
        # tem_img = data[0]['template_images'].cpu().numpy()
        # fig = plt.figure("ImageTIR")
        # plt.subplot(131)
        # plt.imshow(((TIRimages[1].transpose(1, 2, 0))+1)/2)
        # plt.subplot(132)
        # plt.imshow(((se_img[1].transpose(1, 2, 0))+1)/2)
        # plt.subplot(133)
        # plt.imshow(((tem_img[1].transpose(1, 2, 0))+1)/2)
        # plt.pause(0.0001)
        # fig.clf()

        outputs = self.net(data[0]['search_images'], data[0]['template_images'], data[1]['TIRTemplate'], data[1]['TIRSearch'])

        # generate labels
        targets = []
        # for reconstruct
        targets_origin = data[0]['search_anno']
        for i in range(len(targets_origin)):
            h, w = data[0]['search_images'][i][0].shape
            target_origin = targets_origin[i]
            target = {}
            target_origin = target_origin.reshape([1,-1])
            target_origin[0][0] += target_origin[0][2] / 2
            target_origin[0][0] /= w
            target_origin[0][1] += target_origin[0][3] / 2
            target_origin[0][1] /= h
            target_origin[0][2] /= w
            target_origin[0][3] /= h
            target['boxes'] = target_origin
            label = np.array([0])
            label = torch.tensor(label, device=data[0]['search_anno'].device)
            target['labels'] = label
            targets.append(target)
        # target_img = {}
        # target_img['Re_RGB'] = data[0]['search_images'].to(device=outputs['Re_RGB'].device)
        # target_img['Re_TIR'] = data[1]['TIRImage'].to(device=outputs['Re_TIR'].device)
        # targets.append(target_img)

        # Compute loss
        # outputs:(center_x, center_y, width, height)
        loss_dict = self.objective(outputs, targets)
        weight_dict = self.objective.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # Return training stats
        # here can add new loss function to show used in 'tracking/transt.py'
        stats = {'Loss/total': losses.item(),
                 'Loss/ce': loss_dict['loss_ce'].item(),
                 'Loss/bbox': loss_dict['loss_bbox'].item(),
                 'Loss/giou': loss_dict['loss_giou'].item(),
                 'iou': loss_dict['iou'].item(),
                 'Loss/domain': loss_dict['loss_domain'].item(),
                 'Loss/select': loss_dict['loss_select'].item()
                 }
        # 'Loss/domain': loss_dict['loss_domain'].item()

        return losses, stats
