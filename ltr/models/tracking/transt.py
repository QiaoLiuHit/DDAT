import math

import torch.nn as nn
from ltr import model_constructor

import torch
import torch.nn.functional as F
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor,
                       nested_tensor_from_tensor_2,
                       accuracy)

from ltr.models.backbone.transt_backbone import build_backbone
from ltr.models.loss.matcher import build_matcher
from ltr.models.neck.featurefusion_network import build_featurefusion_network
from ltr.pytorch_msssim import msssim, ssim
# from ltr.models.backbone import GRL
from ltr.models.backbone import Discriminator
from ltr.models.loss import Advloss
from ltr.models.loss import mmd_loss
from ltr.models.loss import Tripletloss
from ltr.models.backbone import SRMLayer
from ltr.models.backbone import Mask


class TransT(nn.Module):
    """ This is the TransT module that performs single object tracking """
    def __init__(self, backbone, featurefusion_network, num_classes):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See transt_backbone.py
            featurefusion_network: torch module of the featurefusion_network architecture, a variant of transformer.
                                   See featurefusion_network.py
            num_classes: number of object classes, always 1 for single object tracking
        """
        super().__init__()
        self.featurefusion_network = featurefusion_network
        hidden_dim = featurefusion_network.d_model
        self.class_embed = MLP(hidden_dim, hidden_dim, num_classes + 1, 3)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        # self.reduce_dim = nn.Conv2d(hidden_dim, hidden_dim//4, kernel_size=1)
        # self.reduce_dim = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        self.reduce_dim = nn.AdaptiveAvgPool2d((1, 1))
        # self.decoder = DecoderTr(hidden_dim*4)
        self.discriminator1 = Discriminator.Discriminator()
        self.discriminator2 = Discriminator.Discriminator()
        # self.srm = SRMLayer.SRMLayer(hidden_dim)
        self.feature_reweight = FeatureReweight(hidden_dim)
        self.feature_select = Selector_Network(hidden_dim)


    def forward(self, search, template, TIRTemplate, TIRSearch): #TIR's tem and ser is same
        """ The forward expects a NestedTensor, which consists of:
               - search.tensors: batched images, of shape [batch_size x 3 x H_search x W_search]
               - search.mask: a binary mask of shape [batch_size x H_search x W_search], containing 1 on padded pixels
               - template.tensors: batched images, of shape [batch_size x 3 x H_template x W_template]
               - template.mask: a binary mask of shape [batch_size x H_template x W_template], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits for all feature vectors.
                                Shape= [batch_size x num_vectors x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all feature vectors, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image.

        """
        if not isinstance(search, NestedTensor):
            search = nested_tensor_from_tensor(search)
        if not isinstance(template, NestedTensor):
            template = nested_tensor_from_tensor(template)
        # for getting TIR tempalte and search feature
        if not isinstance(TIRTemplate, NestedTensor):
            TIRimg_T = nested_tensor_from_tensor(TIRTemplate)
        TIR_T_feature, pos_template_tir = self.backbone(TIRimg_T)
        TIR_template_feature, mask_template_tir = TIR_T_feature[-1].decompose()

        if not isinstance(TIRSearch, NestedTensor):
            TIRimg_S = nested_tensor_from_tensor(TIRSearch)
        TIR_S_feature, pos_search_tir = self.backbone(TIRimg_S)
        TIR_search_feature, mask_search_tir = TIR_S_feature[-1].decompose()

        feature_search, pos_search = self.backbone(search)
        feature_template, pos_template = self.backbone(template)
        src_search, mask_search = feature_search[-1].decompose()
        assert mask_search is not None
        src_template, mask_template = feature_template[-1].decompose()
        assert mask_template is not None

        src_template = self.input_proj(src_template)
        src_search = self.input_proj(src_search)
        # TIR_template = self.srm(self.input_proj(TIR_template_feature))
        # TIR_search = self.srm(self.input_proj(TIR_search_feature))
        TIR_template = self.input_proj(TIR_template_feature)
        TIR_search = self.input_proj(TIR_search_feature)
        # src_template_adain = self.adain(src_template, TIR_template)
        # src_search_adain = self.adain(src_search, TIR_search)
        # hs_rgb = self.featurefusion_network(src_template_adain, mask_template, src_search_adain, mask_search, pos_template[-1], pos_search[-1])
        hs_rgb = self.featurefusion_network(src_template, mask_template, src_search, mask_search, pos_template[-1], pos_search[-1])

        hs_tir = self.featurefusion_network(TIR_template, mask_template_tir, TIR_search, mask_search_tir, pos_template_tir[-1], pos_search_tir[-1])

        outputs_class = self.class_embed(hs_rgb)
        outputs_coord = self.bbox_embed(hs_rgb).sigmoid()


        # reconstruct for RGB and TIR images
        # feature2, __ = feature_search[1].decompose()
        # feature3, __ = feature_search[0].decompose()
        # RGB_re_img = self.decoder(src_search, feature2, feature3)
        # #

        # featureTIR2, __ = TIRfeature[1].decompose()
        # featureTIR3, __ = TIRfeature[0].decompose()
        # TIR_re_img = self.decoder(TIR_src_feature, featureTIR2, featureTIR3)
        #
        # domain classifier after feature fusion
        x = torch.cat(((self.reduce_dim(hs_rgb.contiguous().view(-1, 256, 32, 32))).view(-1, 256), (self.reduce_dim(hs_tir.contiguous().view(-1, 256, 32, 32))).view(-1, 256)), 0)
        domain_predict_search = self.discriminator1(x)

        # domain classifier at last layer of feature backbone  for template
        # first to reweight feature at sample-level
        src_template,  TIR_template= self.feature_reweight(src_template, TIR_template)
        # second selected feature at channel-level using hard weights [0 or 1]
        select_vec = self.feature_select(src_template, TIR_template) #[1,256,16,16,]
        src_template_selected = torch.mul(src_template, select_vec)
        TIR_template_selected = torch.mul(TIR_template, select_vec)
        y = torch.cat((self.reduce_dim(src_template_selected).view(-1, 256), self.reduce_dim(TIR_template_selected).view(-1, 256)), 0)
        domain_predict_backbone = self.discriminator2(y)

        # y = torch.cat((self.reduce_dim(src_template).view(-1, 256), self.reduce_dim(TIR_template).view(-1, 256)), 0)
        # domain_predict_template = self.discriminator2(y)
        # out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1], 'src_template': self.reduce_dim(src_template_adain), 'src_search': self.reduce_dim(src_search_adain), 'TIR_template': self.reduce_dim(TIR_template), 'TIR_search': self.reduce_dim(TIR_search)}
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1],
               'src_template': src_template_selected, 'src_search': src_search,
               'TIR_template': TIR_template_selected, 'TIR_search': TIR_search,
               'domain_p_search': domain_predict_search, 'domain_p_backbone': domain_predict_backbone,
               'select_vec': select_vec}
        # 'domain_p_template': domain_predict_template
        return out

    def track(self, search):
        if not isinstance(search, NestedTensor):
            search = nested_tensor_from_tensor_2(search)
        features_search, pos_search = self.backbone(search)
        feature_template = self.zf
        pos_template = self.pos_template
        src_search, mask_search= features_search[-1].decompose()
        assert mask_search is not None
        src_template, mask_template = feature_template[-1].decompose()
        assert mask_template is not None
        hs = self.featurefusion_network(self.input_proj(src_template), mask_template, self.input_proj(src_search), mask_search, pos_template[-1], pos_search[-1])

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        return out

    def template(self, z):
        if not isinstance(z, NestedTensor):
            z = nested_tensor_from_tensor_2(z)
        zf, pos_template = self.backbone(z)
        self.zf = zf
        self.pos_template = pos_template

class SetCriterion(nn.Module):
    """ This class computes the loss for TransT.
    The process happens in two steps:
        1) we compute assignment between ground truth box and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, always be 1 for single object tracking.
            matcher: module able to compute a matching between target and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)



    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)]).to(device='cuda')

        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0).to(device='cuda')
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        giou, iou = box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes))
        giou = torch.diag(giou)
        iou = torch.diag(iou)
        loss_giou = 1 - giou
        iou = iou
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        losses['iou'] = iou.sum() / num_boxes
        return losses

    # loss for reconstruct
    # def loss_recon(self, outputs, targets, indices, num_boxes):
    #     """Compute the losses for image reconstruct
    #     """
    #     assert 'Re_RGB' or 'Re_TIR' in outputs
    #     out_re_RGB = outputs['Re_RGB']
    #     out_re_TIR = outputs['Re_TIR']
    #     losses = {}
    #     loss_RGB = (1 - msssim(out_re_RGB, targets[-1]['Re_RGB']))  # + F.mse_loss(out_re_RGB, targets[-1]['Re_RGB'])
    #     loss_TIR = (1 - msssim(out_re_TIR, targets[-1]['Re_TIR']))  # + F.mse_loss(out_re_TIR, targets[-1]['Re_TIR'])
    #     loss_recon = loss_RGB + loss_TIR
    #     losses['recon'] = loss_recon
    #     return losses

    # loss for domain classifier
    def loss_domain(self, outputs, targets, indices, num_boxes):
        src_template = outputs['src_template']
        TIR_template = outputs['TIR_template']
        src_search = outputs['src_search']
        TIR_search = outputs['TIR_search']
        domain_predict_search = outputs['domain_p_search']
        domain_predict_backbone = outputs['domain_p_backbone']

        d_s, d_t = domain_predict_search.chunk(2, dim=0)
        d_label_s = torch.ones((src_search.size(0), 1)).to(src_search.device)
        d_label_t = torch.zeros((TIR_search.size(0), 1)).to(TIR_search.device)
        loss_domain1 = 0.5 * (F.binary_cross_entropy(d_s, d_label_s, reduction='mean') + F.binary_cross_entropy(d_t, d_label_t, reduction='mean'))

        # for backbone feature at template
        d_s2, d_t2 = domain_predict_backbone.chunk(2, dim=0)
        d_label_s2 = torch.ones((src_template.size(0), 1)).to(src_template.device)
        d_label_t2 = torch.zeros((TIR_template.size(0), 1)).to(TIR_template.device)
        loss_domain2 = 0.5 * (F.binary_cross_entropy(d_s2, d_label_s2, reduction='mean') + F.binary_cross_entropy(d_t2, d_label_t2, reduction='mean'))
        loss_domain = loss_domain1 + loss_domain2
        losses = {'loss_domain': loss_domain}
        return losses

    # def loss_style(self, outputs, targets, indices, num_boxes):
    #     src_template = outputs['src_template']
    #     TIR_template = outputs['TIR_template']
    #     src_search = outputs['src_search']
    #     TIR_search = outputs['TIR_search']
    #     loss_mse = torch.nn.MSELoss()
    #     loss_style1 = loss_mse(gram(TIR_template), gram(src_template))
    #     loss_style2 = loss_mse(gram(TIR_search), gram(src_search))
    #     loss_style = loss_style1 + loss_style2
    #     losses = {'loss_style': loss_style}
    #     return losses

    def loss_select(self, outputs, targets, indices, num_boxes):
        src_template = outputs['src_template']
        TIR_template = outputs['TIR_template']
        src_search = outputs['src_search']
        TIR_search = outputs['TIR_search']
        select_vec = outputs['select_vec']
        loss_mmd = mmd_loss.MMD_loss()
        loss_select_mmd = loss_mmd(src_template, TIR_template)
        loss_regulation = F.mse_loss(select_vec, torch.ones_like(select_vec))
        loss_select = loss_select_mmd + loss_regulation
        losses = {'loss_select': loss_select}
        return losses


    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
            'domain': self.loss_domain,
            'select': self.loss_select
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the target
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes_pos = sum(len(t[0]) for t in indices)

        num_boxes_pos = torch.as_tensor([num_boxes_pos], dtype=torch.float, device=next(iter(outputs.values())).device)

        num_boxes_pos = torch.clamp(num_boxes_pos, min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes_pos))

        return losses

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


@model_constructor
def transt_resnet50(settings):
    num_classes = 1
    backbone_net = build_backbone(settings, backbone_pretrained=True)
    featurefusion_network = build_featurefusion_network(settings)
    model = TransT(
        backbone_net,
        featurefusion_network,
        num_classes=num_classes
    )
    device = torch.device(settings.device)
    model.to(device)
    return model

def transt_loss(settings):
    num_classes = 1
    matcher = build_matcher()
    weight_dict = {'loss_ce': 8.334, 'loss_bbox': 5}
    weight_dict['loss_giou'] = 2
    weight_dict['loss_domain'] = 0.1
    weight_dict['loss_select'] = 0.1
    losses = ['labels', 'boxes', 'domain', 'select']
    # losses = ['labels', 'boxes']
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=0.0625, losses=losses)
    device = torch.device(settings.device)
    criterion.to(device)
    return criterion


class DecoderT(nn.Module):
    def __init__(self, input_channel):
        super(DecoderT, self).__init__()
        # self.batchsize = batchsize
        self.input_channel = input_channel
        self.decoder_net = nn.Sequential(
            nn.Conv2d(in_channels=self.input_channel, out_channels=self.input_channel // 4, kernel_size=1),
            nn.ReLU(),
            # x 2
            nn.ConvTranspose2d(in_channels=self.input_channel // 4, out_channels=self.input_channel // 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.input_channel // 8),
            nn.ReLU(),
            # x 2
            nn.ConvTranspose2d(in_channels=self.input_channel // 8, out_channels=self.input_channel // 16, kernel_size=4,
                               stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.input_channel // 16),
            nn.ReLU(),
            # x 2
            nn.ConvTranspose2d(in_channels=self.input_channel // 16, out_channels=self.input_channel // 32, kernel_size=4,
                               stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.input_channel // 32),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.input_channel // 32, out_channels=3, kernel_size=1),
            nn.Tanh()
        )

    def forward(self, x):
        out = self.decoder_net(x)
        return out


class DecoderTr(nn.Module):
    def __init__(self, input_channel):
        super(DecoderTr, self).__init__()
        # self.batchsize = batchsize
        self.input_channel = input_channel
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=self.input_channel, out_channels=self.input_channel // 4, kernel_size=1), nn.ReLU())
        self.upsample1 = upsample(self.input_channel // 4, self.input_channel // 2)
        self.upsample2 = upsample(self.input_channel, self.input_channel // 4)
        self.upsample3 = upsample(self.input_channel//2, self.input_channel//32)
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=self.input_channel // 32, out_channels=3, kernel_size=1), nn.Tanh())

    def forward(self, x3, x2, x1):
        conv1_out = self.conv1(x3)
        up_sample1_out = self.upsample1(conv1_out)
        cat_1_out = torch.cat((up_sample1_out, x2), 1)
        up_sample2_out = self.upsample2(cat_1_out)
        cat_2_out = torch.cat((up_sample2_out, x1), 1)
        up_sample3_out = self.upsample3(cat_2_out)
        out = self.conv2(up_sample3_out)
        return out

# Up sampling module
def upsample(ch_coarse, ch_fine):
    return nn.Sequential(
        nn.ConvTranspose2d(ch_coarse, ch_fine, 4, 2, 1, bias=False),
        nn.ReLU()
    )

# Calculate Gram matrix (G = FF^T)
def gram(x):
    (bs, ch, h, w) = x.size()
    f = x.view(bs, ch, w*h)
    f_T = f.transpose(1, 2)
    G = f.bmm(f_T) / (ch * h * w)
    return G


class FeatureReweight(nn.Module):
    def __init__(self, input_channel):
        super(FeatureReweight, self).__init__()
        # self.batchsize = batchsize
        self.input_channel = input_channel
        self.sigmoid = nn.Sigmoid()
        # self.sum = torch.tensor(0.0).cuda()
        # self.num = 0
    def forward(self, rgb_feat, tir_feat):
        rgb_feat_p = torch.nn.functional.adaptive_avg_pool2d(rgb_feat, (1, 1))
        tir_feat_p = torch.nn.functional.adaptive_avg_pool2d(tir_feat, (1, 1))
        correlation = torch.bmm(rgb_feat_p.view(-1, 1, self.input_channel), tir_feat_p.view(-1, self.input_channel, 1))
        cor_re = self.sigmoid(correlation.view(-1, 1, 1, 1))
        # self.sum +=  torch.mean(cor_re)
        # self.num += 1
        # print('sum={}'.format(self.sum.item()))
        # print('num={}'.format(self.num))
        # print('avg={}'.format(self.sum/self.num))

        rgb_feat_reweight = torch.mul(rgb_feat, cor_re)
        tir_feat_reweight = torch.mul(tir_feat, cor_re)
        return rgb_feat_reweight, tir_feat_reweight

class Selector_Network(nn.Module):
    """
    Selector Network.
    """
    def __init__(self, input_channel):
        super(Selector_Network, self).__init__()
        self.input_channel = input_channel
        self.mask_c = Mask.Mask_c(self.input_channel, self.input_channel)
        #self.mask_s = Mask.Mask_s() no use?
        # self.fc1 = nn.Linear(in_features=self.input_channel, out_features=self.input_channel)
        # self.fc2 = nn.Linear(in_features=self.input_channel, out_features=self.input_channel)

    def forward(self, rgb_feat, tir_feat):
        select_vec = self.mask_c(1 - abs(rgb_feat-tir_feat))
        # select_vec_flat = select_vec.view(-1)  # 展平张量
        #
        # # 统计值为 0 的数量
        # num_zeros = (select_vec_flat == 0).sum().item()
        #
        # # 统计值为 1 的数量
        # num_ones = (select_vec_flat == 1).sum().item()
        #
        # # 计算总数
        # total_elements = select_vec.numel()
        #
        # # 计算值为 1 的占比
        # ratio_ones = num_ones / total_elements
        #
        # # 打印结果
        # print(f'Number of 0s: {num_zeros}')
        # print(f'Number of 1s: {num_ones}')
        # print(f'Ratio of 1s: {ratio_ones:.2f}')
        return select_vec

def Entropy(input_):
    # Obtain entropy loss value
    bs = input_.size(0)
    entropy = -input_ * torch.log(input_ + 1e-5)
    entropy = torch.sum(entropy, dim=1)
    return entropy


