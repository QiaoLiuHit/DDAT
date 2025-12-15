import torch
from ltr.dataset import Lasot, MSCOCOSeq, Got10k, TrackingNet, ImagenetVID, LSOTBTIR
from ltr.dataset.TIR_dataset_pair import TIR_dataset_pair
from ltr.data import processing, sampler, LTRLoader
import ltr.models.tracking.transt as transt_models
from ltr import actors
from ltr.trainers import LTRTrainer
import ltr.data.transforms as tfm
from ltr import MultiGPU
import ltr.admin.loading as ltr_loading
from torch.utils.data import DataLoader
from ltr.tllib.modules.domain_discriminator import DomainDiscriminator


def run(settings):
    # Most common settings are assigned in the settings struct
    settings.device = 'cuda'
    settings.description = 'TransT with default settings.'
    settings.batch_size = 32
    settings.num_workers = 4
    settings.multi_gpu = True
    settings.print_interval = 1
    # settings.normalize_mean = [0.485, 0.456, 0.406]
    # settings.normalize_std = [0.229, 0.224, 0.225]
    settings.normalize_mean = [0.5, 0.5, 0.5]
    settings.normalize_std = [0.5, 0.5, 0.5]
    settings.search_area_factor = 4.0
    settings.template_area_factor = 2.0
    settings.search_feature_sz = 32
    settings.template_feature_sz = 16
    settings.search_sz = settings.search_feature_sz * 8
    settings.temp_sz = settings.template_feature_sz * 8
    settings.center_jitter_factor = {'search': 3, 'template': 0}
    settings.scale_jitter_factor = {'search': 0.25, 'template': 0}

    # Transformer
    settings.position_embedding = 'sine'
    settings.hidden_dim = 256
    settings.dropout = 0.1
    settings.nheads = 8
    settings.dim_feedforward = 2048
    settings.featurefusion_layers = 4

    # Train datasets
    # lsotb_tir = LSOTBTIR(settings.env.lsotbtir_dir)
    #vid_train = ImagenetVID(settings.env.imagenet_dir)
    lasot_train = Lasot(settings.env.lasot_dir, split='train')
    got10k_train = Got10k(settings.env.got10k_dir, split='train')
    trackingnet_train = TrackingNet(settings.env.trackingnet_dir, set_ids=[0,1,2,3,4,5,6,7,8,9,10,11])#list(range(12))
    coco_train = MSCOCOSeq(settings.env.coco_dir)

    # TIR Training Dataset
    # root_path = '/media/qiao/data/TIR-nolabel'
    root_path = '/data/liqiao/dataset/sam_crop511'
    tir_nolabel_dataset = TIR_dataset_pair(root_path)

    # The joint augmentation transform, that is applied to the pairs jointly
    # transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05))
    # transform_joint = tfm.Transform(tfm.ToGrayscale())

    # The augmentation transform applied to the training set (individually to each image in the pair)
    # transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2), tfm.Normalize(mean=settings.normalize_mean, std=settings.normalize_std))
    # tfm.ToGrayscale(probability=1),
    transform_train = tfm.Transform(tfm.ToGrayscale(),
                                    tfm.ToTensor(),
                                    tfm.Normalize(mean=settings.normalize_mean, std=settings.normalize_std))

    # Data processing to do on the training pairs
    data_processing_train = processing.TransTProcessing(search_area_factor=settings.search_area_factor,
                                                      template_area_factor = settings.template_area_factor,
                                                      search_sz=settings.search_sz,
                                                      temp_sz=settings.temp_sz,
                                                      center_jitter_factor=settings.center_jitter_factor,
                                                      scale_jitter_factor=settings.scale_jitter_factor,
                                                      mode='sequence',
                                                      transform=transform_train,
                                                      joint_transform=None)

    # The sampler for training
    # dataset_train = sampler.TransTSampler([lasot_train], [1],
    #                                       samples_per_epoch=5000 * settings.batch_size, max_gap=100,
    #                                       processing=data_processing_train)
    dataset_train = sampler.TransTSampler([got10k_train,lasot_train,trackingnet_train, coco_train], [1,1,1,1],#[1, 1, 1, 1],#got10k_train,lasot_train,trackingnet_train, coco_train
                               samples_per_epoch=6000*settings.batch_size, max_gap=100, processing=data_processing_train)
    # The loader for training
    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=settings.batch_size, num_workers=settings.num_workers,
                             shuffle=True, drop_last=True, stack_dim=0)

    # TIR images dataloader
    loader_TIR_train = LTRLoader('train', tir_nolabel_dataset, training=True, batch_size=settings.batch_size, num_workers=settings.num_workers,
                                 shuffle=True, drop_last=True, stack_dim=0)

    # Create network and actor
    model = transt_models.transt_resnet50(settings)

    # use pre-trained transt parameters
    model_dict = model.state_dict()
    net_path = '/data/liqiao/code/TransT_MY/pytracking/networks/transt.pth'
    pretrained_net = torch.load(net_path)
    pretrained_dict = pretrained_net["net"]
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)

    # Wrap the network for multi GPU training
    if settings.multi_gpu:
        model = MultiGPU(model, dim=0)

    objective = transt_models.transt_loss(settings)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    actor = actors.TranstActor(net=model, objective=objective)

    # Optimizer
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": 1e-5,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=1e-5,
                                  weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50)#每经过50个epochs，学习率会按预定的方式减少。

    # Create trainer
    trainer = LTRTrainer(actor, [loader_train, loader_TIR_train], optimizer, settings, lr_scheduler)

    # Run training (set fail_safe=False if you are debugging)
    trainer.train(50, load_latest=True, fail_safe=True)
