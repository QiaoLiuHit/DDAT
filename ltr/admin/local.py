class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/data/liqiao/code/TransT_MY'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = self.workspace_dir + '/tensorboard/'    # Directory for tensorboard files.
        self.lasot_dir = '/data/liqiao/dataset/LaSOT/lasot'
        self.got10k_dir = '/data/liqiao/dataset/GOT-10k/train'
        self.trackingnet_dir = '/data/liqiao/dataset/trackingnet'
        self.coco_dir = '/data/liqiao/dataset/coco'
        self.lvis_dir = ''
        self.sbd_dir = ''
        #self.imagenet_dir = '/home/liqiao/dataset/ILSVRC2015_VID'
       # self.lsotbtir_dir = '/media/qiao/data/LSOTB-TIR_TrainingData'
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
