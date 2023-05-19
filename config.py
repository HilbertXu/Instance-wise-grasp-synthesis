import os
import numpy as np
import torch
import torch.distributed as dist
from functools import partial

from utils.gr_augmentation import gr_train_aug, gr_val_aug, gr_train_aug_jacquard, gr_val_aug_jacquard

os.makedirs('results/images', exist_ok=True)
os.makedirs('results/videos', exist_ok=True)
os.makedirs('weights/', exist_ok=True)
os.makedirs('tensorboard_log/', exist_ok=True)


GRASPNET_DATASET = ['background', 'cracker box', 'sugar box', 'tomato soup can', 'mustard bottle', 'potted meat can', 'banana', 'bowl', 'mug', 'power drill',
                    'scissors', 'chips can', 'strawberry', 'apple', 'lemon', 'peach', 'pear', 'orange', 'plum', 'knife', 'phillips screw driver',
                    'flat screwdriver', 'racquetball', 'b cups', 'd cups', 'toy a', 'toy c', 'toy d', 'toy f',
                    'toy h', 'toy i', 'toy j', 'toy k', 'padlock', 'dragon', 'secret', 'cleansing foam', 'wash soup', 'skincare mouth rinse',
                    'dabao sod', 'soap box', 'kispa cleanser', 'tooth paste', 'nivea', 'marker', 'hosjam', 'pitcher cap', 'dish', 'white mouse', 'camel',
                    'deer', 'zebra', 'big elephant', 'rhino', 'small elephant', 'monkey', 'giraffe', 'gorilla', 'weiquan', 'charlie box', 'soap', 'black mouse',
                    'dabao face wash', 'pantene', 'head shoulder supreme', 'thera med', 'dove', 'head shoulder care', 'lion', 'coconut juice', 'hippo', 'tape',
                    'rubiks cube', 'peeler cover', 'peeler', 'ice cube mould', 'bar clamp', 'climbing hold', 'endstop holder', 'gear box', 'mount1', 'mount2', 'nozzle',
                    'part1', 'part3', 'pawn', 'pipe connector', 'turbine housing', 'vase']


JACQUARD_DATASET = ['background', 'object']



from OCID_class_dict import cls_list as OCID_GRASP_DATASET

PER_CLASS_MAX_GRASP_WIDTH = [65, 83, 45, 64, 43, 23, 140, 62, 29, 107, 147, 70, 34, 103, 112, 118, 101, 70, 41, 51, 80, 61, 77, 74, 57, 56, 74, 42, 54, 49, 75]


# from graspnet_dataset import GraspNetDataset
from ocid_grasp import OCIDGraspDataset
from jacquard_dataset import JacquardGraspDataset
from osgd_dataset import OSGDDataset


class res101_osgd:
    def __init__(self, args):
        self.mode = args.mode
        self.cuda = args.cuda
        self.gpu_id = args.gpu_id
        assert args.img_size % 32 == 0, f'Img_size must be divisible by 32, got {args.img_size}.'
        self.img_size = args.img_size
        self.class_names = OCID_GRASP_DATASET
        self.num_classes = 11
        # self.class_names = CORNELL_CLASSES
        # self.num_classes = len(CORNELL_CLASSES)
        self.continuous_id = COCO_LABEL_MAP
        self.scales = [int(self.img_size / 544 * aa) for aa in (24, 48, 96, 192, 384)]
        self.aspect_ratios = [1, 1 / 2, 2]

        if self.mode == 'train':
            self.weight = args.resume if args.resume else 'weights/backbone_res101.pth'
        else:
            self.weight = args.weight

        self.data_root = '/home/hilbertxu/dataset'

        self.summary_comment = "OSGD"
        self.weight_dir = "CoGr-OSGD"
        os.makedirs('weights/{}'.format(self.weight_dir), exist_ok=True)

        self.dataset = OSGDDataset(root_path="/home/puzek/sdb/dataset/OSGD_IMG/taskoritv0", transform=None, mode="train")
        self.val_dataset = None


        if self.mode == 'train':
            self.train_imgs = self.data_root + 'coco2017/train2017/'
            self.train_ann = self.data_root + 'coco2017/annotations/instances_train2017.json'
            self.train_bs = args.train_bs
            self.bs_per_gpu = args.bs_per_gpu
            self.val_interval = args.val_interval

            self.bs_factor = self.train_bs / 8
            self.lr = 0.001 * self.bs_factor
            self.warmup_init = self.lr * 0.1
            self.warmup_until = 500  # If adapted with bs_factor, inifinte loss may appear.
            self.lr_steps = tuple([int(aa / self.bs_factor) for aa in (0, 280000, 560000, 620000, 680000)])

            self.pos_iou_thre = 0.5
            self.neg_iou_thre = 0.4

            self.conf_alpha = 1
            self.bbox_alpha = 1.5
            self.mask_alpha = 6.125
            self.grasp_alpha = 6.125
            self.aff_alpha = 6.125
            self.semantic_alpha = 1

            # The max number of masks to train for one image.
            self.masks_to_train = 100

        if self.mode in ('train', 'val'):
            self.val_imgs = self.data_root + 'coco2017/val2017/'
            self.val_ann = self.data_root + 'coco2017/annotations/instances_val2017.json'
            self.val_bs = 1
            self.val_num = args.val_num
            self.coco_api = args.coco_api

        self.traditional_nms = args.traditional_nms
        self.nms_score_thre = 0.05
        self.nms_iou_thre = 0.25
        self.top_k = 200
        self.max_detections = 100

        if self.mode == 'detect':
            for k, v in vars(args).items():
                self.__setattr__(k, v)

    def print_cfg(self):
        print()
        print('-' * 30 + self.__class__.__name__ + '-' * 30)
        for k, v in vars(self).items():
            if k not in ('continuous_id', 'data_root', 'cfg'):
                print(f'{k}: {v}')
        print()


class res101_jacquard:
    def __init__(self, args):
        self.mode = args.mode
        self.cuda = args.cuda
        self.gpu_id = args.gpu_id
        assert args.img_size % 32 == 0, f'Img_size must be divisible by 32, got {args.img_size}.'
        self.img_size = args.img_size
        self.class_names = JACQUARD_DATASET
        self.num_classes = len(JACQUARD_DATASET)
        # self.class_names = CORNELL_CLASSES
        # self.num_classes = len(CORNELL_CLASSES)
        self.continuous_id = COCO_LABEL_MAP
        self.scales = [int(self.img_size / 544 * aa) for aa in (24, 48, 96, 192, 384)]
        self.aspect_ratios = [1, 1 / 2, 2]


        self.summary_comment = "JACQUARD"
        self.weight_dir = "CoGr-JACQUARD"
        os.makedirs('weights/{}'.format(self.weight_dir), exist_ok=True)

        self.dataset = JacquardGraspDataset(
                        "/home/puzek/sdb/dataset/JACQUARD/jacquard",
                        origin_size=1024, 
                        target_size=544, 
                        transform=gr_train_aug_jacquard,
                        multi_obj_aug=True,
                        mode="train"
                    )
        self.val_dataset = JacquardGraspDataset(
                        "/home/puzek/sdb/dataset/JACQUARD/jacquard",
                        origin_size=1024, 
                        target_size=544, 
                        transform=gr_val_aug_jacquard,
                        multi_obj_aug=False,
                        mode="test"
                    )

        if self.mode == 'train':
            self.weight = args.resume if args.resume else 'weights/backbone_res101.pth'
        else:
            self.weight = args.weight


        if self.mode == 'train':
            self.train_bs = args.train_bs
            self.bs_per_gpu = args.bs_per_gpu
            self.val_interval = args.val_interval

            self.bs_factor = self.train_bs / 8
            self.lr = 0.001 * self.bs_factor
            self.warmup_init = self.lr * 0.1
            self.warmup_until = 500  # If adapted with bs_factor, inifinte loss may appear.
            self.lr_steps = tuple([int(aa / self.bs_factor) for aa in (0, 280000, 560000, 620000, 680000)])

            self.pos_iou_thre = 0.5
            self.neg_iou_thre = 0.4

            self.conf_alpha = 1
            self.bbox_alpha = 2.5
            self.mask_alpha = 6.125
            self.grasp_alpha = 6.125
            self.semantic_alpha = 1

            # The max number of masks to train for one image.
            self.masks_to_train = 100

        if self.mode in ('train', 'val'):
            self.val_bs = 1
            self.val_num = args.val_num
            self.coco_api = args.coco_api

        self.traditional_nms = args.traditional_nms
        self.nms_score_thre = 0.05
        self.nms_iou_thre = 0.5
        self.top_k = 200
        self.max_detections = 100

        if self.mode == 'detect':
            for k, v in vars(args).items():
                self.__setattr__(k, v)

    def print_cfg(self):
        print()
        print('-' * 30 + self.__class__.__name__ + '-' * 30)
        for k, v in vars(self).items():
            if k not in ('continuous_id', 'data_root', 'cfg'):
                print(f'{k}: {v}')
        print()



class res101_ocid_grasp_only:
    def __init__(self, args):
        self.mode = args.mode
        self.cuda = args.cuda
        self.gpu_id = args.gpu_id
        assert args.img_size % 32 == 0, f'Img_size must be divisible by 32, got {args.img_size}.'
        self.img_size = args.img_size
        self.class_names = OCID_GRASP_DATASET
        self.num_classes = len(OCID_GRASP_DATASET)
        # self.class_names = CORNELL_CLASSES
        # self.num_classes = len(CORNELL_CLASSES)
        self.continuous_id = COCO_LABEL_MAP
        self.scales = [int(self.img_size / 544 * aa) for aa in (24, 48, 96, 192, 384)]
        self.aspect_ratios = [1, 1 / 2, 2]

        if self.mode == 'train':
            self.weight = args.resume if args.resume else 'weights/backbone_res101.pth'
        else:
            self.weight = args.weight

        self.data_root = '/home/hilbertxu/dataset'

        self.summary_comment = "OCID"
        self.weight_dir = "CoGr-OCID-Grasp-ONLY"
        os.makedirs('weights/{}'.format(self.weight_dir), exist_ok=True)

        self.dataset = OCIDGraspDataset(
                        "/home/puzek/sdb/dataset/OCID_grasp",
                        "training_0",
                        transform=partial(gr_train_aug, self.img_size)
                    )
        self.val_dataset = OCIDGraspDataset(
                        "/home/puzek/sdb/dataset/OCID_grasp",
                        "validation_0",
                        mode='test',
                        transform=partial(gr_val_aug, self.img_size)
                    )


        if self.mode == 'train':
            self.train_imgs = self.data_root + 'coco2017/train2017/'
            self.train_ann = self.data_root + 'coco2017/annotations/instances_train2017.json'
            self.train_bs = args.train_bs
            self.bs_per_gpu = args.bs_per_gpu
            self.val_interval = args.val_interval

            self.bs_factor = self.train_bs / 8
            self.lr = 0.001 * self.bs_factor
            self.warmup_init = self.lr * 0.1
            self.warmup_until = 500  # If adapted with bs_factor, inifinte loss may appear.
            self.lr_steps = tuple([int(aa / self.bs_factor) for aa in (0, 280000, 560000, 620000, 680000)])

            self.pos_iou_thre = 0.5
            self.neg_iou_thre = 0.4

            self.conf_alpha = 1
            self.bbox_alpha = 1.5
            self.mask_alpha = 6.125
            self.grasp_alpha = 6.125
            self.semantic_alpha = 1

            # The max number of masks to train for one image.
            self.masks_to_train = 100

        if self.mode in ('train', 'val'):
            self.val_imgs = self.data_root + 'coco2017/val2017/'
            self.val_ann = self.data_root + 'coco2017/annotations/instances_val2017.json'
            self.val_bs = 1
            self.val_num = args.val_num
            self.coco_api = args.coco_api

        self.traditional_nms = args.traditional_nms
        self.nms_score_thre = 0.05
        self.nms_iou_thre = 0.5
        self.top_k = 200
        self.max_detections = 100

        if self.mode == 'detect':
            for k, v in vars(args).items():
                self.__setattr__(k, v)

    def print_cfg(self):
        print()
        print('-' * 30 + self.__class__.__name__ + '-' * 30)
        for k, v in vars(self).items():
            if k not in ('continuous_id', 'data_root', 'cfg'):
                print(f'{k}: {v}')
        print()


class res101_ocid_rgb:
    def __init__(self, args):
        self.mode = args.mode
        self.cuda = args.cuda
        self.gpu_id = args.gpu_id
        assert args.img_size % 32 == 0, f'Img_size must be divisible by 32, got {args.img_size}.'
        self.img_size = args.img_size
        self.class_names = OCID_GRASP_DATASET
        self.num_classes = len(OCID_GRASP_DATASET)
        # self.class_names = CORNELL_CLASSES
        # self.num_classes = len(CORNELL_CLASSES)
        self.continuous_id = COCO_LABEL_MAP
        self.scales = [int(self.img_size / 544 * aa) for aa in (24, 48, 96, 192, 384)]
        self.aspect_ratios = [1, 1 / 2, 2]

        if self.mode == 'train':
            self.weight = args.resume if args.resume else 'weights/backbone_res101.pth'
        else:
            self.weight = args.weight

        self.data_root = '/home/hilbertxu/dataset'

        self.summary_comment = "OCID-RGB"
        self.weight_dir = "CoGr-OCID-RGB"
        os.makedirs('weights/{}'.format(self.weight_dir), exist_ok=True)

        self.dataset = OCIDGraspDataset(
                        "/home/puzek/sdb/dataset/OCID_grasp",
                        "training_0",
                        transform=partial(gr_train_aug, self.img_size)
                    )
        self.val_dataset = OCIDGraspDataset(
                        "/home/puzek/sdb/dataset/OCID_grasp",
                        "validation_0",
                        mode='test',
                        transform=partial(gr_val_aug, self.img_size)
                    )


        if self.mode == 'train':
            self.train_imgs = self.data_root + 'coco2017/train2017/'
            self.train_ann = self.data_root + 'coco2017/annotations/instances_train2017.json'
            self.train_bs = args.train_bs
            self.bs_per_gpu = args.bs_per_gpu
            self.val_interval = args.val_interval

            self.bs_factor = self.train_bs / 8
            self.lr = 0.001 * self.bs_factor
            self.warmup_init = self.lr * 0.1
            self.warmup_until = 500  # If adapted with bs_factor, inifinte loss may appear.
            self.lr_steps = tuple([int(aa / self.bs_factor) for aa in (0, 280000, 560000, 620000, 680000)])

            self.pos_iou_thre = 0.5
            self.neg_iou_thre = 0.4

            self.conf_alpha = 1
            self.bbox_alpha = 1.5
            self.mask_alpha = 6.125
            self.grasp_alpha = 6.125
            self.semantic_alpha = 1

            # The max number of masks to train for one image.
            self.masks_to_train = 100

        if self.mode in ('train', 'val'):
            self.val_imgs = self.data_root + 'coco2017/val2017/'
            self.val_ann = self.data_root + 'coco2017/annotations/instances_val2017.json'
            self.val_bs = 1
            self.val_num = args.val_num
            self.coco_api = args.coco_api

        self.traditional_nms = args.traditional_nms
        self.nms_score_thre = 0.05
        self.nms_iou_thre = 0.5
        self.top_k = 200
        self.max_detections = 100

        if self.mode == 'detect':
            for k, v in vars(args).items():
                self.__setattr__(k, v)

    def print_cfg(self):
        print()
        print('-' * 30 + self.__class__.__name__ + '-' * 30)
        for k, v in vars(self).items():
            if k not in ('continuous_id', 'data_root', 'cfg'):
                print(f'{k}: {v}')
        print()


class res101_ocid:
    def __init__(self, args):
        self.mode = args.mode
        self.cuda = args.cuda
        self.gpu_id = args.gpu_id
        assert args.img_size % 32 == 0, f'Img_size must be divisible by 32, got {args.img_size}.'
        self.img_size = args.img_size
        self.class_names = OCID_GRASP_DATASET
        self.num_classes = len(OCID_GRASP_DATASET)
        # self.class_names = CORNELL_CLASSES
        # self.num_classes = len(CORNELL_CLASSES)
        self.continuous_id = COCO_LABEL_MAP
        self.scales = [int(self.img_size / 544 * aa) for aa in (24, 48, 96, 192, 384)]
        self.aspect_ratios = [1, 1 / 2, 2]

        if self.mode == 'train':
            self.weight = args.resume if args.resume else 'weights/backbone_res101.pth'
        else:
            self.weight = args.weight

        self.data_root = '/home/hilbertxu/dataset'

        self.summary_comment = "OCID"
        self.weight_dir = "CoGr-OCID-RGBD"
        os.makedirs('weights/{}'.format(self.weight_dir), exist_ok=True)

        self.dataset = OCIDGraspDataset(
                        "/home/puzek/sdb/dataset/OCID_grasp",
                        "training_0",
                        transform=partial(gr_train_aug, self.img_size)
                    )
        self.val_dataset = OCIDGraspDataset(
                        "/home/puzek/sdb/dataset/OCID_grasp",
                        "validation_0",
                        mode='test',
                        transform=partial(gr_val_aug, self.img_size)
                    )


        if self.mode == 'train':
            self.train_imgs = self.data_root + 'coco2017/train2017/'
            self.train_ann = self.data_root + 'coco2017/annotations/instances_train2017.json'
            self.train_bs = args.train_bs
            self.bs_per_gpu = args.bs_per_gpu
            self.val_interval = args.val_interval

            self.bs_factor = self.train_bs / 8
            self.lr = 0.001 * self.bs_factor
            self.warmup_init = self.lr * 0.1
            self.warmup_until = 500  # If adapted with bs_factor, inifinte loss may appear.
            self.lr_steps = tuple([int(aa / self.bs_factor) for aa in (0, 280000, 560000, 620000, 680000)])

            self.pos_iou_thre = 0.5
            self.neg_iou_thre = 0.4

            self.conf_alpha = 1
            self.bbox_alpha = 1.5
            self.mask_alpha = 6.125
            self.grasp_alpha = 6.125
            self.semantic_alpha = 1

            # The max number of masks to train for one image.
            self.masks_to_train = 100

        if self.mode in ('train', 'val'):
            self.val_imgs = self.data_root + 'coco2017/val2017/'
            self.val_ann = self.data_root + 'coco2017/annotations/instances_val2017.json'
            self.val_bs = 1
            self.val_num = args.val_num
            self.coco_api = args.coco_api

        self.traditional_nms = args.traditional_nms
        self.nms_score_thre = 0.05
        self.nms_iou_thre = 0.5
        self.top_k = 200
        self.max_detections = 100

        if self.mode == 'detect':
            for k, v in vars(args).items():
                self.__setattr__(k, v)

    def print_cfg(self):
        print()
        print('-' * 30 + self.__class__.__name__ + '-' * 30)
        for k, v in vars(self).items():
            if k not in ('continuous_id', 'data_root', 'cfg'):
                print(f'{k}: {v}')
        print()


def get_config(args, mode):
    args.cuda = torch.cuda.is_available()
    # args.cuda = False
    args.mode = mode

    if args.cuda:
        args.gpu_id = os.environ.get('CUDA_VISIBLE_DEVICES') if os.environ.get('CUDA_VISIBLE_DEVICES') else '0'
        if args.mode == 'train':
            torch.cuda.set_device(args.local_rank)
            dist.init_process_group(backend='nccl', init_method='env://')

            # Only launched by torch.distributed.launch, 'WORLD_SIZE' can be add to environment variables.
            num_gpus = int(os.environ['WORLD_SIZE'])
            assert args.train_bs % num_gpus == 0, 'Total training batch size must be divisible by GPU number.'
            args.bs_per_gpu = int(args.train_bs / num_gpus)
        else:
            assert args.gpu_id.isdigit(), f'Only one GPU can be used in val/detect mode, got {args.gpu_id}.'
    else:
        args.gpu_id = None
        if args.mode == 'train':
            args.bs_per_gpu = args.train_bs
            print('\n-----No GPU found, training on CPU.-----')
        else:
            print('\n-----No GPU found, validate on CPU.-----')

    cfg = globals()[args.cfg](args)

    if not args.cuda or args.mode != 'train':
        cfg.print_cfg()
    elif dist.get_rank() == 0:
        cfg.print_cfg()

    return cfg
