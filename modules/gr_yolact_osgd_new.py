import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from modules.resnet import ResNet
from utils.box_utils import match, crop, ones_crop, make_anchors
from modules.swin_transformer import SwinTransformer
import pdb


class PredictionModule(nn.Module):
    def __init__(self, cfg, coef_dim=32):
        super().__init__()
        self.num_classes = cfg.num_classes
        self.coef_dim = coef_dim
        self.gr_coef_dim = coef_dim / 2

        self.upfeature = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                       nn.ReLU(inplace=True))
        self.bbox_layer = nn.Conv2d(256, len(cfg.aspect_ratios) * 4, kernel_size=3, padding=1)
        self.conf_layer = nn.Conv2d(256, len(cfg.aspect_ratios) * self.num_classes, kernel_size=3, padding=1)
        

        self.gr_pos_coef_layer = nn.Sequential(nn.Conv2d(256, len(cfg.aspect_ratios) * self.coef_dim,
                                                  kernel_size=3, padding=1),
                                        nn.Tanh())
        
        self.gr_sin_coef_layer = nn.Sequential(nn.Conv2d(256, len(cfg.aspect_ratios) * self.coef_dim,
                                                  kernel_size=3, padding=1),
                                        nn.Tanh())

        self.gr_cos_coef_layer = nn.Sequential(nn.Conv2d(256, len(cfg.aspect_ratios) * self.coef_dim,
                                                  kernel_size=3, padding=1),
                                        nn.Tanh())
        
        self.gr_wid_coef_layer = nn.Sequential(nn.Conv2d(256, len(cfg.aspect_ratios) * self.coef_dim,
                                                  kernel_size=3, padding=1),
                                        nn.Tanh())

        self.gr_act_coef_layer1 = nn.Sequential(nn.Conv2d(256, len(cfg.aspect_ratios) * self.coef_dim,
                                                  kernel_size=3, padding=1),
                                        nn.Tanh())
        self.gr_act_coef_layer2 = nn.Sequential(nn.Conv2d(256, len(cfg.aspect_ratios) * self.coef_dim,
                                                  kernel_size=3, padding=1),
                                        nn.Tanh())
        self.gr_act_coef_layer3 = nn.Sequential(nn.Conv2d(256, len(cfg.aspect_ratios) * self.coef_dim,
                                                  kernel_size=3, padding=1),
                                        nn.Tanh())
        self.gr_act_coef_layer4 = nn.Sequential(nn.Conv2d(256, len(cfg.aspect_ratios) * self.coef_dim,
                                                  kernel_size=3, padding=1),
                                        nn.Tanh())
        self.gr_act_coef_layer5 = nn.Sequential(nn.Conv2d(256, len(cfg.aspect_ratios) * self.coef_dim,
                                                  kernel_size=3, padding=1),
                                        nn.Tanh())
        self.gr_act_coef_layer6 = nn.Sequential(nn.Conv2d(256, len(cfg.aspect_ratios) * self.coef_dim,
                                                  kernel_size=3, padding=1),
                                        nn.Tanh())
        self.gr_act_coef_layer7 = nn.Sequential(nn.Conv2d(256, len(cfg.aspect_ratios) * self.coef_dim,
                                                  kernel_size=3, padding=1),
                                        nn.Tanh())
        self.gr_act_coef_layer8 = nn.Sequential(nn.Conv2d(256, len(cfg.aspect_ratios) * self.coef_dim,
                                                  kernel_size=3, padding=1),
                                        nn.Tanh())
        self.gr_act_coef_layer9 = nn.Sequential(nn.Conv2d(256, len(cfg.aspect_ratios) * self.coef_dim,
                                                  kernel_size=3, padding=1),
                                        nn.Tanh())
        self.gr_act_coef_layer10 = nn.Sequential(nn.Conv2d(256, len(cfg.aspect_ratios) * self.coef_dim,
                                                  kernel_size=3, padding=1),
                                        nn.Tanh())
        self.gr_act_coef_layer11 = nn.Sequential(nn.Conv2d(256, len(cfg.aspect_ratios) * self.coef_dim,
                                                  kernel_size=3, padding=1),
                                        nn.Tanh())


    def forward(self, x):
        x = self.upfeature(x)
        conf = self.conf_layer(x).permute(0, 2, 3, 1).reshape(x.size(0), -1, self.num_classes)
        box = self.bbox_layer(x).permute(0, 2, 3, 1).reshape(x.size(0), -1, 4)

        gr_pos_coef = self.gr_pos_coef_layer(x).permute(0, 2, 3, 1).reshape(x.size(0), -1, self.coef_dim)
        gr_sin_coef = self.gr_sin_coef_layer(x).permute(0, 2, 3, 1).reshape(x.size(0), -1, self.coef_dim)
        gr_cos_coef = self.gr_cos_coef_layer(x).permute(0, 2, 3, 1).reshape(x.size(0), -1, self.coef_dim)
        gr_wid_coef = self.gr_wid_coef_layer(x).permute(0, 2, 3, 1).reshape(x.size(0), -1, self.coef_dim)
        act_coef1 = self.gr_act_coef_layer1(x).permute(0, 2, 3, 1).reshape(x.size(0), -1, self.coef_dim)
        act_coef2 = self.gr_act_coef_layer2(x).permute(0, 2, 3, 1).reshape(x.size(0), -1, self.coef_dim)
        act_coef3 = self.gr_act_coef_layer3(x).permute(0, 2, 3, 1).reshape(x.size(0), -1, self.coef_dim)
        act_coef4 = self.gr_act_coef_layer4(x).permute(0, 2, 3, 1).reshape(x.size(0), -1, self.coef_dim)
        act_coef5 = self.gr_act_coef_layer5(x).permute(0, 2, 3, 1).reshape(x.size(0), -1, self.coef_dim)
        act_coef6 = self.gr_act_coef_layer6(x).permute(0, 2, 3, 1).reshape(x.size(0), -1, self.coef_dim)
        act_coef7 = self.gr_act_coef_layer7(x).permute(0, 2, 3, 1).reshape(x.size(0), -1, self.coef_dim)
        act_coef8 = self.gr_act_coef_layer8(x).permute(0, 2, 3, 1).reshape(x.size(0), -1, self.coef_dim)
        act_coef9 = self.gr_act_coef_layer9(x).permute(0, 2, 3, 1).reshape(x.size(0), -1, self.coef_dim)
        act_coef10 = self.gr_act_coef_layer10(x).permute(0, 2, 3, 1).reshape(x.size(0), -1, self.coef_dim)
        act_coef11 = self.gr_act_coef_layer11(x).permute(0, 2, 3, 1).reshape(x.size(0), -1, self.coef_dim)


        
        return (
            conf, box, gr_pos_coef, gr_sin_coef, gr_cos_coef, gr_wid_coef,
            act_coef1, act_coef2, act_coef3, act_coef4,
            act_coef5, act_coef6, act_coef7, act_coef8,
            act_coef9, act_coef10, act_coef11
        )


class ProtoNet(nn.Module):
    def __init__(self, coef_dim):
        super().__init__()
        self.proto1 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(inplace=True))
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.proto2 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(256, coef_dim, kernel_size=1, stride=1),
                                    nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.proto1(x)
        x = self.upsample(x)
        x = self.proto2(x)
        return x


class FPN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.lat_layers = nn.ModuleList([nn.Conv2d(x, 256, kernel_size=1) for x in self.in_channels])
        self.pred_layers = nn.ModuleList([nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                                        nn.ReLU(inplace=True)) for _ in self.in_channels])

        self.downsample_layers = nn.ModuleList([nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2),
                                                              nn.ReLU(inplace=True)),
                                                nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2),
                                                              nn.ReLU(inplace=True))])

        self.upsample_module = nn.ModuleList([nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                              nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)])

    def forward(self, backbone_outs):
        p5_1 = self.lat_layers[2](backbone_outs[2])
        p5_upsample = self.upsample_module[1](p5_1)

        p4_1 = self.lat_layers[1](backbone_outs[1]) + p5_upsample
        p4_upsample = self.upsample_module[0](p4_1)

        p3_1 = self.lat_layers[0](backbone_outs[0]) + p4_upsample

        p5 = self.pred_layers[2](p5_1)
        p4 = self.pred_layers[1](p4_1)
        p3 = self.pred_layers[0](p3_1)

        p6 = self.downsample_layers[0](p5)
        p7 = self.downsample_layers[1](p6)

        return p3, p4, p5, p6, p7


class GraspYolact(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.coef_dim = 32

        if cfg.__class__.__name__.startswith('res101'):
            self.backbone = ResNet(layers=(3, 4, 23, 3))
            self.fpn = FPN(in_channels=(512, 1024, 2048))
        elif cfg.__class__.__name__.startswith('res50'):
            self.backbone = ResNet(layers=(3, 4, 6, 3))
            self.fpn = FPN(in_channels=(512, 1024, 2048))
        elif cfg.__class__.__name__.startswith('swin_tiny'):
            self.backbone = SwinTransformer()
            self.fpn = FPN(in_channels=(192, 384, 768))
        

        print(self.backbone)

        self.proto_net = ProtoNet(coef_dim=self.coef_dim)
        self.prediction_layers = PredictionModule(cfg, coef_dim=self.coef_dim)

        self.anchors = []
        fpn_fm_shape = [math.ceil(cfg.img_size / stride) for stride in (8, 16, 32, 64, 128)]
        for i, size in enumerate(fpn_fm_shape):
            self.anchors += make_anchors(self.cfg, size, size, self.cfg.scales[i])

        # if cfg.mode == 'train':
        #     # For OCID grasp dataset
        #     self.semantic_seg_conv = nn.Conv2d(256, cfg.num_classes, kernel_size=1)

        #     # self.semantic_seg_conv = nn.Conv2d(256, cfg.num_classes - 1, kernel_size=1)

        # init weights, backbone weights will be covered later
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight.data)

                if module.bias is not None:
                    module.bias.data.zero_()

    def load_weights(self, weight, cuda):
        if isinstance(weight, str):
            if cuda:
                state_dict = torch.load(weight)
            else:
                state_dict = torch.load(weight, map_location='cpu')

            for key in list(state_dict.keys()):
                if self.cfg.mode != 'train' and key.startswith('semantic_seg_conv'):
                    del state_dict[key]

            self.load_state_dict(state_dict, strict=True)
        elif isinstance(weight, dict):
            for key in list(weight.keys()):
                if self.cfg.mode != 'train' and key.startswith('semantic_seg_conv'):
                    del weight[key]

            self.load_state_dict(weight, strict=True)
        # print(f'Model loaded with {weight}.\n')
        # print(f'Number of all parameters: {sum([p.numel() for p in self.parameters()])}\n')
    

    def load_partial_weights(self, weight, cuda):
        if cuda:
            pt_state_dict = torch.load(weight)
        else:
            pt_state_dict = torch.load(weight, map_location='cpu')
        
        for key in list(pt_state_dict.keys()):
            if self.cfg.mode != 'train' and key.startswith('semantic_seg_conv'):
                del pt_state_dict[key]
        
        state_dict = self.state_dict()
        for name, param in pt_state_dict.items():
            if state_dict[name].shape != param.shape:
                continue
            state_dict[name] = param
        
        self.load_state_dict(state_dict)


    def forward(self, img, box_classes=None, pos_mask_gt=None, qua_mask_gt=None, sin_mask_gt=None, cos_mask_gt=None, wid_mask_gt=None, act_mask_gt=None, act_mask_weights=None):

        outs = self.backbone(img)
        # print(len(outs))

        # for out in outs:
        #     print(out.shape)
        outs = self.fpn(outs[1:4])
        proto_out = self.proto_net(outs[0])  # feature map P3

        proto_out = proto_out.permute(0, 2, 3, 1).contiguous()

        class_pred, box_pred, gr_pos_coef_pred, gr_sin_coef_pred, gr_cos_coef_pred, gr_wid_coef_pred = [], [], [], [], [], []
        act1_coef_pred = []
        act2_coef_pred = []
        act3_coef_pred = []
        act4_coef_pred = []
        act5_coef_pred = []
        act6_coef_pred = []
        act7_coef_pred = []
        act8_coef_pred = []
        act9_coef_pred = []
        act10_coef_pred = []
        act11_coef_pred = []

        for aa in outs:
            class_p, box_p, gr_pos_coef_p, gr_sin_coef_p, gr_cos_coef_p, gr_wid_coef_p, act_coef1, act_coef2, act_coef3, act_coef4, act_coef5, act_coef6, act_coef7, act_coef8, act_coef9, act_coef10, act_coef11= self.prediction_layers(aa)
            class_pred.append(class_p)
            box_pred.append(box_p)
            gr_pos_coef_pred.append(gr_pos_coef_p)
            gr_sin_coef_pred.append(gr_sin_coef_p)
            gr_cos_coef_pred.append(gr_cos_coef_p)
            gr_wid_coef_pred.append(gr_wid_coef_p)
            act1_coef_pred.append(act_coef1)
            act2_coef_pred.append(act_coef2)
            act3_coef_pred.append(act_coef3)
            act4_coef_pred.append(act_coef4)
            act5_coef_pred.append(act_coef5)
            act6_coef_pred.append(act_coef6)
            act7_coef_pred.append(act_coef7)
            act8_coef_pred.append(act_coef8)
            act9_coef_pred.append(act_coef9)
            act10_coef_pred.append(act_coef10)
            act11_coef_pred.append(act_coef11)


        class_pred = torch.cat(class_pred, dim=1)
        box_pred = torch.cat(box_pred, dim=1)

        gr_pos_coef_pred = torch.cat(gr_pos_coef_pred, dim=1)
        gr_sin_coef_pred = torch.cat(gr_sin_coef_pred, dim=1)
        gr_cos_coef_pred = torch.cat(gr_cos_coef_pred, dim=1)
        gr_wid_coef_pred = torch.cat(gr_wid_coef_pred, dim=1)

        act1_coef_pred = torch.cat(act1_coef_pred, dim=1)
        act2_coef_pred = torch.cat(act2_coef_pred, dim=1)
        act3_coef_pred = torch.cat(act3_coef_pred, dim=1)
        act4_coef_pred = torch.cat(act4_coef_pred, dim=1)
        act5_coef_pred = torch.cat(act5_coef_pred, dim=1)
        act6_coef_pred = torch.cat(act6_coef_pred, dim=1)
        act7_coef_pred = torch.cat(act7_coef_pred, dim=1)
        act8_coef_pred = torch.cat(act8_coef_pred, dim=1)
        act9_coef_pred = torch.cat(act9_coef_pred, dim=1)
        act10_coef_pred = torch.cat(act10_coef_pred, dim=1)
        act11_coef_pred = torch.cat(act11_coef_pred, dim=1)

        if self.training:
            return self.compute_loss(
                class_pred, 
                box_pred, 
                gr_pos_coef_pred, gr_sin_coef_pred, gr_cos_coef_pred, gr_wid_coef_pred, 
                act1_coef_pred, act2_coef_pred, act3_coef_pred, act4_coef_pred,
                act5_coef_pred, act6_coef_pred, act7_coef_pred, act8_coef_pred,
                act9_coef_pred, act10_coef_pred, act11_coef_pred,
                proto_out, box_classes,
                pos_mask_gt, qua_mask_gt, sin_mask_gt, cos_mask_gt, wid_mask_gt,
                act_mask_gt, act_mask_weights
            )
        else:
            class_pred = F.softmax(class_pred, -1)
            return (
                class_pred, box_pred, gr_pos_coef_pred, gr_sin_coef_pred, gr_cos_coef_pred, gr_wid_coef_pred, 
                act1_coef_pred, act2_coef_pred, act3_coef_pred, act4_coef_pred,
                act5_coef_pred, act6_coef_pred, act7_coef_pred, act8_coef_pred,
                act9_coef_pred, act10_coef_pred, act11_coef_pred,
                proto_out
            )

    def compute_loss(
        self, 
        class_p, 
        box_p, 
        pos_coef_p, sin_coef_p, cos_coef_p, wid_coef_p, 
        act1_coef_pred, act2_coef_pred, act3_coef_pred, act4_coef_pred,
        act5_coef_pred, act6_coef_pred, act7_coef_pred, act8_coef_pred,
        act9_coef_pred, act10_coef_pred, act11_coef_pred,
        proto_p, box_class, 
        pos_mask_gt, qua_mask_gt, sin_mask_gt, cos_mask_gt, wid_mask_gt,
        act_mask_gt, act_mask_weights
    ):
        
        device = class_p.device
        class_gt = [None] * len(box_class)
        batch_size = box_p.size(0)

        if isinstance(self.anchors, list):
            self.anchors = torch.tensor(self.anchors, device=device).reshape(-1, 4)

        num_anchors = self.anchors.shape[0]

        all_offsets = torch.zeros((batch_size, num_anchors, 4), dtype=torch.float32, device=device)
        conf_gt = torch.zeros((batch_size, num_anchors), dtype=torch.int64, device=device)
        anchor_max_gt = torch.zeros((batch_size, num_anchors, 4), dtype=torch.float32, device=device)
        anchor_max_i = torch.zeros((batch_size, num_anchors), dtype=torch.int64, device=device)

        for i in range(batch_size):
            box_gt = box_class[i][:, :-1]
            class_gt[i] = box_class[i][:, -1].long()

            all_offsets[i], conf_gt[i], anchor_max_gt[i], anchor_max_i[i] = match(self.cfg, box_gt,
                                                                                  self.anchors, class_gt[i])

        # all_offsets: the transformed box coordinate offsets of each pair of anchor and gt box
        # conf_gt: the foreground and background labels according to the 'pos_thre' and 'neg_thre',
        #          '0' means background, '>0' means foreground.
        # anchor_max_gt: the corresponding max IoU gt box for each anchor
        # anchor_max_i: the index of the corresponding max IoU gt box for each anchor
        assert (not all_offsets.requires_grad) and (not conf_gt.requires_grad) and \
               (not anchor_max_i.requires_grad), 'Incorrect computation graph, check the grad.'

        # only compute losses from positive samples
        pos_bool = conf_gt > 0

        loss_c = self.category_loss(class_p, conf_gt, pos_bool)
        loss_b = self.box_loss(box_p, all_offsets, pos_bool)
        # loss_m = self.lincomb_mask_loss(pos_bool, anchor_max_i, coef_p, proto_p, mask_gt, anchor_max_gt)

        # @TODO Grasp mask loss
        loss_g_pos, loss_g_qua, loss_g_ang, loss_g_wid, loss_g_act1, loss_g_act2, loss_g_act3, loss_g_act4, loss_g_act5, loss_g_act6, loss_g_act7, loss_g_act8, loss_g_act9, loss_g_act10, loss_g_act11 = self.lincomb_grasp_mask_loss(
            pos_bool, anchor_max_i, 
            pos_coef_p, sin_coef_p, cos_coef_p, wid_coef_p, 
            act1_coef_pred, act2_coef_pred, act3_coef_pred, act4_coef_pred,
            act5_coef_pred, act6_coef_pred, act7_coef_pred, act8_coef_pred,
            act9_coef_pred, act10_coef_pred, act11_coef_pred,
            proto_p,
            pos_mask_gt, qua_mask_gt, sin_mask_gt, cos_mask_gt, wid_mask_gt, act_mask_gt, act_mask_weights,
            anchor_max_gt
        )
        loss_g_act = loss_g_act1 + loss_g_act2 + loss_g_act3 + loss_g_act4 + loss_g_act5 + loss_g_act6 + loss_g_act7 + loss_g_act8 + loss_g_act9 + loss_g_act10 +loss_g_act11

        return loss_c, loss_b, loss_g_pos, loss_g_qua, loss_g_ang, loss_g_wid, loss_g_act

    def category_loss(self, class_p, conf_gt, pos_bool, np_ratio=3):
        # Compute max conf across batch for hard negative mining
        batch_conf = class_p.reshape(-1, self.cfg.num_classes)

        batch_conf_max = batch_conf.max()

        mark = torch.log(torch.sum(torch.exp(batch_conf - batch_conf_max), 1)) + batch_conf_max - batch_conf[:, 0]

        # Hard Negative Mining
        mark = mark.reshape(class_p.size(0), -1)
        mark[pos_bool] = 0  # filter out pos boxes
        mark[conf_gt < 0] = 0  # filter out neutrals (conf_gt = -1)

        _, idx = mark.sort(1, descending=True)
        _, idx_rank = idx.sort(1)

        num_pos = pos_bool.long().sum(1, keepdim=True)
        num_neg = torch.clamp(np_ratio * num_pos, max=pos_bool.size(1) - 1)
        neg_bool = idx_rank < num_neg.expand_as(idx_rank)

        # Just in case there aren't enough negatives, don't start using positives as negatives
        neg_bool[pos_bool] = 0
        neg_bool[conf_gt < 0] = 0  # Filter out neutrals

        # Confidence Loss Including Positive and Negative Examples
        class_p_mined = class_p[(pos_bool + neg_bool)].reshape(-1, self.cfg.num_classes)
        class_gt_mined = conf_gt[(pos_bool + neg_bool)]

        return self.cfg.conf_alpha * F.cross_entropy(class_p_mined, class_gt_mined, reduction='sum') / num_pos.sum()

    def box_loss(self, box_p, all_offsets, pos_bool):
        num_pos = pos_bool.sum()
        pos_box_p = box_p[pos_bool, :]
        pos_offsets = all_offsets[pos_bool, :]

        return self.cfg.bbox_alpha * F.smooth_l1_loss(pos_box_p, pos_offsets, reduction='sum') / num_pos


    # @TODO
    def lincomb_grasp_mask_loss(
        self, 
        pos_bool, anchor_max_i, 
        gr_pos_coef_p, gr_sin_coef_p, gr_cos_coef_p, gr_wid_coef_p, 
        act1_coef_pred, act2_coef_pred, act3_coef_pred, act4_coef_pred,
        act5_coef_pred, act6_coef_pred, act7_coef_pred, act8_coef_pred,
        act9_coef_pred, act10_coef_pred, act11_coef_pred,
        proto_p, 
        pos_mask_gt, qua_mask_gt, sin_mask_gt, cos_mask_gt, wid_mask_gt,
        act_mask_gt, act_mask_weights,
        anchor_max_gt):

        assert gr_pos_coef_p.shape == gr_sin_coef_p.shape == gr_cos_coef_p.shape == gr_wid_coef_p.shape

        proto_h, proto_w = proto_p.shape[1:3]
        total_pos_num = pos_bool.sum()
        loss_g_pos = 0
        loss_g_qua = 0
        loss_g_ang = 0
        loss_g_wid = 0
        loss_g_act1 = 0
        loss_g_act2 = 0
        loss_g_act3 = 0
        loss_g_act4 = 0

        loss_g_act5 = 0
        loss_g_act6 = 0
        loss_g_act7 = 0
        loss_g_act8 = 0

        loss_g_act9 = 0
        loss_g_act10 = 0
        loss_g_act11 = 0


        for i in range(gr_pos_coef_p.size(0)):
            # downsample the gt mask to the size of 'proto_p'
            ds_pos_masks = F.interpolate(pos_mask_gt[i].unsqueeze(0), (proto_h, proto_w), mode='bilinear',
                                              align_corners=False).squeeze(0)
            ds_pos_masks = ds_pos_masks.permute(1, 2, 0).contiguous().float()
            ds_pos_masks = ds_pos_masks.gt(0.5).float()

            ds_act1_masks = F.interpolate(act_mask_gt[i][:,0,:,:].unsqueeze(0), (proto_h, proto_w), mode='bilinear',
                                              align_corners=False).squeeze(0)
            ds_act1_masks = ds_act1_masks.permute(1, 2, 0).contiguous().float()
            ds_act1_masks = ds_act1_masks.gt(0.5).float()

            
            ds_act2_masks = F.interpolate(act_mask_gt[i][:,1, :, :].unsqueeze(0), (proto_h, proto_w), mode='bilinear',
                                              align_corners=False).squeeze(0)
            ds_act2_masks = ds_act2_masks.permute(1, 2, 0).contiguous().float()
            ds_act2_masks = ds_act2_masks.gt(0.5).float()


            ds_act3_masks = F.interpolate(act_mask_gt[i][:,2, :, :].unsqueeze(0), (proto_h, proto_w), mode='bilinear',
                                              align_corners=False).squeeze(0)
            ds_act3_masks = ds_act3_masks.permute(1, 2, 0).contiguous().float()
            ds_act3_masks = ds_act3_masks.gt(0.5).float()

            ####################################################

            ds_act4_masks = F.interpolate(act_mask_gt[i][:,3, :, :].unsqueeze(0), (proto_h, proto_w), mode='bilinear',
                                              align_corners=False).squeeze(0)
            ds_act4_masks = ds_act4_masks.permute(1, 2, 0).contiguous().float()
            ds_act4_masks = ds_act4_masks.gt(0.5).float()

            
            ds_act5_masks = F.interpolate(act_mask_gt[i][:,4,:,:].unsqueeze(0), (proto_h, proto_w), mode='bilinear',
                                              align_corners=False).squeeze(0)
            ds_act5_masks = ds_act5_masks.permute(1, 2, 0).contiguous().float()
            ds_act5_masks = ds_act5_masks.gt(0.5).float()


            ds_act6_masks = F.interpolate(act_mask_gt[i][:,5,:,:].unsqueeze(0), (proto_h, proto_w), mode='bilinear',
                                              align_corners=False).squeeze(0)
            ds_act6_masks = ds_act6_masks.permute(1, 2, 0).contiguous().float()
            ds_act6_masks = ds_act6_masks.gt(0.5).float()

            ####################################################

            ds_act7_masks = F.interpolate(act_mask_gt[i][:,6,:,:].unsqueeze(0), (proto_h, proto_w), mode='bilinear',
                                              align_corners=False).squeeze(0)
            ds_act7_masks = ds_act7_masks.permute(1, 2, 0).contiguous().float()
            ds_act7_masks = ds_act7_masks.gt(0.5).float()

            
            ds_act8_masks = F.interpolate(act_mask_gt[i][:,7,:,:].unsqueeze(0), (proto_h, proto_w), mode='bilinear',
                                              align_corners=False).squeeze(0)
            ds_act8_masks = ds_act8_masks.permute(1, 2, 0).contiguous().float()
            ds_act8_masks = ds_act8_masks.gt(0.5).float()


            ds_act9_masks = F.interpolate(act_mask_gt[i][:,8,:,:].unsqueeze(0), (proto_h, proto_w), mode='bilinear',
                                              align_corners=False).squeeze(0)
            ds_act9_masks = ds_act9_masks.permute(1, 2, 0).contiguous().float()
            ds_act9_masks = ds_act9_masks.gt(0.5).float()

            ####################################################

            ds_act10_masks = F.interpolate(act_mask_gt[i][:,9,:,:].unsqueeze(0), (proto_h, proto_w), mode='bilinear',
                                              align_corners=False).squeeze(0)
            ds_act10_masks = ds_act10_masks.permute(1, 2, 0).contiguous().float()
            ds_act10_masks = ds_act10_masks.gt(0.5).float()


            ds_act11_masks = F.interpolate(act_mask_gt[i][:,10,:,:].unsqueeze(0), (proto_h, proto_w), mode='bilinear',
                                              align_corners=False).squeeze(0)
            ds_act11_masks = ds_act11_masks.permute(1, 2, 0).contiguous().float()
            ds_act11_masks = ds_act11_masks.gt(0.5).float()

            ####################################################

            ds_qua_masks = F.interpolate(qua_mask_gt[i].unsqueeze(0), (proto_h, proto_w), mode='bilinear',
                                              align_corners=False).squeeze(0)
            ds_qua_masks = ds_qua_masks.permute(1, 2, 0).contiguous().float()

            # @TODO Not sure should we upsample the protos to the original size and compute the loss
            # This will be test in the next version
            ds_sin_masks = F.interpolate(sin_mask_gt[i].unsqueeze(0), (proto_h, proto_w), mode='bilinear',
                                              align_corners=False).squeeze(0)
            ds_sin_masks = ds_sin_masks.permute(1, 2, 0).contiguous().float()

            ds_cos_masks = F.interpolate(cos_mask_gt[i].unsqueeze(0), (proto_h, proto_w), mode='bilinear',
                                              align_corners=False).squeeze(0)
            ds_cos_masks = ds_cos_masks.permute(1, 2, 0).contiguous().float()


            ds_wid_masks = F.interpolate(wid_mask_gt[i].unsqueeze(0), (proto_h, proto_w), mode='bilinear',
                                              align_corners=False).squeeze(0)
            ds_wid_masks = ds_wid_masks.permute(1, 2, 0).contiguous().float()
            
            anchor_i = anchor_max_i[i][pos_bool[i]]
            anchor_box = anchor_max_gt[i][pos_bool[i]]
            gr_pos_coef = gr_pos_coef_p[i][pos_bool[i]]
            gr_sin_coef = gr_sin_coef_p[i][pos_bool[i]]
            gr_cos_coef = gr_cos_coef_p[i][pos_bool[i]]
            gr_wid_coef = gr_wid_coef_p[i][pos_bool[i]]

            act_coef1 = act1_coef_pred[i][pos_bool[i]]
            act_coef2 = act2_coef_pred[i][pos_bool[i]]
            act_coef3 = act3_coef_pred[i][pos_bool[i]]
            act_coef4 = act4_coef_pred[i][pos_bool[i]]

            act_coef5 = act5_coef_pred[i][pos_bool[i]]
            act_coef6 = act6_coef_pred[i][pos_bool[i]]
            act_coef7 = act7_coef_pred[i][pos_bool[i]]
            act_coef8 = act8_coef_pred[i][pos_bool[i]]

            act_coef9 = act9_coef_pred[i][pos_bool[i]]
            act_coef10 = act10_coef_pred[i][pos_bool[i]]
            act_coef11 = act11_coef_pred[i][pos_bool[i]]

            assert gr_pos_coef.shape == gr_sin_coef.shape == gr_cos_coef.shape == gr_wid_coef.shape

            if anchor_i.size(0) == 0:
                continue

            # If exceeds the number of masks for training, select a random subset
            old_num_pos = gr_pos_coef.size(0)
            if old_num_pos > self.cfg.masks_to_train:
                perm = torch.randperm(gr_pos_coef.size(0))
                select = perm[:self.cfg.masks_to_train]
                gr_pos_coef = gr_pos_coef[select]
                gr_sin_coef = gr_sin_coef[select]
                gr_cos_coef = gr_cos_coef[select]
                gr_wid_coef = gr_wid_coef[select]

                act_coef1 = act_coef1[select]
                act_coef2 = act_coef2[select]
                act_coef3 = act_coef3[select]
                act_coef4 = act_coef4[select]

                act_coef5 = act_coef5[select]
                act_coef6 = act_coef6[select]
                act_coef7 = act_coef7[select]
                act_coef8 = act_coef8[select]

                act_coef9 = act_coef9[select]
                act_coef10 = act_coef10[select]
                act_coef11 = act_coef11[select]

                anchor_i = anchor_i[select]
                anchor_box = anchor_box[select]

            num_pos = gr_pos_coef.size(0)

            pos_mask_gt_i = ds_pos_masks[:, :, anchor_i]
            qua_mask_gt_i = ds_qua_masks[:, :, anchor_i]
            sin_mask_gt_i = ds_sin_masks[:, :, anchor_i]
            cos_mask_gt_i = ds_cos_masks[:, :, anchor_i]
            wid_mask_gt_i = ds_wid_masks[:, :, anchor_i]

            act1_mask_gt_i = ds_act1_masks[:, :, anchor_i]
            act2_mask_gt_i = ds_act2_masks[:, :, anchor_i]
            act3_mask_gt_i = ds_act3_masks[:, :, anchor_i]
            act4_mask_gt_i = ds_act4_masks[:, :, anchor_i]

            act5_mask_gt_i = ds_act5_masks[:, :, anchor_i]
            act6_mask_gt_i = ds_act6_masks[:, :, anchor_i]
            act7_mask_gt_i = ds_act7_masks[:, :, anchor_i]
            act8_mask_gt_i = ds_act8_masks[:, :, anchor_i]

            act9_mask_gt_i = ds_act9_masks[:, :, anchor_i]
            act10_mask_gt_i = ds_act10_masks[:, :, anchor_i]
            act11_mask_gt_i = ds_act11_masks[:, :, anchor_i]


            # mask assembly by linear combination
            # @ means dot product
            pos_mask_p = torch.sigmoid(proto_p[i] @ gr_pos_coef.t())
            pos_mask_p = crop(pos_mask_p, anchor_box)  # pos_anchor_box.shape: (num_pos, 4)

            sin_mask_p = torch.tanh(proto_p[i] @ gr_sin_coef.t())
            sin_mask_p = crop(sin_mask_p, anchor_box)

            cos_mask_p = torch.tanh(proto_p[i] @ gr_cos_coef.t())
            cos_mask_p = ones_crop(cos_mask_p, anchor_box)

            wid_mask_p = torch.sigmoid(proto_p[i] @ gr_wid_coef.t())
            wid_mask_p = crop(wid_mask_p, anchor_box)

            act1_mask_p = torch.sigmoid(proto_p[i] @ act_coef1.t())
            act1_mask_p = crop(act1_mask_p, anchor_box)

            act2_mask_p = torch.sigmoid(proto_p[i] @ act_coef2.t())
            act2_mask_p = crop(act2_mask_p, anchor_box)
            
            act3_mask_p = torch.sigmoid(proto_p[i] @ act_coef3.t())
            act3_mask_p = crop(act3_mask_p, anchor_box)

            act4_mask_p = torch.sigmoid(proto_p[i] @ act_coef4.t())
            act4_mask_p = crop(act4_mask_p, anchor_box)

            act5_mask_p = torch.sigmoid(proto_p[i] @ act_coef5.t())
            act5_mask_p = crop(act5_mask_p, anchor_box)

            act6_mask_p = torch.sigmoid(proto_p[i] @ act_coef6.t())
            act6_mask_p = crop(act6_mask_p, anchor_box)

            act7_mask_p = torch.sigmoid(proto_p[i] @ act_coef7.t())
            act7_mask_p = crop(act7_mask_p, anchor_box)

            act8_mask_p = torch.sigmoid(proto_p[i] @ act_coef8.t())
            act8_mask_p = crop(act8_mask_p, anchor_box)

            act9_mask_p = torch.sigmoid(proto_p[i] @ act_coef9.t())
            act9_mask_p = crop(act9_mask_p, anchor_box)

            act10_mask_p = torch.sigmoid(proto_p[i] @ act_coef10.t())
            act10_mask_p = crop(act10_mask_p, anchor_box)
            
            act11_mask_p = torch.sigmoid(proto_p[i] @ act_coef11.t())
            act11_mask_p = crop(act11_mask_p, anchor_box)



            # TODO: grad out of gt box is 0, should it be modified?
            # TODO: need an upsample before computing loss?
            pos_mask_loss = F.binary_cross_entropy(torch.clamp(pos_mask_p, 0, 1), torch.clamp(pos_mask_gt_i, 0, 1), reduction='none')
            qua_mask_loss = F.smooth_l1_loss(torch.clamp(pos_mask_p, 0, 1), qua_mask_gt_i, reduction="none")
            sin_ang_mask_loss = F.smooth_l1_loss(torch.clamp(sin_mask_p, -1, 1), sin_mask_gt_i, reduction="none")
            cos_ang_mask_loss = F.smooth_l1_loss(torch.clamp(cos_mask_p, -1, 1), cos_mask_gt_i, reduction="none")
            ang_mask_loss = sin_ang_mask_loss + cos_ang_mask_loss
            wid_mask_loss = F.smooth_l1_loss(torch.clamp(wid_mask_p, 0, 1), wid_mask_gt_i, reduction="none")


            act1_mask_loss = F.smooth_l1_loss(torch.clamp(act1_mask_p, 0, 1), act1_mask_gt_i, reduction="none")
            act2_mask_loss = F.smooth_l1_loss(torch.clamp(act2_mask_p, 0, 1), act2_mask_gt_i, reduction="none")
            act3_mask_loss = F.smooth_l1_loss(torch.clamp(act3_mask_p, 0, 1), act3_mask_gt_i, reduction="none")
            act4_mask_loss = F.smooth_l1_loss(torch.clamp(act4_mask_p, 0, 1), act4_mask_gt_i, reduction="none")


            act5_mask_loss = F.smooth_l1_loss(torch.clamp(act5_mask_p, 0, 1), act5_mask_gt_i, reduction="none")
            act6_mask_loss = F.smooth_l1_loss(torch.clamp(act6_mask_p, 0, 1), act6_mask_gt_i, reduction="none")
            act7_mask_loss = F.smooth_l1_loss(torch.clamp(act7_mask_p, 0, 1), act7_mask_gt_i, reduction="none")
            act8_mask_loss = F.smooth_l1_loss(torch.clamp(act8_mask_p, 0, 1), act8_mask_gt_i, reduction="none")


            act9_mask_loss = F.smooth_l1_loss(torch.clamp(act9_mask_p, 0, 1), act9_mask_gt_i, reduction="none")
            act10_mask_loss = F.smooth_l1_loss(torch.clamp(act10_mask_p, 0, 1), act10_mask_gt_i, reduction="none")
            act11_mask_loss = F.smooth_l1_loss(torch.clamp(act11_mask_p, 0, 1), act11_mask_gt_i, reduction="none")
            
            
            anchor_area = (anchor_box[:, 2] - anchor_box[:, 0]) * (anchor_box[:, 3] - anchor_box[:, 1])

            pos_mask_loss = pos_mask_loss.sum(dim=(0, 1)) / anchor_area
            qua_mask_loss = qua_mask_loss.sum(dim=(0, 1)) / anchor_area
            ang_mask_loss = ang_mask_loss.sum(dim=(0, 1)) / anchor_area
            wid_mask_loss = wid_mask_loss.sum(dim=(0, 1)) / anchor_area

            act1_mask_loss = act1_mask_loss.sum(dim=(0,1)) / anchor_area
            act2_mask_loss = act2_mask_loss.sum(dim=(0,1)) / anchor_area
            act3_mask_loss = act3_mask_loss.sum(dim=(0,1)) / anchor_area
            act4_mask_loss = act4_mask_loss.sum(dim=(0,1)) / anchor_area

            act5_mask_loss = act5_mask_loss.sum(dim=(0,1)) / anchor_area
            act6_mask_loss = act6_mask_loss.sum(dim=(0,1)) / anchor_area
            act7_mask_loss = act7_mask_loss.sum(dim=(0,1)) / anchor_area
            act8_mask_loss = act8_mask_loss.sum(dim=(0,1)) / anchor_area

            act9_mask_loss = act9_mask_loss.sum(dim=(0,1)) / anchor_area
            act10_mask_loss = act10_mask_loss.sum(dim=(0,1)) / anchor_area
            act11_mask_loss = act11_mask_loss.sum(dim=(0,1)) / anchor_area
                
            if old_num_pos > num_pos:
                pos_mask_loss *= old_num_pos / num_pos
                qua_mask_loss *= old_num_pos / num_pos
                ang_mask_loss *= old_num_pos / num_pos
                wid_mask_loss *= old_num_pos / num_pos

                act1_mask_loss *= old_num_pos / num_pos
                act2_mask_loss *= old_num_pos / num_pos
                act3_mask_loss *= old_num_pos / num_pos
                act4_mask_loss *= old_num_pos / num_pos

                act5_mask_loss *= old_num_pos / num_pos
                act6_mask_loss *= old_num_pos / num_pos
                act7_mask_loss *= old_num_pos / num_pos
                act8_mask_loss *= old_num_pos / num_pos

                act9_mask_loss *= old_num_pos / num_pos
                act10_mask_loss *= old_num_pos / num_pos
                act11_mask_loss *= old_num_pos / num_pos

            loss_g_pos += torch.sum(pos_mask_loss) 
            loss_g_qua += torch.sum(qua_mask_loss)
            loss_g_ang += torch.sum(ang_mask_loss) 
            loss_g_wid += torch.sum(wid_mask_loss)

            
            # loss_g_act1 += torch.sum(act1_mask_loss * act_mask_weight[:, 0])
            # loss_g_act2 += torch.sum(act2_mask_loss * act_mask_weight[:, 1])
            # loss_g_act3 += torch.sum(act3_mask_loss * act_mask_weight[:, 2])
            # loss_g_act4 += torch.sum(act4_mask_loss * act_mask_weight[:, 3])

            # loss_g_act5 += torch.sum(act5_mask_loss * act_mask_weight[:, 4])
            # loss_g_act6 += torch.sum(act6_mask_loss * act_mask_weight[:,5])
            # loss_g_act7 += torch.sum(act7_mask_loss * act_mask_weight[:,6])
            # loss_g_act8 += torch.sum(act8_mask_loss * act_mask_weight[:,7])

            # loss_g_act9 += torch.sum(act9_mask_loss * act_mask_weight[:,8])
            # loss_g_act10 += torch.sum(act10_mask_loss * act_mask_weight[:,9])
            # loss_g_act11 += torch.sum(act11_mask_loss * act_mask_weight[:,10])

            loss_g_act1 += torch.sum(act1_mask_loss)
            loss_g_act2 += torch.sum(act2_mask_loss)
            loss_g_act3 += torch.sum(act3_mask_loss)
            loss_g_act4 += torch.sum(act4_mask_loss)

            loss_g_act5 += torch.sum(act5_mask_loss)
            loss_g_act6 += torch.sum(act6_mask_loss)
            loss_g_act7 += torch.sum(act7_mask_loss)
            loss_g_act8 += torch.sum(act8_mask_loss)

            loss_g_act9 += torch.sum(act9_mask_loss)
            loss_g_act10 += torch.sum(act10_mask_loss)
            loss_g_act11 += torch.sum(act11_mask_loss)
        
        return (
            self.cfg.grasp_alpha * loss_g_pos / proto_h / proto_w / total_pos_num,
            self.cfg.grasp_alpha * loss_g_qua / proto_h / proto_w / total_pos_num,
            self.cfg.grasp_alpha * loss_g_ang / proto_h / proto_w / total_pos_num,
            self.cfg.grasp_alpha * loss_g_wid / proto_h / proto_w / total_pos_num,

            self.cfg.grasp_alpha * loss_g_act1 / proto_h / proto_w / total_pos_num,
            self.cfg.grasp_alpha * loss_g_act2 / proto_h / proto_w / total_pos_num,
            self.cfg.grasp_alpha * loss_g_act3 / proto_h / proto_w / total_pos_num,
            self.cfg.grasp_alpha * loss_g_act4 / proto_h / proto_w / total_pos_num,

            self.cfg.grasp_alpha * loss_g_act5 / proto_h / proto_w / total_pos_num,
            self.cfg.grasp_alpha * loss_g_act6 / proto_h / proto_w / total_pos_num,
            self.cfg.grasp_alpha * loss_g_act7 / proto_h / proto_w / total_pos_num,
            self.cfg.grasp_alpha * loss_g_act8 / proto_h / proto_w / total_pos_num,

            self.cfg.grasp_alpha * loss_g_act9 / proto_h / proto_w / total_pos_num,
            self.cfg.grasp_alpha * loss_g_act10 / proto_h / proto_w / total_pos_num,
            self.cfg.grasp_alpha * loss_g_act11 / proto_h / proto_w / total_pos_num,
        )


    def lincomb_mask_loss(self, pos_bool, anchor_max_i, coef_p, proto_p, mask_gt, anchor_max_gt):
        proto_h, proto_w = proto_p.shape[1:3]
        total_pos_num = pos_bool.sum()
        loss_m = 0
        for i in range(coef_p.size(0)):
            # downsample the gt mask to the size of 'proto_p'
            downsampled_masks = F.interpolate(mask_gt[i].unsqueeze(0), (proto_h, proto_w), mode='bilinear',
                                              align_corners=False).squeeze(0)
            downsampled_masks = downsampled_masks.permute(1, 2, 0).contiguous()
            # binarize the gt mask because of the downsample operation
            downsampled_masks = downsampled_masks.gt(0.5).float()

            pos_anchor_i = anchor_max_i[i][pos_bool[i]]
            pos_anchor_box = anchor_max_gt[i][pos_bool[i]]
            pos_coef = coef_p[i][pos_bool[i]]

            if pos_anchor_i.size(0) == 0:
                continue

            # If exceeds the number of masks for training, select a random subset
            old_num_pos = pos_coef.size(0)
            if old_num_pos > self.cfg.masks_to_train:
                perm = torch.randperm(pos_coef.size(0))
                select = perm[:self.cfg.masks_to_train]
                pos_coef = pos_coef[select]
                pos_anchor_i = pos_anchor_i[select]
                pos_anchor_box = pos_anchor_box[select]

            num_pos = pos_coef.size(0)

            pos_mask_gt = downsampled_masks[:, :, pos_anchor_i]

            # mask assembly by linear combination
            # @ means dot product
            mask_p = torch.sigmoid(proto_p[i] @ pos_coef.t())
            mask_p = crop(mask_p, pos_anchor_box)  # pos_anchor_box.shape: (num_pos, 4)
            # TODO: grad out of gt box is 0, should it be modified?
            # TODO: need an upsample before computing loss?
            mask_loss = F.binary_cross_entropy(torch.clamp(mask_p, 0, 1), pos_mask_gt, reduction='none')
            # mask_loss = -pos_mask_gt*torch.log(mask_p) - (1-pos_mask_gt) * torch.log(1-mask_p)

            # Normalize the mask loss to emulate roi pooling's effect on loss.
            anchor_area = (pos_anchor_box[:, 2] - pos_anchor_box[:, 0]) * (pos_anchor_box[:, 3] - pos_anchor_box[:, 1])
            mask_loss = mask_loss.sum(dim=(0, 1)) / anchor_area

            if old_num_pos > num_pos:
                mask_loss *= old_num_pos / num_pos

            loss_m += torch.sum(mask_loss)
            # with torch.no_grad():
            #     print("Lincomb mask loss: ", torch.sum(mask_loss))

        return self.cfg.mask_alpha * loss_m / proto_h / proto_w / total_pos_num

    def semantic_seg_loss(self, segmentation_p, mask_gt, class_gt):
        # Note classes here exclude the background class, so num_classes = cfg.num_classes - 1
        batch_size, num_classes, mask_h, mask_w = segmentation_p.size()
        loss_s = 0

        for i in range(batch_size):
            cur_segment = segmentation_p[i]
            cur_class_gt = class_gt[i]

            downsampled_masks = F.interpolate(mask_gt[i].unsqueeze(0), (mask_h, mask_w), mode='bilinear',
                                              align_corners=False).squeeze(0)
            downsampled_masks = downsampled_masks.gt(0.5).float()

            # Construct Semantic Segmentation
            segment_gt = torch.zeros_like(cur_segment, requires_grad=False)
            for j in range(downsampled_masks.size(0)):
                segment_gt[cur_class_gt[j]] = torch.max(segment_gt[cur_class_gt[j]], downsampled_masks[j])

            loss_s += F.binary_cross_entropy_with_logits(cur_segment, segment_gt, reduction='sum')

        return self.cfg.semantic_alpha * loss_s / mask_h / mask_w / batch_size