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
        self.coef_layer = nn.Sequential(nn.Conv2d(256, len(cfg.aspect_ratios) * self.coef_dim,
                                                  kernel_size=3, padding=1),
                                        nn.Tanh())
        

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
        

    def forward(self, x):
        x = self.upfeature(x)
        conf = self.conf_layer(x).permute(0, 2, 3, 1).reshape(x.size(0), -1, self.num_classes)
        box = self.bbox_layer(x).permute(0, 2, 3, 1).reshape(x.size(0), -1, 4)
        coef = self.coef_layer(x).permute(0, 2, 3, 1).reshape(x.size(0), -1, self.coef_dim)

        gr_pos_coef = self.gr_pos_coef_layer(x).permute(0, 2, 3, 1).reshape(x.size(0), -1, self.coef_dim)
        gr_sin_coef = self.gr_sin_coef_layer(x).permute(0, 2, 3, 1).reshape(x.size(0), -1, self.coef_dim)
        gr_cos_coef = self.gr_cos_coef_layer(x).permute(0, 2, 3, 1).reshape(x.size(0), -1, self.coef_dim)
        gr_wid_coef = self.gr_wid_coef_layer(x).permute(0, 2, 3, 1).reshape(x.size(0), -1, self.coef_dim)
        return conf, box, coef, gr_pos_coef, gr_sin_coef, gr_cos_coef, gr_wid_coef


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

        self.proto_net = ProtoNet(coef_dim=self.coef_dim)
        self.prediction_layers = PredictionModule(cfg, coef_dim=self.coef_dim)

        self.anchors = []
        fpn_fm_shape = [math.ceil(cfg.img_size / stride) for stride in (8, 16, 32, 64, 128)]
        for i, size in enumerate(fpn_fm_shape):
            self.anchors += make_anchors(self.cfg, size, size, self.cfg.scales[i])

        if cfg.mode == 'train':
            # For OCID grasp dataset
            self.semantic_seg_conv = nn.Conv2d(256, cfg.num_classes, kernel_size=1)

            # self.semantic_seg_conv = nn.Conv2d(256, cfg.num_classes - 1, kernel_size=1)

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


    def forward(self, img, box_classes=None, masks_gt=None, pos_mask_gt=None, qua_mask_gt=None, sin_mask_gt=None, cos_mask_gt=None, wid_mask_gt=None):

        outs = self.backbone(img)
        outs = self.fpn(outs[1:4])
        proto_out = self.proto_net(outs[0])  # feature map P3

        proto_out = proto_out.permute(0, 2, 3, 1).contiguous()

        class_pred, box_pred, coef_pred, gr_pos_coef_pred, gr_sin_coef_pred, gr_cos_coef_pred, gr_wid_coef_pred = [], [], [], [], [], [], []

        for aa in outs:
            class_p, box_p, coef_p, gr_pos_coef_p, gr_sin_coef_p, gr_cos_coef_p, gr_wid_coef_p = self.prediction_layers(aa)
            class_pred.append(class_p)
            box_pred.append(box_p)
            coef_pred.append(coef_p)
            gr_pos_coef_pred.append(gr_pos_coef_p)
            gr_sin_coef_pred.append(gr_sin_coef_p)
            gr_cos_coef_pred.append(gr_cos_coef_p)
            gr_wid_coef_pred.append(gr_wid_coef_p)


        class_pred = torch.cat(class_pred, dim=1)
        box_pred = torch.cat(box_pred, dim=1)
        coef_pred = torch.cat(coef_pred, dim=1)

        gr_pos_coef_pred = torch.cat(gr_pos_coef_pred, dim=1)
        gr_sin_coef_pred = torch.cat(gr_sin_coef_pred, dim=1)
        gr_cos_coef_pred = torch.cat(gr_cos_coef_pred, dim=1)
        gr_wid_coef_pred = torch.cat(gr_wid_coef_pred, dim=1)

        if self.training:
            seg_pred = self.semantic_seg_conv(outs[0])
            return self.compute_loss(
                class_pred, 
                box_pred, 
                coef_pred, 
                gr_pos_coef_pred, gr_sin_coef_pred, gr_cos_coef_pred, gr_wid_coef_pred, 
                proto_out, seg_pred, box_classes, masks_gt,
                pos_mask_gt, qua_mask_gt, sin_mask_gt, cos_mask_gt, wid_mask_gt
            )
        else:
            class_pred = F.softmax(class_pred, -1)
            return class_pred, box_pred, coef_pred, gr_pos_coef_pred, gr_sin_coef_pred, gr_cos_coef_pred, gr_wid_coef_pred, proto_out

    def compute_loss(
        self, 
        class_p, 
        box_p, 
        coef_p, 
        pos_coef_p, sin_coef_p, cos_coef_p, wid_coef_p, 
        proto_p, seg_p, box_class, mask_gt, 
        pos_mask_gt, qua_mask_gt, sin_mask_gt, cos_mask_gt, wid_mask_gt
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
        loss_m = self.lincomb_mask_loss(pos_bool, anchor_max_i, coef_p, proto_p, mask_gt, anchor_max_gt)

        # @TODO Grasp mask loss
        loss_g_pos, loss_g_qua, loss_g_sin, loss_g_cos, loss_g_wid = self.lincomb_grasp_mask_loss(
            pos_bool, anchor_max_i, 
            pos_coef_p, sin_coef_p, cos_coef_p, wid_coef_p, 
            proto_p,
            pos_mask_gt, qua_mask_gt, sin_mask_gt, cos_mask_gt, wid_mask_gt, 
            anchor_max_gt
        )
        loss_s = self.semantic_seg_loss(seg_p, mask_gt, class_gt)
        return loss_c, loss_b, loss_m, loss_g_pos, loss_g_qua, loss_g_sin, loss_g_cos, loss_g_wid, loss_s

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
        proto_p, 
        pos_mask_gt, qua_mask_gt, sin_mask_gt, cos_mask_gt, wid_mask_gt,
        anchor_max_gt):

        assert gr_pos_coef_p.shape == gr_sin_coef_p.shape == gr_cos_coef_p.shape == gr_wid_coef_p.shape

        proto_h, proto_w = proto_p.shape[1:3]
        total_pos_num = pos_bool.sum()
        loss_g_pos = 0
        loss_g_qua = 0
        loss_g_sin = 0
        loss_g_cos = 0
        loss_g_wid = 0


        for i in range(gr_pos_coef_p.size(0)):
            # downsample the gt mask to the size of 'proto_p'
            ds_pos_masks = F.interpolate(pos_mask_gt[i].unsqueeze(0), (proto_h, proto_w), mode='bilinear',
                                              align_corners=False).squeeze(0)
            ds_pos_masks = ds_pos_masks.permute(1, 2, 0).contiguous().float()
            ds_pos_masks = ds_pos_masks.gt(0.5).float()

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

                anchor_i = anchor_i[select]
                anchor_box = anchor_box[select]

            num_pos = gr_pos_coef.size(0)

            pos_mask_gt_i = ds_pos_masks[:, :, anchor_i]
            qua_mask_gt_i = ds_qua_masks[:, :, anchor_i]
            sin_mask_gt_i = ds_sin_masks[:, :, anchor_i]
            cos_mask_gt_i = ds_cos_masks[:, :, anchor_i]
            wid_mask_gt_i = ds_wid_masks[:, :, anchor_i]



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

            # TODO: grad out of gt box is 0, should it be modified?
            # TODO: need an upsample before computing loss?
            pos_mask_loss = F.smooth_l1_loss(torch.clamp(pos_mask_p, 0, 1), pos_mask_gt_i, reduction='none')
            # pos_mask_loss = F.binary_cross_entropy(torch.clamp(pos_mask_p, 0, 1), pos_mask_gt_i, reduction='none')
            qua_mask_loss = F.smooth_l1_loss(torch.clamp(pos_mask_p, 0, 1), qua_mask_gt_i, reduction="none")
            sin_mask_loss = F.smooth_l1_loss(torch.clamp(sin_mask_p, -1, 1), sin_mask_gt_i, reduction="none")
            cos_mask_loss = F.smooth_l1_loss(torch.clamp(cos_mask_p, -1, 1), cos_mask_gt_i, reduction="none")
            # ang_mask_loss = sin_mask_loss + cos_mask_loss
            wid_mask_loss = F.smooth_l1_loss(torch.clamp(wid_mask_p, 0, 1), wid_mask_gt_i, reduction="none")

            anchor_area = (anchor_box[:, 2] - anchor_box[:, 0]) * (anchor_box[:, 3] - anchor_box[:, 1])

            pos_mask_loss = pos_mask_loss.sum(dim=(0, 1)) / anchor_area
            qua_mask_loss = qua_mask_loss.sum(dim=(0, 1)) / anchor_area
            sin_mask_loss = sin_mask_loss.sum(dim=(0, 1)) / anchor_area
            cos_mask_loss = cos_mask_loss.sum(dim=(0, 1)) / anchor_area
            wid_mask_loss = wid_mask_loss.sum(dim=(0, 1)) / anchor_area
                
            if old_num_pos > num_pos:
                pos_mask_loss *= old_num_pos / num_pos
                qua_mask_loss *= old_num_pos / num_pos
                sin_mask_loss *= old_num_pos / num_pos
                cos_mask_loss *= old_num_pos / num_pos
                wid_mask_loss *= old_num_pos / num_pos

            loss_g_pos += torch.sum(pos_mask_loss) 
            loss_g_qua += torch.sum(qua_mask_loss)
            loss_g_sin += torch.sum(sin_mask_loss) 
            loss_g_cos += torch.sum(cos_mask_loss)
            loss_g_wid += torch.sum(wid_mask_loss)
        
        return (
            self.cfg.grasp_alpha * loss_g_pos / proto_h / proto_w / total_pos_num,
            self.cfg.grasp_alpha * 0.1 * loss_g_qua / proto_h / proto_w / total_pos_num,
            self.cfg.grasp_alpha * loss_g_sin / proto_h / proto_w / total_pos_num,
            self.cfg.grasp_alpha * loss_g_cos / proto_h / proto_w / total_pos_num,
            self.cfg.grasp_alpha * loss_g_wid / proto_h / proto_w / total_pos_num,
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
