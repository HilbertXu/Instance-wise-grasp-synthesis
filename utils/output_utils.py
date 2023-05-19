import os
import torch.nn.functional as F
import cv2
from utils.box_utils import crop, box_iou, box_iou_numpy, crop_numpy
import torch
import numpy as np
from config import COLORS
from cython_nms import nms as cnms
import pdb
from skimage.feature import peak_local_max
from skimage.filters import gaussian
from skimage.draw import polygon
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


norm_mean = np.array([103.94, 116.78, 123.68], dtype=np.float32)
norm_std = np.array([57.38, 57.12, 58.40], dtype=np.float32)


def fast_nms(box_thre, coef_thre, class_thre, cfg):
    class_thre, idx = class_thre.sort(1, descending=True)  # [80, 64 (the number of kept boxes)]

    idx = idx[:, :cfg.top_k]
    class_thre = class_thre[:, :cfg.top_k]

    num_classes, num_dets = idx.size()
    box_thre = box_thre[idx.reshape(-1), :].reshape(num_classes, num_dets, 4)  # [80, 64, 4]
    coef_thre = coef_thre[idx.reshape(-1), :].reshape(num_classes, num_dets, -1)  # [80, 64, 32]

    iou = box_iou(box_thre, box_thre)
    iou.triu_(diagonal=1)
    iou_max, _ = iou.max(dim=1)

    # Now just filter out the ones higher than the threshold
    keep = (iou_max <= cfg.nms_iou_thre)

    # Assign each kept detection to its corresponding class
    class_ids = torch.arange(num_classes, device=box_thre.device)[:, None].expand_as(keep)

    class_ids, box_nms, coef_nms, class_nms = class_ids[keep], box_thre[keep], coef_thre[keep], class_thre[keep]

    # Only keep the top cfg.max_num_detections highest scores across all classes
    class_nms, idx = class_nms.sort(0, descending=True)

    idx = idx[:cfg.max_detections]
    class_nms = class_nms[:cfg.max_detections]

    class_ids = class_ids[idx]
    box_nms = box_nms[idx]
    coef_nms = coef_nms[idx]

    return box_nms, coef_nms, class_ids, class_nms



def fast_osgd_nms(box_thre, gr_pos_coef_thre, gr_sin_coef_thre, gr_cos_coef_thre, gr_wid_coef_thre, gr_act_coef_thre, class_thre, cfg):
    class_thre, idx = class_thre.sort(1, descending=True)  # [80, 64 (the number of kept boxes)]
    print("1111111111111111")
    print(class_thre.shape)
    print(class_thre)
    print("==============")

    idx = idx[:, :cfg.top_k]
    class_thre = class_thre[:, :cfg.top_k]

    print("22222222222222")
    print(class_thre.shape)
    print(class_thre)
    print("==============")

    num_classes, num_dets = idx.size()
    print("idx: ", idx.shape)
    box_thre = box_thre[idx.reshape(-1), :].reshape(num_classes, num_dets, 4)  # [80, 64, 4]
    gr_pos_coef_thre = gr_pos_coef_thre[idx.reshape(-1), :].reshape(num_classes, num_dets, -1)  # [80, 64, 32]
    gr_sin_coef_thre = gr_sin_coef_thre[idx.reshape(-1), :].reshape(num_classes, num_dets, -1)  # [80, 64, 32]
    gr_cos_coef_thre = gr_cos_coef_thre[idx.reshape(-1), :].reshape(num_classes, num_dets, -1)  # [80, 64, 32]
    gr_wid_coef_thre = gr_wid_coef_thre[idx.reshape(-1), :].reshape(num_classes, num_dets, -1)  # [80, 64, 32]
    gr_act_coef_thre = gr_act_coef_thre[idx.reshape(-1), :, :].reshape(num_classes, num_dets, 11, -1)
    
    print("fast osgd nms", gr_pos_coef_thre.shape, gr_act_coef_thre.shape)
    
    iou = box_iou(box_thre, box_thre)
    iou.triu_(diagonal=1)
    iou_max, _ = iou.max(dim=1)
    keep = (iou_max <= cfg.nms_iou_thre)
    class_ids = torch.arange(num_classes, device=box_thre.device)[:, None].expand_as(keep)
    class_ids, box_nms, class_nms = class_ids[keep], box_thre[keep], class_thre[keep]
    pos_coef_nms, sin_coef_nms, cos_coef_nms, wid_coef_nms = gr_pos_coef_thre[keep], gr_sin_coef_thre[keep], gr_cos_coef_thre[keep], gr_wid_coef_thre[keep]
    act_coef_nms = gr_act_coef_thre[keep]

    class_nms, idx = class_nms.sort(0, descending=True)
    idx = idx[:cfg.max_detections]
    class_nms = class_nms[:cfg.max_detections]

    print("22222222222222")
    print(class_nms.shape)
    print(class_nms)
    print("==============")

    class_ids = class_ids[idx]
    box_nms = box_nms[idx]
    pos_coef_nms = pos_coef_nms[idx]
    sin_coef_nms = sin_coef_nms[idx]
    cos_coef_nms = cos_coef_nms[idx]
    wid_coef_nms = wid_coef_nms[idx]
    act_coef_nms = act_coef_nms[idx]

    print(pos_coef_nms.shape, act_coef_nms.shape)

    return box_nms, pos_coef_nms, sin_coef_nms, cos_coef_nms, wid_coef_nms, act_coef_nms, class_ids, class_nms


def fast_gr_nms_grasp_only(box_thre, gr_pos_coef_thre, gr_sin_coef_thre, gr_cos_coef_thre, gr_wid_coef_thre, class_thre, cfg):
    class_thre, idx = class_thre.sort(1, descending=True)  # [80, 64 (the number of kept boxes)]

    idx = idx[:, :cfg.top_k]
    class_thre = class_thre[:, :cfg.top_k]

    num_classes, num_dets = idx.size()
    box_thre = box_thre[idx.reshape(-1), :].reshape(num_classes, num_dets, 4)  # [80, 64, 4]
    gr_pos_coef_thre = gr_pos_coef_thre[idx.reshape(-1), :].reshape(num_classes, num_dets, -1)  # [80, 64, 32]
    gr_sin_coef_thre = gr_sin_coef_thre[idx.reshape(-1), :].reshape(num_classes, num_dets, -1)  # [80, 64, 32]
    gr_cos_coef_thre = gr_cos_coef_thre[idx.reshape(-1), :].reshape(num_classes, num_dets, -1)  # [80, 64, 32]
    gr_wid_coef_thre = gr_wid_coef_thre[idx.reshape(-1), :].reshape(num_classes, num_dets, -1)  # [80, 64, 32]

    iou = box_iou(box_thre, box_thre)
    iou.triu_(diagonal=1)
    iou_max, _ = iou.max(dim=1)

    # Now just filter out the ones higher than the threshold
    keep = (iou_max <= cfg.nms_iou_thre)

    # Assign each kept detection to its corresponding class
    class_ids = torch.arange(num_classes, device=box_thre.device)[:, None].expand_as(keep)

    class_ids, box_nms, class_nms = class_ids[keep], box_thre[keep], class_thre[keep]
    pos_coef_nms, sin_coef_nms, cos_coef_nms, wid_coef_nms = gr_pos_coef_thre[keep], gr_sin_coef_thre[keep], gr_cos_coef_thre[keep], gr_wid_coef_thre[keep]

    # Only keep the top cfg.max_num_detections highest scores across all classes
    class_nms, idx = class_nms.sort(0, descending=True)

    idx = idx[:cfg.max_detections]
    class_nms = class_nms[:cfg.max_detections]

    class_ids = class_ids[idx]
    box_nms = box_nms[idx]
    pos_coef_nms = pos_coef_nms[idx]
    sin_coef_nms = sin_coef_nms[idx]
    cos_coef_nms = cos_coef_nms[idx]
    wid_coef_nms = wid_coef_nms[idx]

    return box_nms, pos_coef_nms, sin_coef_nms, cos_coef_nms, wid_coef_nms, class_ids, class_nms


def fast_gr_nms(box_thre, coef_thre, gr_pos_coef_thre, gr_sin_coef_thre, gr_cos_coef_thre, gr_wid_coef_thre, class_thre, cfg):
    class_thre, idx = class_thre.sort(1, descending=True)  # [80, 64 (the number of kept boxes)]

    idx = idx[:, :cfg.top_k]
    class_thre = class_thre[:, :cfg.top_k]

    num_classes, num_dets = idx.size()
    box_thre = box_thre[idx.reshape(-1), :].reshape(num_classes, num_dets, 4)  # [80, 64, 4]
    coef_thre = coef_thre[idx.reshape(-1), :].reshape(num_classes, num_dets, -1)  # [80, 64, 32]
    gr_pos_coef_thre = gr_pos_coef_thre[idx.reshape(-1), :].reshape(num_classes, num_dets, -1)  # [80, 64, 32]
    gr_sin_coef_thre = gr_sin_coef_thre[idx.reshape(-1), :].reshape(num_classes, num_dets, -1)  # [80, 64, 32]
    gr_cos_coef_thre = gr_cos_coef_thre[idx.reshape(-1), :].reshape(num_classes, num_dets, -1)  # [80, 64, 32]
    gr_wid_coef_thre = gr_wid_coef_thre[idx.reshape(-1), :].reshape(num_classes, num_dets, -1)  # [80, 64, 32]

    iou = box_iou(box_thre, box_thre)
    iou.triu_(diagonal=1)
    iou_max, _ = iou.max(dim=1)

    # Now just filter out the ones higher than the threshold
    keep = (iou_max <= cfg.nms_iou_thre)

    # Assign each kept detection to its corresponding class
    class_ids = torch.arange(num_classes, device=box_thre.device)[:, None].expand_as(keep)

    class_ids, box_nms, coef_nms, class_nms = class_ids[keep], box_thre[keep], coef_thre[keep], class_thre[keep]
    pos_coef_nms, sin_coef_nms, cos_coef_nms, wid_coef_nms = gr_pos_coef_thre[keep], gr_sin_coef_thre[keep], gr_cos_coef_thre[keep], gr_wid_coef_thre[keep]

    # Only keep the top cfg.max_num_detections highest scores across all classes
    class_nms, idx = class_nms.sort(0, descending=True)

    idx = idx[:cfg.max_detections]
    class_nms = class_nms[:cfg.max_detections]

    class_ids = class_ids[idx]
    box_nms = box_nms[idx]
    coef_nms = coef_nms[idx]
    pos_coef_nms = pos_coef_nms[idx]
    sin_coef_nms = sin_coef_nms[idx]
    cos_coef_nms = cos_coef_nms[idx]
    wid_coef_nms = wid_coef_nms[idx]

    return box_nms, coef_nms, pos_coef_nms, sin_coef_nms, cos_coef_nms, wid_coef_nms, class_ids, class_nms




def fast_nms_numpy(box_thre, coef_thre, class_thre, cfg):
    # descending sort
    idx = np.argsort(-class_thre, axis=1)
    class_thre = np.sort(class_thre, axis=1)[:, ::-1]

    idx = idx[:, :cfg.top_k]
    class_thre = class_thre[:, :cfg.top_k]

    num_classes, num_dets = idx.shape
    box_thre = box_thre[idx.reshape(-1), :].reshape(num_classes, num_dets, 4)  # [80, 64, 4]
    coef_thre = coef_thre[idx.reshape(-1), :].reshape(num_classes, num_dets, -1)  # [80, 64, 32]

    iou = box_iou_numpy(box_thre, box_thre)
    iou = np.triu(iou, k=1)
    iou_max = np.max(iou, axis=1)

    # Now just filter out the ones higher than the threshold
    keep = (iou_max <= cfg.nms_iou_thre)

    # Assign each kept detection to its corresponding class
    class_ids = np.tile(np.arange(num_classes)[:, None], (1, keep.shape[1]))

    class_ids, box_nms, coef_nms, class_nms = class_ids[keep], box_thre[keep], coef_thre[keep], class_thre[keep]

    # Only keep the top cfg.max_num_detections highest scores across all classes
    idx = np.argsort(-class_nms, axis=0)
    class_nms = np.sort(class_nms, axis=0)[::-1]

    idx = idx[:cfg.max_detections]
    class_nms = class_nms[:cfg.max_detections]

    class_ids = class_ids[idx]
    box_nms = box_nms[idx]
    coef_nms = coef_nms[idx]

    return box_nms, coef_nms, class_ids, class_nms


def traditional_nms(boxes, masks, scores, cfg):
    num_classes = scores.size(0)

    idx_lst, cls_lst, scr_lst = [], [], []

    # Multiplying by max_size is necessary because of how cnms computes its area and intersections
    boxes = boxes * cfg.img_size

    for _cls in range(num_classes):
        cls_scores = scores[_cls, :]
        conf_mask = cls_scores > cfg.nms_score_thre
        idx = torch.arange(cls_scores.size(0), device=boxes.device)

        cls_scores = cls_scores[conf_mask]
        idx = idx[conf_mask]

        if cls_scores.size(0) == 0:
            continue

        preds = torch.cat([boxes[conf_mask], cls_scores[:, None]], dim=1).cpu().numpy()
        keep = cnms(preds, cfg.nms_iou_thre)
        keep = torch.tensor(keep, device=boxes.device).long()

        idx_lst.append(idx[keep])
        cls_lst.append(keep * 0 + _cls)
        scr_lst.append(cls_scores[keep])

    idx = torch.cat(idx_lst, dim=0)
    class_ids = torch.cat(cls_lst, dim=0)
    scores = torch.cat(scr_lst, dim=0)

    scores, idx2 = scores.sort(0, descending=True)
    idx2 = idx2[:cfg.max_detections]
    scores = scores[:cfg.max_detections]

    idx = idx[idx2]
    class_ids = class_ids[idx2]

    # Undo the multiplication above
    return boxes[idx] / cfg.img_size, masks[idx], class_ids, scores


def nms(class_pred, box_pred, coef_pred, proto_out, anchors, cfg):
    class_p = class_pred.squeeze()  # [19248, 81]
    box_p = box_pred.squeeze()  # [19248, 4]
    coef_p = coef_pred.squeeze()  # [19248, 32]
    proto_p = proto_out.squeeze()  # [138, 138, 32]

    if isinstance(anchors, list):
        anchors = torch.tensor(anchors, device=class_p.device).reshape(-1, 4)

    class_p = class_p.transpose(1, 0).contiguous()  # [81, 19248]

    # exclude the background class
    class_p = class_p[1:, :]
    # get the max score class of 19248 predicted boxes
    class_p_max, _ = torch.max(class_p, dim=0)  # [19248]

    # filter predicted boxes according the class score
    keep = (class_p_max > cfg.nms_score_thre)
    class_thre = class_p[:, keep]
    box_thre, anchor_thre, coef_thre = box_p[keep, :], anchors[keep, :], coef_p[keep, :]

    # decode boxes
    box_thre = torch.cat((anchor_thre[:, :2] + box_thre[:, :2] * 0.1 * anchor_thre[:, 2:],
                          anchor_thre[:, 2:] * torch.exp(box_thre[:, 2:] * 0.2)), 1)
    box_thre[:, :2] -= box_thre[:, 2:] / 2
    box_thre[:, 2:] += box_thre[:, :2]

    box_thre = torch.clip(box_thre, min=0., max=1.)

    if class_thre.shape[1] == 0:
        return None, None, None, None, None
    else:
        if not cfg.traditional_nms:
            box_thre, coef_thre, class_ids, class_thre = fast_nms(box_thre, coef_thre, class_thre, cfg)
        else:
            box_thre, coef_thre, class_ids, class_thre = traditional_nms(box_thre, coef_thre, class_thre, cfg)

        return class_ids, class_thre, box_thre, coef_thre, proto_p



def gr_nms_osgd(
    class_pred, box_pred, 
    proto_out, 
    gr_pos_coef_pred, gr_sin_coef_pred, gr_cos_coef_pred, gr_wid_coef_pred, gr_act_coef_pred,
    anchors, cfg):

    class_p = class_pred.squeeze()  # [19248, 81]

    box_p = box_pred.squeeze()  # [19248, 4]
    proto_p = proto_out.squeeze()  # [138, 138, 32]

    gr_pos_coef_p = gr_pos_coef_pred.squeeze()
    gr_sin_coef_p = gr_sin_coef_pred.squeeze()
    gr_cos_coef_p = gr_cos_coef_pred.squeeze()
    gr_wid_coef_p = gr_wid_coef_pred.squeeze()
    gr_act_coef_p = gr_act_coef_pred.squeeze()

    

    if isinstance(anchors, list):
        anchors = torch.tensor(anchors, device=class_p.device).reshape(-1, 4)

    class_p = class_p.transpose(1, 0).contiguous()
    class_p = class_p[1:, :]
    class_p_max, _ = torch.max(class_p, dim=0)

    keep = (class_p_max > cfg.nms_score_thre)

    class_thre = class_p[:, keep]
    print(class_thre.shape)
    print(class_thre)
    print("==============")

    box_thre, anchor_thre = box_p[keep, :], anchors[keep, :]
    gr_pos_coef_thre = gr_pos_coef_p[keep, :]
    gr_sin_coef_thre = gr_sin_coef_p[keep, :]
    gr_cos_coef_thre = gr_cos_coef_p[keep, :]
    gr_wid_coef_thre = gr_wid_coef_p[keep, :]
    gr_act_coef_thre = gr_act_coef_p[keep, :, :]

    box_thre = torch.cat((anchor_thre[:, :2] + box_thre[:, :2] * 0.1 * anchor_thre[:, 2:],
                          anchor_thre[:, 2:] * torch.exp(box_thre[:, 2:] * 0.2)), 1)

    box_thre = torch.clip(box_thre, min=0., max=1.)


    print(gr_act_coef_thre.shape)
    print(gr_pos_coef_thre.shape)

    if class_thre.shape[1] == 0:
        return None, None, None, None, None, None, None, None, None, None

    else:
        box_thre, pos_coef_thre, sin_coef_thre, cos_coef_thre, wid_coef_thre, act_coef_thre, class_ids, class_thre = fast_osgd_nms(box_thre, gr_pos_coef_thre, gr_sin_coef_thre, gr_cos_coef_thre, gr_wid_coef_thre, gr_act_coef_thre, class_thre, cfg)
        
        return class_ids, class_thre, box_thre, pos_coef_thre, sin_coef_thre, cos_coef_thre, wid_coef_thre, act_coef_thre, proto_p


def gr_nms_v2_grasp_only(
    class_pred, box_pred, 
    proto_out, 
    gr_pos_coef_pred, gr_sin_coef_pred, gr_cos_coef_pred, gr_wid_coef_pred,
    anchors, cfg):

    class_p = class_pred.squeeze()  # [19248, 81]
    box_p = box_pred.squeeze()  # [19248, 4]
    proto_p = proto_out.squeeze()  # [138, 138, 32]

    gr_pos_coef_p = gr_pos_coef_pred.squeeze()
    gr_sin_coef_p = gr_sin_coef_pred.squeeze()
    gr_cos_coef_p = gr_cos_coef_pred.squeeze()
    gr_wid_coef_p = gr_wid_coef_pred.squeeze()


    if isinstance(anchors, list):
        anchors = torch.tensor(anchors, device=class_p.device).reshape(-1, 4)

    class_p = class_p.transpose(1, 0).contiguous()  # [81, 19248]

    # exclude the background class
    class_p = class_p[1:, :]
    # get the max score class of 19248 predicted boxes
    class_p_max, _ = torch.max(class_p, dim=0)  # [19248]

    # filter predicted boxes according the class score
    keep = (class_p_max > cfg.nms_score_thre)
    
    class_thre = class_p[:, keep]
    box_thre, anchor_thre = box_p[keep, :], anchors[keep, :]
    gr_pos_coef_thre = gr_pos_coef_p[keep, :]
    gr_sin_coef_thre = gr_sin_coef_p[keep, :]
    gr_cos_coef_thre = gr_cos_coef_p[keep, :]
    gr_wid_coef_thre = gr_wid_coef_p[keep, :]


    # decode boxes
    box_thre = torch.cat((anchor_thre[:, :2] + box_thre[:, :2] * 0.1 * anchor_thre[:, 2:],
                          anchor_thre[:, 2:] * torch.exp(box_thre[:, 2:] * 0.2)), 1)
    box_thre[:, :2] -= box_thre[:, 2:] / 2
    box_thre[:, 2:] += box_thre[:, :2]

    box_thre = torch.clip(box_thre, min=0., max=1.)

    if class_thre.shape[1] == 0:
        return None, None, None, None, None, None, None, None
    else:
        if not cfg.traditional_nms:
            box_thre, pos_coef_thre, sin_coef_thre, cos_coef_thre, wid_coef_thre, class_ids, class_thre = fast_gr_nms_grasp_only(
                box_thre,  
                gr_pos_coef_thre, gr_sin_coef_thre, gr_cos_coef_thre, gr_wid_coef_thre, 
                class_thre, cfg
            )
        else:
            box_thre, coef_thre, class_ids, class_thre = traditional_nms(box_thre, coef_thre, class_thre, cfg)

        return class_ids, class_thre, box_thre, pos_coef_thre, sin_coef_thre, cos_coef_thre, wid_coef_thre, proto_p




def gr_nms_v2(
    class_pred, box_pred, 
    coef_pred, proto_out, 
    gr_pos_coef_pred, gr_sin_coef_pred, gr_cos_coef_pred, gr_wid_coef_pred,
    anchors, cfg):

    class_p = class_pred.squeeze()  # [19248, 81]
    box_p = box_pred.squeeze()  # [19248, 4]
    coef_p = coef_pred.squeeze()  # [19248, 32]
    proto_p = proto_out.squeeze()  # [138, 138, 32]

    gr_pos_coef_p = gr_pos_coef_pred.squeeze()
    gr_sin_coef_p = gr_sin_coef_pred.squeeze()
    gr_cos_coef_p = gr_cos_coef_pred.squeeze()
    gr_wid_coef_p = gr_wid_coef_pred.squeeze()


    if isinstance(anchors, list):
        anchors = torch.tensor(anchors, device=class_p.device).reshape(-1, 4)

    class_p = class_p.transpose(1, 0).contiguous()  # [81, 19248]

    # exclude the background class
    class_p = class_p[1:, :]
    # get the max score class of 19248 predicted boxes
    class_p_max, _ = torch.max(class_p, dim=0)  # [19248]

    # filter predicted boxes according the class score
    keep = (class_p_max > cfg.nms_score_thre)
    
    class_thre = class_p[:, keep]
    box_thre, anchor_thre, coef_thre = box_p[keep, :], anchors[keep, :], coef_p[keep, :]
    gr_pos_coef_thre = gr_pos_coef_p[keep, :]
    gr_sin_coef_thre = gr_sin_coef_p[keep, :]
    gr_cos_coef_thre = gr_cos_coef_p[keep, :]
    gr_wid_coef_thre = gr_wid_coef_p[keep, :]


    # decode boxes
    box_thre = torch.cat((anchor_thre[:, :2] + box_thre[:, :2] * 0.1 * anchor_thre[:, 2:],
                          anchor_thre[:, 2:] * torch.exp(box_thre[:, 2:] * 0.2)), 1)
    box_thre[:, :2] -= box_thre[:, 2:] / 2
    box_thre[:, 2:] += box_thre[:, :2]

    box_thre = torch.clip(box_thre, min=0., max=1.)

    if class_thre.shape[1] == 0:
        return None, None, None, None, None, None, None, None, None
    else:
        if not cfg.traditional_nms:
            box_thre, coef_thre, pos_coef_thre, sin_coef_thre, cos_coef_thre, wid_coef_thre, class_ids, class_thre = fast_gr_nms(
                box_thre, coef_thre, 
                gr_pos_coef_thre, gr_sin_coef_thre, gr_cos_coef_thre, gr_wid_coef_thre, 
                class_thre, cfg
            )
        else:
            box_thre, coef_thre, class_ids, class_thre = traditional_nms(box_thre, coef_thre, class_thre, cfg)

        return class_ids, class_thre, box_thre, coef_thre, pos_coef_thre, sin_coef_thre, cos_coef_thre, wid_coef_thre, proto_p


def gr_nms(
    class_pred, box_pred, 
    coef_pred, proto_out, 
    gr_pos_coef_pred, gr_ang_coef_pred, gr_wid_coef_pred, gr_proto_out, 
    anchors, cfg):

    class_p = class_pred.squeeze()  # [19248, 81]
    box_p = box_pred.squeeze()  # [19248, 4]
    coef_p = coef_pred.squeeze()  # [19248, 32]
    proto_p = proto_out.squeeze()  # [138, 138, 32]

    gr_proto_p = gr_proto_out.squeeze()
    gr_pos_coef_p = gr_pos_coef_pred.squeeze()
    gr_ang_coef_p = gr_ang_coef_pred.squeeze()
    gr_wid_coef_p = gr_wid_coef_pred.squeeze()


    if isinstance(anchors, list):
        anchors = torch.tensor(anchors, device=class_p.device).reshape(-1, 4)

    class_p = class_p.transpose(1, 0).contiguous()  # [81, 19248]

    # exclude the background class
    class_p = class_p[1:, :]
    # get the max score class of 19248 predicted boxes
    class_p_max, _ = torch.max(class_p, dim=0)  # [19248]

    # filter predicted boxes according the class score
    keep = (class_p_max > cfg.nms_score_thre)
    class_thre = class_p[:, keep]
    box_thre, anchor_thre, coef_thre = box_p[keep, :], anchors[keep, :], coef_p[keep, :]
    gr_pos_coef_thre = gr_pos_coef_p[keep, :]
    gr_sin_coef_thre = gr_ang_coef_p[keep, :]
    gr_wid_coef_thre = gr_wid_coef_p[keep, :]


    # decode boxes
    box_thre = torch.cat((anchor_thre[:, :2] + box_thre[:, :2] * 0.1 * anchor_thre[:, 2:],
                          anchor_thre[:, 2:] * torch.exp(box_thre[:, 2:] * 0.2)), 1)
    box_thre[:, :2] -= box_thre[:, 2:] / 2
    box_thre[:, 2:] += box_thre[:, :2]

    box_thre = torch.clip(box_thre, min=0., max=1.)

    if class_thre.shape[1] == 0:
        return None, None, None, None, None
    else:
        if not cfg.traditional_nms:
            box_thre, coef_thre, pos_coef_thre, ang_coef_thre, wid_coef_thre, class_ids, class_thre = fast_gr_nms(
                box_thre, coef_thre, 
                gr_pos_coef_thre, gr_sin_coef_thre, gr_wid_coef_thre, 
                class_thre, cfg
            )
        else:
            box_thre, coef_thre, class_ids, class_thre = traditional_nms(box_thre, coef_thre, class_thre, cfg)

        return class_ids, class_thre, box_thre, coef_thre, pos_coef_thre, ang_coef_thre, wid_coef_thre, proto_p, gr_proto_p


def nms_numpy(class_pred, box_pred, coef_pred, proto_out, anchors, cfg):
    class_p = class_pred.squeeze()  # [19248, 81]
    box_p = box_pred.squeeze()  # [19248, 4]
    coef_p = coef_pred.squeeze()  # [19248, 32]
    proto_p = proto_out.squeeze()  # [138, 138, 32]
    anchors = np.array(anchors).reshape(-1, 4)

    class_p = class_p.transpose(1, 0)
    # exclude the background class
    class_p = class_p[1:, :]
    # get the max score class of 19248 predicted boxes

    class_p_max = np.max(class_p, axis=0)  # [19248]

    # filter predicted boxes according the class score
    keep = (class_p_max > cfg.nms_score_thre)
    class_thre = class_p[:, keep]

    box_thre, anchor_thre, coef_thre = box_p[keep, :], anchors[keep, :], coef_p[keep, :]

    # decode boxes
    box_thre = np.concatenate((anchor_thre[:, :2] + box_thre[:, :2] * 0.1 * anchor_thre[:, 2:],
                               anchor_thre[:, 2:] * np.exp(box_thre[:, 2:] * 0.2)), axis=1)
    box_thre[:, :2] -= box_thre[:, 2:] / 2
    box_thre[:, 2:] += box_thre[:, :2]

    if class_thre.shape[1] == 0:
        return None, None, None, None, None
    else:
        assert not cfg.traditional_nms, 'Traditional nms is not supported with numpy.'
        box_thre, coef_thre, class_ids, class_thre = fast_nms_numpy(box_thre, coef_thre, class_thre, cfg)
        return class_ids, class_thre, box_thre, coef_thre, proto_p


def after_nms(ids_p, class_p, box_p, coef_p, proto_p, img_h, img_w, cfg=None, img_name=None):
    if ids_p is None:
        return None, None, None, None

    if cfg and cfg.visual_thre > 0:
        keep = class_p >= cfg.visual_thre
        if not keep.any():
            return None, None, None, None

        ids_p = ids_p[keep]
        class_p = class_p[keep]
        box_p = box_p[keep]
        coef_p = coef_p[keep]

    if cfg and cfg.save_lincomb:
        draw_lincomb(proto_p, coef_p, img_name)

    masks = torch.sigmoid(torch.matmul(proto_p, coef_p.t()))

    if not cfg or not cfg.no_crop:  # Crop masks by box_p
        masks = crop(masks, box_p)

    masks = masks.permute(2, 0, 1).contiguous()

    ori_size = max(img_h, img_w)
    # in OpenCV, cv2.resize is `align_corners=False`.
    masks = F.interpolate(masks.unsqueeze(0), (ori_size, ori_size), mode='bilinear', align_corners=False).squeeze(0)
    masks.gt_(0.5)  # Binarize the masks because of interpolation.
    masks = masks[:, 0: img_h, :] if img_h < img_w else masks[:, :, 0: img_w]

    box_p *= ori_size
    box_p = box_p.int()

    return ids_p, class_p, box_p, masks


def after_nms_numpy(ids_p, class_p, box_p, coef_p, proto_p, img_h, img_w, cfg=None):
    def np_sigmoid(x):
        return 1 / (1 + np.exp(-x))

    if ids_p is None:
        return None, None, None, None

    if cfg and cfg.visual_thre > 0:
        keep = class_p >= cfg.visual_thre
        if not keep.any():
            return None, None, None, None

        ids_p = ids_p[keep]
        class_p = class_p[keep]
        box_p = box_p[keep]
        coef_p = coef_p[keep]

    assert not cfg.save_lincomb, 'save_lincomb is not supported in onnx mode.'

    masks = np_sigmoid(np.matmul(proto_p, coef_p.T))

    if not cfg or not cfg.no_crop:  # Crop masks by box_p
        masks = crop_numpy(masks, box_p)

    ori_size = max(img_h, img_w)
    masks = cv2.resize(masks, (ori_size, ori_size), interpolation=cv2.INTER_LINEAR)

    if masks.ndim == 2:
        masks = masks[:, :, None]

    masks = np.transpose(masks, (2, 0, 1))
    masks = masks > 0.5  # Binarize the masks because of interpolation.
    masks = masks[:, 0: img_h, :] if img_h < img_w else masks[:, :, 0: img_w]

    box_p *= ori_size
    box_p = box_p.astype('int32')

    return ids_p, class_p, box_p, masks



def gr_post_processing_osgd(depth, ids_p, class_p, box_p, pos_coef_p, sin_coef_p, cos_coef_p, wid_coef_p, act_coef_p, proto_p, num_grasp_per_object=1, per_object_width=None, ori_w=544, ori_h=544):
    keep = (class_p >= 0.7)
    if not keep.any():
        print("No valid instance")
    else:
        ids_p = ids_p[keep]
        class_p = class_p[keep]
        box_p = box_p[keep]
        pos_coef_p = pos_coef_p[keep]
        sin_coef_p = sin_coef_p[keep]
        cos_coef_p = cos_coef_p[keep]
        wid_coef_p = wid_coef_p[keep]
        act_coef_p = act_coef_p[keep]

    ids_p = (ids_p + 1)
    ids_p = ids_p.cpu().numpy()
    box_p = box_p

    pos_masks = torch.sigmoid(torch.matmul(proto_p, pos_coef_p.t())).contiguous()
    pos_masks = crop(pos_masks, box_p).permute(2,0,1)

    sin_masks = torch.matmul(proto_p, sin_coef_p.t()).contiguous()
    sin_masks = crop(sin_masks, box_p).permute(2,0,1)

    cos_masks = torch.matmul(proto_p, cos_coef_p.t()).contiguous()
    cos_masks = crop(cos_masks, box_p).permute(2,0,1)

    wid_masks = torch.sigmoid(torch.matmul(proto_p, wid_coef_p.t())).permute(2,0,1).contiguous()
    # wid_masks = crop(wid_masks, box_p).permute(2,0,1)
    wid_masks = wid_masks * pos_masks

    print(proto_p.shape, act_coef_p.shape)
    print(pos_coef_p.shape)
    act_masks = []
    for i in range(11):
        act_mask = torch.sigmoid(proto_p @ act_coef_p[:,i,:].squeeze().t())
        act_mask = crop(act_mask, box_p).permute(2, 0, 1).unsqueeze(1)
        act_masks.append(act_mask)
    
    act_masks = torch.cat(act_masks, dim=1)
    print(act_masks.shape)
        
    pos_masks = F.interpolate(pos_masks.unsqueeze(0), (ori_w, ori_w), mode='bilinear', align_corners=False).squeeze(0)
    sin_masks = F.interpolate(sin_masks.unsqueeze(0), (ori_w, ori_w), mode='bilinear', align_corners=False).squeeze(0)
    cos_masks = F.interpolate(cos_masks.unsqueeze(0), (ori_w, ori_w), mode='bilinear', align_corners=False).squeeze(0)
    wid_masks = F.interpolate(wid_masks.unsqueeze(0), (ori_w, ori_w), mode='bilinear', align_corners=False).squeeze(0)
    act_masks = F.interpolate(act_masks, (ori_w, ori_w), mode='bilinear', align_corners=False)

    box_p = box_p.cpu().numpy()
    pos_masks = pos_masks.cpu().numpy()
    wid_masks = wid_masks.cpu().numpy()

    ang_masks = []

    for i in range(pos_masks.shape[0]):
        pos_masks[i] = gaussian(pos_masks[i], 2.0, preserve_range=True)
        # ang_masks[i] = gaussian(ang_masks[i], 2.0, preserve_range=True)
        # wid_masks[i] = gaussian(wid_masks[i], 1.0, preserve_range=True)
        ang_mask = (torch.atan2(sin_masks[i], cos_masks[i]) / 2.0).cpu().numpy().squeeze()
        ang_masks.append(ang_mask)
    
    grasps = detect_grasps(pos_masks, ang_masks, wid_masks, ids_p, num_peaks=num_grasp_per_object, per_object_width=per_object_width)

    return ids_p, box_p, pos_masks, ang_masks, wid_masks, act_masks, grasps

def gr_post_processing(img, depth, ids_p, class_p, box_p, coef_p, pos_coef_p, sin_coef_p, cos_coef_p, wid_coef_p, proto_p, ori_h, ori_w, visualize_lincomb=False, visualize_results=False, num_grasp_per_object=1, per_object_width=None, target_dir=None):
    keep = (class_p >= 0.3)
    if not keep.any():
        print("No valid instance")
    else:
        ids_p = ids_p[keep]
        class_p = class_p[keep]
        box_p = box_p[keep]
        coef_p = coef_p[keep]   
        pos_coef_p = pos_coef_p[keep]
        sin_coef_p = sin_coef_p[keep]
        cos_coef_p = cos_coef_p[keep]
        wid_coef_p = wid_coef_p[keep]
    
    ids_p = (ids_p + 1)
    ids_p = ids_p.cpu().numpy()
    box_p = box_p

    ones_coef = torch.ones(pos_coef_p.shape).float().cuda()

    if visualize_lincomb:
        # print("ProtoTypes")
        draw_lincomb(ids_p, proto_p, ones_coef, "prototypes.png", target_dir)

        # print("Semantic")
        draw_lincomb(ids_p, proto_p, coef_p, "cogr-sem.png", target_dir)
        # print("Grasp pos")
        draw_lincomb(ids_p, proto_p, pos_coef_p, "cogr-gr-pos.png", target_dir)
        # print("Grasp sin")
        draw_lincomb(ids_p, proto_p, sin_coef_p, "cogr-gr-sin.png", target_dir)
        # print("Grasp cos")
        draw_lincomb(ids_p, proto_p, cos_coef_p, "cogr-gr-cos.png", target_dir)
        # print("Grasp wid")
        draw_lincomb(ids_p, proto_p, wid_coef_p, "cogr-gr-wid.png", target_dir)

    instance_masks = torch.sigmoid(torch.matmul(proto_p, coef_p.t())).contiguous()
    # print("Instance masks: ", instance_masks.shape)
    # vis_masks = (instance_masks.clone().cpu().numpy()[:,:,-1] * 255).astype('uint8')
    # print(vis_masks.shape)
    # vis_masks = cv2.applyColorMap(vis_masks, cv2.COLORMAP_WINTER)
    # cv2.imwrite("results/images/vis_masks.png", vis_masks)
    
    pos_masks = torch.sigmoid(torch.matmul(proto_p, pos_coef_p.t())).contiguous()
    sin_masks = torch.matmul(proto_p, sin_coef_p.t()).contiguous()
    cos_masks = torch.matmul(proto_p, cos_coef_p.t()).contiguous()
    wid_masks = torch.sigmoid(torch.matmul(proto_p, wid_coef_p.t())).permute(2,0,1).contiguous()


    # for i in range(ids_p.shape[0]):
    #     target_id = int(ids_p[i])

    #     plt.figure()
    #     ax = plt.gca()
    #     im = ax.imshow(instance_masks[:,:,i].squeeze().cpu().numpy())
    #     plt.axis("off")
    #     plt.savefig('{}/{}/ori_ins_mask.png'.format(target_dir, target_id), bbox_inches='tight',pad_inches = 0)
    #     plt.cla()
    #     plt.clf()

    #     plt.figure()
    #     ax = plt.gca()
    #     im = ax.imshow(pos_masks[:,:,i].squeeze().cpu().numpy(), cmap='jet', vmin=0, vmax=1)
    #     plt.axis("off")
    #     divider = make_axes_locatable(ax)
    #     cax = divider.append_axes("right", size="5%", pad=0.05)
    #     cbar = plt.colorbar(im, cax=cax)
    #     for t in cbar.ax.get_yticklabels():
    #         t.set_fontsize(20)
    #     plt.savefig('{}/{}/ori_pos_mas.png'.format(target_dir, target_id), bbox_inches='tight',pad_inches = 0)
    #     plt.cla()
    #     plt.clf()


    #     plt.figure()
    #     ax = plt.gca()
    #     im = ax.imshow(sin_masks[:,:,i].squeeze().cpu().numpy(), cmap='rainbow', vmin=-1, vmax=1)
    #     plt.axis("off")
    #     divider = make_axes_locatable(ax)
    #     cax = divider.append_axes("right", size="5%", pad=0.05)
    #     cbar = plt.colorbar(im, cax=cax)
    #     for t in cbar.ax.get_yticklabels():
    #         t.set_fontsize(20)
    #     plt.savefig('{}/{}/ori_ang_sin_mask.png'.format(target_dir, target_id), bbox_inches='tight',pad_inches = 0)
    #     plt.cla()
    #     plt.clf()

    #     plt.figure()
    #     ax = plt.gca()
    #     im = ax.imshow(cos_masks[:,:,i].squeeze().cpu().numpy(), cmap='jet', vmin=-1, vmax=1)
    #     plt.axis("off")
    #     divider = make_axes_locatable(ax)
    #     cax = divider.append_axes("right", size="5%", pad=0.05)
    #     cbar = plt.colorbar(im, cax=cax)
    #     for t in cbar.ax.get_yticklabels():
    #         t.set_fontsize(20)
    #     plt.savefig('{}/{}/ori_ang_cos_mask.png'.format(target_dir, target_id), bbox_inches='tight',pad_inches = 0)
    #     plt.cla()
    #     plt.clf()

    #     ang_mask = (torch.atan2(sin_masks[:,:,i], cos_masks[:,:,i]) / 2.0).cpu().numpy().squeeze()
    #     plt.figure()
    #     ax = plt.gca()
    #     im = ax.imshow(ang_mask, cmap='jet', vmin=-np.pi/2, vmax=np.pi/2)
    #     plt.axis("off")
    #     divider = make_axes_locatable(ax)
    #     cax = divider.append_axes("right", size="5%", pad=0.05)
    #     cbar = plt.colorbar(im, cax=cax)
    #     for t in cbar.ax.get_yticklabels():
    #         t.set_fontsize(20)
    #     plt.savefig('{}/{}/ori_ang_mask.png'.format(target_dir, target_id), bbox_inches='tight',pad_inches = 0)
    #     plt.cla()
    #     plt.clf()

    #     plt.figure()
    #     ax = plt.gca()
    #     im = ax.imshow(wid_masks[i,:,:].squeeze().cpu().numpy(), cmap='jet', vmin=0, vmax=1)
    #     plt.axis("off")
    #     divider = make_axes_locatable(ax)
    #     cax = divider.append_axes("right", size="5%", pad=0.05)
    #     cbar = plt.colorbar(im, cax=cax)
    #     for t in cbar.ax.get_yticklabels():
    #         t.set_fontsize(20)
    #     plt.savefig('{}/{}/ori_wid_mas.png'.format(target_dir, target_id), bbox_inches='tight',pad_inches = 0)
    #     plt.cla()
    #     plt.clf()


        

    instance_masks = crop(instance_masks, box_p).permute(2,0,1)
    pos_masks = crop(pos_masks, box_p).permute(2,0,1)
    sin_masks = crop(sin_masks, box_p).permute(2,0,1)
    cos_masks = crop(cos_masks, box_p).permute(2,0,1)
    wid_masks = wid_masks * pos_masks
    # wid_masks = crop(wid_masks, box_p).permute(2,0,1)
    

    instance_masks = F.interpolate(instance_masks.unsqueeze(0), (ori_w, ori_w), mode='bilinear', align_corners=False).squeeze(0)
    instance_masks.gt_(0.5)
    pos_masks = F.interpolate(pos_masks.unsqueeze(0), (ori_w, ori_w), mode='bilinear', align_corners=False).squeeze(0)
    sin_masks = F.interpolate(sin_masks.unsqueeze(0), (ori_w, ori_w), mode='bilinear', align_corners=False).squeeze(0)
    cos_masks = F.interpolate(cos_masks.unsqueeze(0), (ori_w, ori_w), mode='bilinear', align_corners=False).squeeze(0)
    wid_masks = F.interpolate(wid_masks.unsqueeze(0), (ori_w, ori_w), mode='bilinear', align_corners=False).squeeze(0)
    
    # Convert processed image to original size

    img = cv2.resize(img, (ori_w, ori_w))
    depth = cv2.resize(depth, (ori_w, ori_w))

    ori_img = img[0:ori_h, 0:ori_w, :]
    ori_img = ori_img * norm_std + norm_mean
    ori_depth = depth[0:ori_h, 0:ori_w]
    instance_masks = instance_masks[:, 0:ori_h, 0:ori_w]
    pos_masks = pos_masks[:, 0:ori_h, 0:ori_w]
    sin_masks = sin_masks[:, 0:ori_h, 0:ori_w]
    cos_masks = cos_masks[:, 0:ori_h, 0:ori_w]
    wid_masks = wid_masks[:, 0:ori_h, 0:ori_w]

    box_p = box_p.cpu().numpy()
    instance_masks = instance_masks.cpu().numpy()
    pos_masks = pos_masks.cpu().numpy()
    wid_masks = wid_masks.cpu().numpy()

    ang_masks = []

    for i in range(pos_masks.shape[0]):
        pos_masks[i] = gaussian(pos_masks[i], 2.0, preserve_range=True)
        # ang_masks[i] = gaussian(ang_masks[i], 2.0, preserve_range=True)
        # wid_masks[i] = gaussian(wid_masks[i], 1.0, preserve_range=True)
        ang_mask = (torch.atan2(sin_masks[i], cos_masks[i]) / 2.0).cpu().numpy().squeeze()
        ang_masks.append(ang_mask)
    

    ang_masks = np.array(ang_masks)


    scale = np.array([ori_w, ori_w, ori_w, ori_w])
    box_p *= scale
    box_p = np.concatenate([box_p, ids_p.reshape(-1,1)], axis=-1)

    grasps = detect_grasps(pos_masks, ang_masks, wid_masks, ids_p, num_peaks=num_grasp_per_object, per_object_width=per_object_width)

    if visualize_results:
        # instance_masks_np = instance_masks.sum(dim=0).cpu().numpy()
        instance_masks_np = instance_masks[-1,:,:]
        cv2.imwrite("results/images/sem_mask.png", instance_masks_np*255)

    return ori_img, ori_depth, box_p, instance_masks, grasps, pos_masks, ang_masks, wid_masks, ids_p




def gr_post_processing_grasp_only(img, depth, ids_p, class_p, box_p, pos_coef_p, sin_coef_p, cos_coef_p, wid_coef_p, proto_p, ori_h, ori_w, visualize_lincomb=False, visualize_results=False, num_grasp_per_object=1, per_object_width=None, target_dir=None):
    keep = (class_p >= 0.3)
    if not keep.any():
        print("No valid instance")
    else:
        ids_p = ids_p[keep]
        class_p = class_p[keep]
        box_p = box_p[keep]
        pos_coef_p = pos_coef_p[keep]
        sin_coef_p = sin_coef_p[keep]
        cos_coef_p = cos_coef_p[keep]
        wid_coef_p = wid_coef_p[keep]
    
    ids_p = (ids_p + 1)
    ids_p = ids_p.cpu().numpy()
    box_p = box_p

    ones_coef = torch.ones(pos_coef_p.shape).float().cuda()

    if visualize_lincomb:
        # print("ProtoTypes")
        draw_lincomb(ids_p, proto_p, ones_coef, "prototypes.png", target_dir)

        # print("Semantic")
        # print("Grasp pos")
        draw_lincomb(ids_p, proto_p, pos_coef_p, "cogr-gr-pos.png", target_dir)
        # print("Grasp sin")
        draw_lincomb(ids_p, proto_p, sin_coef_p, "cogr-gr-sin.png", target_dir)
        # print("Grasp cos")
        draw_lincomb(ids_p, proto_p, cos_coef_p, "cogr-gr-cos.png", target_dir)
        # print("Grasp wid")
        draw_lincomb(ids_p, proto_p, wid_coef_p, "cogr-gr-wid.png", target_dir)

    
    pos_masks = torch.sigmoid(torch.matmul(proto_p, pos_coef_p.t())).contiguous()
    sin_masks = torch.matmul(proto_p, sin_coef_p.t()).contiguous()
    cos_masks = torch.matmul(proto_p, cos_coef_p.t()).contiguous()
    wid_masks = torch.sigmoid(torch.matmul(proto_p, wid_coef_p.t())).permute(2,0,1).contiguous()


    pos_masks = crop(pos_masks, box_p).permute(2,0,1)
    sin_masks = crop(sin_masks, box_p).permute(2,0,1)
    cos_masks = crop(cos_masks, box_p).permute(2,0,1)
    wid_masks = wid_masks * pos_masks
    # wid_masks = crop(wid_masks, box_p).permute(2,0,1)
    

    pos_masks = F.interpolate(pos_masks.unsqueeze(0), (ori_w, ori_w), mode='bilinear', align_corners=False).squeeze(0)
    sin_masks = F.interpolate(sin_masks.unsqueeze(0), (ori_w, ori_w), mode='bilinear', align_corners=False).squeeze(0)
    cos_masks = F.interpolate(cos_masks.unsqueeze(0), (ori_w, ori_w), mode='bilinear', align_corners=False).squeeze(0)
    wid_masks = F.interpolate(wid_masks.unsqueeze(0), (ori_w, ori_w), mode='bilinear', align_corners=False).squeeze(0)
    
    # Convert processed image to original size

    img = cv2.resize(img, (ori_w, ori_w))
    depth = cv2.resize(depth, (ori_w, ori_w))

    ori_img = img[0:ori_h, 0:ori_w, :]
    ori_img = ori_img * norm_std + norm_mean
    ori_depth = depth[0:ori_h, 0:ori_w]
    pos_masks = pos_masks[:, 0:ori_h, 0:ori_w]
    sin_masks = sin_masks[:, 0:ori_h, 0:ori_w]
    cos_masks = cos_masks[:, 0:ori_h, 0:ori_w]
    wid_masks = wid_masks[:, 0:ori_h, 0:ori_w]

    box_p = box_p.cpu().numpy()
    pos_masks = pos_masks.cpu().numpy()
    wid_masks = wid_masks.cpu().numpy()

    ang_masks = []

    for i in range(pos_masks.shape[0]):
        pos_masks[i] = gaussian(pos_masks[i], 2.0, preserve_range=True)
        # ang_masks[i] = gaussian(ang_masks[i], 2.0, preserve_range=True)
        # wid_masks[i] = gaussian(wid_masks[i], 1.0, preserve_range=True)
        ang_mask = (torch.atan2(sin_masks[i], cos_masks[i]) / 2.0).cpu().numpy().squeeze()
        ang_masks.append(ang_mask)
    

    ang_masks = np.array(ang_masks)


    scale = np.array([ori_w, ori_w, ori_w, ori_w])
    box_p *= scale
    box_p = np.concatenate([box_p, ids_p.reshape(-1,1)], axis=-1)

    grasps = detect_grasps(pos_masks, ang_masks, wid_masks, ids_p, num_peaks=num_grasp_per_object, per_object_width=per_object_width)


    return ori_img, ori_depth, box_p, grasps, pos_masks, ang_masks, wid_masks, ids_p



def gr_post_processing_jacquard(img, depth, ids_p, class_p, box_p, coef_p, pos_coef_p, sin_coef_p, cos_coef_p, wid_coef_p, proto_p, ori_h, ori_w, visualize_lincomb=False, visualize_results=False, num_grasp_per_object=1, target_dir=None):
    keep = (class_p >= 0.3)
    if not keep.any():
        print("No valid instance")
    else:
        ids_p = ids_p[keep]
        class_p = class_p[keep]
        box_p = box_p[keep]
        coef_p = coef_p[keep]   
        pos_coef_p = pos_coef_p[keep]
        sin_coef_p = sin_coef_p[keep]
        cos_coef_p = cos_coef_p[keep]
        wid_coef_p = wid_coef_p[keep]
    
    ids_p = (ids_p + 1)
    ids_p = ids_p.cpu().numpy()
    box_p = box_p

    ones_coef = torch.ones(pos_coef_p.shape).float().cuda()

    if visualize_lincomb:
        # print("ProtoTypes")
        draw_lincomb(proto_p, ones_coef, "prototypes.png", target_dir)

        # print("Semantic")
        draw_lincomb(proto_p, coef_p, "cogr-sem.png", target_dir)
        # print("Grasp pos")
        draw_lincomb(proto_p, pos_coef_p, "cogr-gr-pos.png", target_dir)
        # print("Grasp sin")
        draw_lincomb(proto_p, sin_coef_p, "cogr-gr-sin.png", target_dir)
        # print("Grasp cos")
        draw_lincomb(proto_p, cos_coef_p, "cogr-gr-cos.png", target_dir)
        # print("Grasp wid")
        draw_lincomb(proto_p, wid_coef_p, "cogr-gr-wid.png", target_dir)

    instance_masks = torch.sigmoid(torch.matmul(proto_p, coef_p.t())).contiguous()
    # print("Instance masks: ", instance_masks.shape)
    # vis_masks = (instance_masks.clone().cpu().numpy()[:,:,-1] * 255).astype('uint8')
    # print(vis_masks.shape)
    # vis_masks = cv2.applyColorMap(vis_masks, cv2.COLORMAP_WINTER)
    # cv2.imwrite("results/images/vis_masks.png", vis_masks)
    instance_masks = crop(instance_masks, box_p).permute(2,0,1)


    pos_masks = torch.sigmoid(torch.matmul(proto_p, pos_coef_p.t())).contiguous()
    pos_masks = crop(pos_masks, box_p).permute(2,0,1)

    sin_masks = torch.matmul(proto_p, sin_coef_p.t()).contiguous()
    sin_masks = crop(sin_masks, box_p).permute(2,0,1)

    cos_masks = torch.matmul(proto_p, cos_coef_p.t()).contiguous()
    cos_masks = crop(cos_masks, box_p).permute(2,0,1)

    wid_masks = torch.sigmoid(torch.matmul(proto_p, wid_coef_p.t())).permute(2,0,1).contiguous()
    # wid_masks = crop(wid_masks, box_p).permute(2,0,1)
    wid_masks = wid_masks * pos_masks

    instance_masks = F.interpolate(instance_masks.unsqueeze(0), (ori_w, ori_w), mode='bilinear', align_corners=False).squeeze(0)
    instance_masks.gt_(0.5)
    pos_masks = F.interpolate(pos_masks.unsqueeze(0), (ori_w, ori_w), mode='bilinear', align_corners=False).squeeze(0)
    sin_masks = F.interpolate(sin_masks.unsqueeze(0), (ori_w, ori_w), mode='bilinear', align_corners=False).squeeze(0)
    cos_masks = F.interpolate(cos_masks.unsqueeze(0), (ori_w, ori_w), mode='bilinear', align_corners=False).squeeze(0)
    wid_masks = F.interpolate(wid_masks.unsqueeze(0), (ori_w, ori_w), mode='bilinear', align_corners=False).squeeze(0)
    
    # Convert processed image to original size

    img = cv2.resize(img, (ori_w, ori_w))
    depth = cv2.resize(depth, (ori_w, ori_w))

    ori_img = img[0:ori_h, 0:ori_w, :]
    ori_img = ori_img * norm_std + norm_mean
    ori_depth = depth[0:ori_h, 0:ori_w]
    instance_masks = instance_masks[:, 0:ori_h, 0:ori_w]
    pos_masks = pos_masks[:, 0:ori_h, 0:ori_w]
    sin_masks = sin_masks[:, 0:ori_h, 0:ori_w]
    cos_masks = cos_masks[:, 0:ori_h, 0:ori_w]
    wid_masks = wid_masks[:, 0:ori_h, 0:ori_w]

    box_p = box_p.cpu().numpy()
    instance_masks = instance_masks.cpu().numpy()
    pos_masks = pos_masks.cpu().numpy()
    wid_masks = wid_masks.cpu().numpy()

    ang_masks = []

    for i in range(pos_masks.shape[0]):
        pos_masks[i] = gaussian(pos_masks[i], 2.0, preserve_range=True)
        # ang_masks[i] = gaussian(ang_masks[i], 2.0, preserve_range=True)
        # wid_masks[i] = gaussian(wid_masks[i], 1.0, preserve_range=True)
        ang_mask = (torch.atan2(sin_masks[i], cos_masks[i]) / 2.0).cpu().numpy().squeeze()
        ang_masks.append(ang_mask)
    

    ang_masks = np.array(ang_masks)


    scale = np.array([ori_w, ori_w, ori_w, ori_w])
    box_p *= scale
    box_p = np.concatenate([box_p, ids_p.reshape(-1,1)], axis=-1)

    size = min(box_p[0][3]-box_p[0][1], box_p[0][2]-box_p[0][0])

    grasps = detect_grasps(pos_masks, ang_masks, wid_masks, ids_p, num_peaks=num_grasp_per_object, per_object_width=[size, size])

    if visualize_results:
        # instance_masks_np = instance_masks.sum(dim=0).cpu().numpy()
        instance_masks_np = instance_masks[-1,:,:]
        cv2.imwrite("results/images/sem_mask.png", instance_masks_np*255)

    return ori_img, ori_depth, box_p, instance_masks, grasps, pos_masks, ang_masks, wid_masks, ids_p



def detect_grasps(pos_masks, ang_masks, wid_masks, cls_ids, min_distance=2, threshold_abs=0.4, num_peaks=5, per_object_width=None):
    if per_object_width is None:
        from config import PER_CLASS_MAX_GRASP_WIDTH
    else:
        PER_CLASS_MAX_GRASP_WIDTH = per_object_width

    # assert cls_ids.shape[0] == pos_masks.shape[0] == ang_masks.shape[0] == wid_masks.shape[0]
    grasps = []
    for i in range(cls_ids.shape[0]):
        tmp = []
        cls_id = cls_ids[i]-1
        max_width = PER_CLASS_MAX_GRASP_WIDTH[cls_id]

        pos_mask = np.array(pos_masks[i], dtype='float')
        
        local_max = peak_local_max(pos_mask, min_distance=min_distance, threshold_abs=threshold_abs, num_peaks=num_peaks)

        for p_array in local_max:
            grasp_point = tuple(p_array)
            grasp_angle = ang_masks[i][grasp_point] / np.pi * 180
            # if grasp_angle > 0:
            #     grasp_angle = grasp_angle - 90
            # elif grasp_angle < 0:
            #     grasp_angle = grasp_angle + 90
            grasp_width = wid_masks[i][grasp_point]
            tmp.append([float(grasp_point[1]), float(grasp_point[0]), grasp_width*max_width, 20, grasp_angle, int(cls_ids[i])])

        grasps.append(tmp)
    
    return grasps


def calculate_iou(rect_p, rect_gt, shape=(480, 640), angle_threshold=30):
    if abs(rect_p[4] - rect_gt[4]) > angle_threshold:
        return 0
    
    center_x, center_y, w_rect, h_rect, theta, cls_id = rect_gt
    gt_r_rect = ((center_x, center_y), (w_rect, h_rect), -theta)
    gt_box = cv2.boxPoints(gt_r_rect)
    gt_box = np.int0(gt_box)
    rr1, cc1 = polygon(gt_box[:, 0], gt_box[:,1], shape)

    mask_rr = rr1 < shape[1]
    rr1 = rr1[mask_rr]
    cc1 = cc1[mask_rr]

    mask_cc = cc1 < shape[0]
    cc1 = cc1[mask_cc]
    rr1 = rr1[mask_cc]

    center_x, center_y, w_rect, h_rect, theta, cls_id = rect_p
    p_r_rect = ((center_x, center_y), (w_rect, h_rect), -theta)
    p_box = cv2.boxPoints(p_r_rect)
    p_box = np.int0(p_box)
    rr2, cc2 = polygon(p_box[:, 0], p_box[:,1], shape)

    mask_rr = rr2 < shape[1]
    rr2 = rr2[mask_rr]
    cc2 = cc2[mask_rr]

    mask_cc = cc2 < shape[0]
    cc2 = cc2[mask_cc]
    rr2 = rr2[mask_cc]

    area = np.zeros(shape)
    area[cc1, rr1] += 1
    area[cc2, rr2] += 1

    union = np.sum(area > 0)
    intersection = np.sum(area == 2)

    if union <= 0:
        return 0
    else:
        return intersection / union
    


def calculate_max_iou(rects_p, rects_gt):
    max_iou = 0
    for rect_gt in rects_gt:
        for rect_p in rects_p:
            iou = calculate_iou(rect_p, rect_gt)
            if iou > max_iou:
                max_iou = iou
    return max_iou



def calculate_grasp_iou_match(rects_p, rects_gt, threshold=0.25):
    """
    Calculate grasp success using the IoU (Jacquard) metric (e.g. in https://arxiv.org/abs/1301.3592)
    A success is counted if grasp rectangle has a 25% IoU with a ground truth, and is withing 30 degrees.
    :param grasp_q: Q outputs of network (Nx300x300x3)
    :param grasp_angle: Angle outputs of network
    :param ground_truth_bbs: Corresponding ground-truth BoundingBoxes
    :param no_grasps: Maximum number of grasps to consider per image.
    :param grasp_width: (optional) Width output from network
    :param threshold: Threshold for IOU matching. Detect with IOU ≥ threshold
    :return: success
    """
    obj_success_count = [0 for i in range(31)]
    obj_num_count = [0 for i in range(31)]
    

    for obj_rects_gts in rects_gt:
        id_gt = int(obj_rects_gts[0][-1])
        obj_num_count[id_gt-1] += 1

        for obj_rects_preds in rects_p:
            if len(obj_rects_preds) == 0:
                continue
            id_p = int(obj_rects_preds[0][-1])

            if id_gt != id_p:
                continue

            max_iou = calculate_max_iou(obj_rects_preds, obj_rects_gts)

            if max_iou > threshold and obj_success_count[id_gt-1] < obj_num_count[id_gt-1]:
                obj_success_count[id_gt-1] += 1
    
    return obj_num_count, obj_success_count



def calculate_grasp_iou_match_jacquard(rects_p, rects_gt, threshold=0.25):
    """
    Calculate grasp success using the IoU (Jacquard) metric (e.g. in https://arxiv.org/abs/1301.3592)
    A success is counted if grasp rectangle has a 25% IoU with a ground truth, and is withing 30 degrees.
    :param grasp_q: Q outputs of network (Nx300x300x3)
    :param grasp_angle: Angle outputs of network
    :param ground_truth_bbs: Corresponding ground-truth BoundingBoxes
    :param no_grasps: Maximum number of grasps to consider per image.
    :param grasp_width: (optional) Width output from network
    :param threshold: Threshold for IOU matching. Detect with IOU ≥ threshold
    :return: success
    """
    obj_success_count = 0
    obj_num_count = 1

    rects_p = [r for r in rects_p[0] if int(r[-1] == 1)]
    max_iou = calculate_max_iou(rects_p, rects_gt)
    if max_iou > threshold:
        obj_success_count = 1
    
    return obj_num_count, obj_success_count

            




def draw_lincomb(ids_p, proto_data, masks, img_name, target_dir=None):
    # print(proto_data.shape)
    # print(masks.shape)


    for kdx in range(masks.shape[0]):
        target_id = int(ids_p[kdx])
        # jdx = kdx + -1
        coeffs = masks[kdx, :].cpu().numpy()
        idx = np.argsort(-np.abs(coeffs))

        coeffs_sort = coeffs[idx]
        arr_h, arr_w = (4, 8)
        p_h, p_w, _ = proto_data.size()
        arr_img = np.zeros([p_h * arr_h, p_w * arr_w])
        arr_run = np.zeros([p_h * arr_h, p_w * arr_w])

        for y in range(arr_h):
            for x in range(arr_w):
                i = arr_w * y + x

                if i == 0:
                    running_total = proto_data[:, :, idx[i]].cpu().numpy() * coeffs_sort[i]
                else:
                    running_total += proto_data[:, :, idx[i]].cpu().numpy() * coeffs_sort[i]

                running_total_nonlin = (1 / (1 + np.exp(-running_total)))

                arr_img[y * p_h:(y + 1) * p_h, x * p_w:(x + 1) * p_w] = (proto_data[:, :, idx[i]] / torch.max(
                    proto_data[:, :, idx[i]])).cpu().numpy() * coeffs_sort[i]
                arr_run[y * p_h:(y + 1) * p_h, x * p_w:(x + 1) * p_w] = (running_total_nonlin > 0.5).astype(np.float)

        arr_img = ((arr_img + 1) * 127.5).astype('uint8')
        arr_img = cv2.applyColorMap(arr_img, cv2.COLORMAP_WINTER)
        if target_dir is None:
            cv2.imwrite(f'results/ocid/lincomb_{img_name}', arr_img)
        else:
            if not os.path.exists(f'{target_dir}/{target_id}'):
                os.makedirs(f'{target_dir}/{target_id}')

            cv2.imwrite(f'{target_dir}/{target_id}/lincomb_{img_name}', arr_img)


def draw_img(ids_p, class_p, box_p, mask_p, img_origin, cfg, img_name=None, fps=None):
    if ids_p is None:
        return img_origin

    if isinstance(ids_p, torch.Tensor):
        ids_p = ids_p.cpu().numpy()
        class_p = class_p.cpu().numpy()
        box_p = box_p.cpu().numpy()
        mask_p = mask_p.cpu().numpy()

    num_detected = ids_p.shape[0]

    img_fused = img_origin
    if not cfg.hide_mask:
        masks_semantic = mask_p * (ids_p[:, None, None] + 1)  # expand ids_p' shape for broadcasting
        # The color of the overlap area is different because of the '%' operation.
        masks_semantic = masks_semantic.astype('int').sum(axis=0) % (cfg.num_classes - 1)
        color_masks = COLORS[masks_semantic].astype('uint8')
        img_fused = cv2.addWeighted(color_masks, 0.4, img_origin, 0.6, gamma=0)

        if cfg.cutout:
            total_obj = (masks_semantic != 0)[:, :, None].repeat(3, 2)
            total_obj = total_obj * img_origin
            new_mask = ((masks_semantic == 0) * 255)[:, :, None].repeat(3, 2)
            img_matting = (total_obj + new_mask).astype('uint8')
            cv2.imwrite(f'results/images/{img_name}_total_obj.jpg', img_matting)

            for i in range(num_detected):
                one_obj = (mask_p[i])[:, :, None].repeat(3, 2)
                one_obj = one_obj * img_origin
                new_mask = ((mask_p[i] == 0) * 255)[:, :, None].repeat(3, 2)
                x1, y1, x2, y2 = box_p[i, :]
                img_matting = (one_obj + new_mask)[y1:y2, x1:x2, :]
                cv2.imwrite(f'results/images/{img_name}_{i}.jpg', img_matting)
    scale = 0.6
    thickness = 1
    font = cv2.FONT_HERSHEY_DUPLEX

    if not cfg.hide_bbox:
        for i in reversed(range(num_detected)):
            x1, y1, x2, y2 = box_p[i, :]

            color = COLORS[ids_p[i] + 1].tolist()
            cv2.rectangle(img_fused, (x1, y1), (x2, y2), color, thickness)

            class_name = cfg.class_names[ids_p[i]]
            text_str = f'{class_name}: {class_p[i]:.2f}' if not cfg.hide_score else class_name

            text_w, text_h = cv2.getTextSize(text_str, font, scale, thickness)[0]
            cv2.rectangle(img_fused, (x1, y1), (x1 + text_w, y1 + text_h + 5), color, -1)
            cv2.putText(img_fused, text_str, (x1, y1 + 15), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    if cfg.real_time:
        fps_str = f'fps: {fps:.2f}'
        text_w, text_h = cv2.getTextSize(fps_str, font, scale, thickness)[0]
        # Create a shadow to show the fps more clearly
        img_fused = img_fused.astype(np.float32)
        img_fused[0:text_h + 8, 0:text_w + 8] *= 0.6
        img_fused = img_fused.astype(np.uint8)
        cv2.putText(img_fused, fps_str, (0, text_h + 2), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return img_fused
