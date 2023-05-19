from utils.output_utils import (
    gr_nms_v2, draw_lincomb, gr_post_processing, 
    gr_post_processing_jacquard, calculate_grasp_iou_match, 
    calculate_grasp_iou_match_jacquard, calculate_iou
)
from utils.box_utils import crop
from skimage.feature import peak_local_max
from skimage.filters import gaussian
from tqdm import tqdm
import torch
import numpy as np

num_grasp = [1]


def evaluate(net, dataset, cfg, rgb_input=False):
    net.eval()
    with torch.no_grad():
        for num in num_grasp:

            total_obj_num_count = np.array([0 for i in range(31)])
            total_obj_success_count = np.array([0 for i in range(31)])

            pbar = tqdm(range(len(dataset)))

            for i in pbar:
                rgbd, bboxes, rects, ins_masks, pos_masks, qua_masks, ang_masks, wid_masks = dataset[i]
                rgbd_tensor = torch.tensor(rgbd).unsqueeze(0).cuda().detach()
                img = rgbd.transpose((1,2,0))[:, :, :3]
                depth = rgbd.transpose((1,2,0))[:, :, 3]

                if rgb_input:
                    class_pred, box_pred, coef_pred, gr_pos_coef_pred, gr_sin_coef_pred, gr_cos_coef_pred, gr_wid_coef_pred, proto_out = net(rgbd_tensor[:,:3,:,:])
                else:
                    class_pred, box_pred, coef_pred, gr_pos_coef_pred, gr_sin_coef_pred, gr_cos_coef_pred, gr_wid_coef_pred, proto_out = net(rgbd_tensor)

                ids_p, class_p, box_p, coef_p, pos_coef_p, sin_coef_p, cos_coef_p, wid_coef_p, proto_p = gr_nms_v2(
                    class_pred, box_pred, coef_pred, proto_out,
                    gr_pos_coef_pred, gr_sin_coef_pred, gr_cos_coef_pred, gr_wid_coef_pred,
                    net.anchors, cfg
                )
                
                img, depth, box_p, instance_masks, grasps, pos_masks, ang_masks, wid_masks, ids_p = gr_post_processing(
                    img, depth, ids_p, class_p, box_p, coef_p, pos_coef_p, sin_coef_p, cos_coef_p, wid_coef_p, proto_p, ori_h=480, ori_w=640, visualize_lincomb=False, visualize_results=False,
                    num_grasp_per_object=num
                )

                obj_num_count, obj_success_count = calculate_grasp_iou_match(grasps, rects)
                total_obj_num_count += np.array(obj_num_count)
                total_obj_success_count += np.array(obj_success_count)
            
            class_rate = np.array(total_obj_success_count) / np.array(total_obj_num_count)
            overrall_rate = np.array(total_obj_success_count).sum() / np.array(total_obj_num_count).sum()

            print("Evaluate results with {} grasp predictions per object: ".format(num))
            print("Number of attempts per class: ")
            print(total_obj_num_count)
            print("Number of success per object: ")
            print(total_obj_success_count)
            print("Class-wise grasping successful rate: ")
            print(class_rate)
            print("Overrall successful rate: ", overrall_rate)

        return class_rate, overrall_rate


def evaluate_jacquard(net, dataset, cfg):
    net.eval()
    with torch.no_grad():
        total_obj_num_count = 0
        total_obj_success_count = 0
        
        pbar = tqdm(range(len(dataset)))
        for i in pbar:
            f, rgbd, bboxes, rects, ins_masks, pos_masks, qua_masks, ang_masks, wid_masks = dataset[i]
            rgbd_tensor = torch.tensor(rgbd).unsqueeze(0).cuda()
            img = rgbd.transpose((1,2,0))[:, :, :3]
            depth = rgbd.transpose((1,2,0))[:, :, 3]

            class_pred, box_pred, coef_pred, gr_pos_coef_pred, gr_sin_coef_pred, gr_cos_coef_pred, gr_wid_coef_pred, proto_out = net(rgbd_tensor)

            ids_p, class_p, box_p, coef_p, pos_coef_p, sin_coef_p, cos_coef_p, wid_coef_p, proto_p = gr_nms_v2(
                class_pred, box_pred, coef_pred, proto_out,
                gr_pos_coef_pred, gr_sin_coef_pred, gr_cos_coef_pred, gr_wid_coef_pred,
                net.anchors, cfg
            )

            if ids_p is None:
                total_obj_num_count += 1
                total_obj_success_count += 1
                continue
            
            img, depth, box_p, instance_masks, grasps, pos_masks, ang_masks, wid_masks, ids_p = gr_post_processing_jacquard(
                img, depth, ids_p, class_p, box_p, coef_p, pos_coef_p, sin_coef_p, cos_coef_p, wid_coef_p, proto_p, ori_h=1024, ori_w=1024, visualize_lincomb=False, visualize_results=False,
                num_grasp_per_object=1
            )

            tmp = []
            for obj_rects in grasps:
                tmp.extend(obj_rects)
            
            if len(tmp) == 0:
                total_obj_num_count += 1
                continue

            
            pred_rect = tmp[0]
            max_iou = 0
            scale = 1024 / 544
            for rect in rects:
                rect_gt = [rect[0]*scale, rect[1]*scale, rect[2]*scale, rect[3]*scale, rect[4], 1]
                
                iou = calculate_iou(pred_rect, rect_gt, shape=(1024, 1024), angle_threshold=30)

                max_iou = max(iou, max_iou)
            
            if max_iou > 0.25:
                total_obj_num_count += 1
                total_obj_success_count += 1
            else:
                total_obj_num_count += 1
                total_obj_success_count += 0
        
        rate = total_obj_success_count / total_obj_num_count

        print("Evaluate results with 1 grasp predictions per object: ")
        print("Number of attempts: ")
        print(total_obj_num_count)
        print("Number of success: ")
        print(total_obj_success_count)
        print("Overrall successful rate: ", rate)
        print("================================================")

    return rate