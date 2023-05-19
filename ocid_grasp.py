import os
import cv2
from cv2 import transform
import numpy as np
from functools import partial

from skimage.measure import regionprops
from skimage.draw import polygon
from scipy import ndimage as ndi

import torch
import torch.utils.data as data

from utils.gr_augmentation import gr_train_aug

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from utils.gr_augmentation import gr_train_aug, gr_val_aug


class OCIDGraspDataset(data.Dataset):
    def __init__(self, root_path, split_name, transform=None, mode="train"):
        super(OCIDGraspDataset, self).__init__()
        self.root_path = root_path
        self.split_name = split_name
        self.transform = transform
        self.mode = mode

        self._data = self._load_split()

        self.max_grasp_width = [65, 83, 45, 64, 43, 23, 140, 62, 29, 107, 147, 70, 34, 103, 112, 118, 101, 70, 41, 51, 80, 61, 77, 74, 57, 56, 74, 42, 54, 49, 75]

    
    def __len__(self):
        return len(self._data)

    
    def save_to_npz(self, index, bboxes, rects, labels, ins_masks, pos_masks, qua_masks, ang_masks, wid_masks):
        seq_path, img_f = self._data[index]
        if not os.path.exists(os.path.join(self.root_path, seq_path, "annos")):
            print("Making target directory: {}".format(os.path.join(self.root_path, seq_path, "annos")))
            os.makedirs(os.path.join(self.root_path, seq_path, "annos"))

        tgt_path = os.path.join(os.path.join(self.root_path, seq_path, "annos"), "{}.npz".format(img_f[:-4]))

        np.savez(
            tgt_path,
            bboxes=bboxes,
            rects=rects,
            labels=labels,
            ins_masks=ins_masks,
            pos_masks=pos_masks,
            qua_masks=qua_masks,
            ang_masks=ang_masks,
            wid_masks=wid_masks
        )
    

    def save_visualize_grasps(self, img, grasps, ins_masks, bboxes, labels, crop=None, target_dir=None):
        from OCID_class_dict import colors_list, cls_list

        masks_semantic = ins_masks * (labels[:, None, None]+1)
        masks_semantic = masks_semantic.astype('int').sum(axis=0) % 31

        colors_list = np.array(colors_list)
        color_masks = colors_list[np.array(masks_semantic)].astype('uint8')
        img_u8 = img.astype('uint8')
        img_fused = (color_masks * 0.2 + img_u8 * 0.9)

        for i in range(bboxes.shape[0]):
            name = cls_list[int(bboxes[i, -1])]
            color = colors_list[int(bboxes[i, -1])]
            cv2.rectangle(img_fused, (int(bboxes[i, 0]), int(bboxes[i, 1])),
                        (int(bboxes[i, 2]), int(bboxes[i, 3])), color.tolist(), 1)
            cv2.putText(img_fused, "{}:{}".format(name, int(bboxes[i, -1])), (int(bboxes[i, 0]), int(bboxes[i, 1])), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        for rect in grasps:
            cls_id = rect[-1]
            name = cls_list[int(cls_id)]
            color = colors_list[int(cls_id)]
            center_x, center_y, width, height, theta, cls_id = rect
            box = ((center_x, center_y), (width, height), -(theta+180))
            box = cv2.boxPoints(box)
            box = np.int0(box)
            cv2.drawContours(img_fused, [box], 0, color.tolist(), 2)
    
        if crop is not None:
            img_fused = img_fused[crop[2]:crop[3], crop[0]:crop[1], :]


        cv2.imwrite(os.path.join(target_dir, "grasps_gt.png"), img_fused)

    
    def save_grasp_masks(self, pos_masks, qua_masks, ang_masks, wid_masks, sin_masks, cos_masks, crop=None, target_dir=None):

        print("save grasp masks: ", pos_masks.shape)

        all_pos_mask = np.zeros((pos_masks.shape[1], pos_masks.shape[2]))
        all_qua_mask = np.zeros((pos_masks.shape[1], pos_masks.shape[2]))
        all_ang_mask = np.zeros((pos_masks.shape[1], pos_masks.shape[2]))
        all_wid_mask = np.zeros((pos_masks.shape[1], pos_masks.shape[2]))
        all_sin_mask = np.zeros((pos_masks.shape[1], pos_masks.shape[2]))
        all_cos_mask = np.zeros((pos_masks.shape[1], pos_masks.shape[2]))

        for pos_mask, qua_mask, ang_mask, wid_mask, sin_mask, cos_mask in zip(pos_masks, qua_masks, ang_masks, wid_masks, sin_masks, cos_masks):
            all_pos_mask += pos_mask
            all_qua_mask += qua_mask
            all_ang_mask += ang_mask
            all_wid_mask += wid_mask
            all_sin_mask += sin_mask
            all_cos_mask += cos_mask

        fig = plt.figure(figsize=(10, 10))

        ax = fig.add_subplot(2, 3, 1)
        plot = ax.imshow(all_pos_mask, cmap='jet', vmin=0, vmax=1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        ax.set_title('Grasp Position')
        ax.axis('off')
        plt.colorbar(plot, cax=cax)

        ax = fig.add_subplot(2, 3, 2)
        plot = ax.imshow(all_qua_mask, cmap='jet', vmin=0, vmax=1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        ax.set_title('Grasp Quality')
        ax.axis('off')
        plt.colorbar(plot, cax=cax)

        ax = fig.add_subplot(2, 3, 3)
        plot = ax.imshow(all_ang_mask, cmap='rainbow', vmin=-np.pi / 2, vmax=np.pi / 2)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        ax.set_title('Grasp Angle')
        ax.axis('off')
        plt.colorbar(plot, cax=cax)

        ax = fig.add_subplot(2, 3, 4)
        plot = ax.imshow(all_wid_mask, cmap='jet', vmin=0, vmax=1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        ax.set_title('Gripper Width')
        ax.axis('off')
        plt.colorbar(plot, cax=cax)


        ax = fig.add_subplot(2, 3, 4)
        plot = ax.imshow(all_wid_mask, cmap='jet', vmin=0, vmax=1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        ax.set_title('Gripper Width')
        ax.axis('off')
        plt.colorbar(plot, cax=cax)

        ax = fig.add_subplot(2, 3, 5)
        plot = ax.imshow(all_sin_mask, cmap='jet', vmin=-1, vmax=1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        ax.set_title('sin(angle)')
        ax.axis('off')
        plt.colorbar(plot, cax=cax)


        ax = fig.add_subplot(2, 3, 6)
        plot = ax.imshow(all_cos_mask, cmap='jet', vmin=0, vmax=1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        ax.set_title('cos(angle)')
        ax.axis('off')
        plt.colorbar(plot, cax=cax)


        plt.savefig(os.path.join(target_dir, "grasp_masks_gt.png"))
        
        plt.close()

    
    def show_data(self, img, depth, bboxes, labels, ins_masks=None, grasps=None, pos_masks=None, ang_masks=None, wid_masks=None, display=False, target_dir=None):
        from OCID_class_dict import colors_list, cls_list
        
        print(f'\nimg shape: {img.shape}')
        print('----------------boxes----------------')
        print(bboxes)
        print('----------------labels---------------')
        print([cls_list[int(i)] for i in labels], '\n')

        masks_semantic = ins_masks * (labels[:, None, None]+1)
        masks_semantic = masks_semantic.astype('int').sum(axis=0) % 31

        colors_list = np.array(colors_list)
        color_masks = colors_list[np.array(masks_semantic)].astype('uint8')
        img_u8 = img.astype('uint8')
        img_fused = (color_masks * 0.6 + img_u8 * 0.8)

        fig = plt.figure(figsize=(10, 10))

        for i in range(bboxes.shape[0]):
            name = cls_list[int(bboxes[i, -1])]
            color = colors_list[int(bboxes[i, -1])]
            cv2.rectangle(img_fused, (int(bboxes[i, 0]), int(bboxes[i, 1])),
                        (int(bboxes[i, 2]), int(bboxes[i, 3])), color.tolist(), 1)
            cv2.putText(img_fused, "{}:{}".format(name, int(bboxes[i, -1])), (int(bboxes[i, 0]), int(bboxes[i, 1])), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        if grasps is not None:
            for rect in grasps:
                cls_id = rect[-1]
                name = cls_list[int(cls_id)]
                color = colors_list[int(cls_id)].tolist()
                center_x, center_y, width, height, theta, cls_id = rect
                box = ((center_x, center_y), (width, height), -(theta))
                box = cv2.boxPoints(box)
                box = np.int0(box)
                # cv2.drawContours(img_fused, [box], 0, color, 2)

                inv_color = (255, 255-color[1], 255-color[2])

                p1, p2, p3, p4 = box
                length = width
                p5 = (p1+p2)/2
                p6 = (p3+p4)/2
                p7 = (p5+p6)/2

                rad = theta / 180 * np.pi
                p8 = (p7[0]-length*np.sin(rad), p7[1]+length*np.cos(rad))
                cv2.circle(img_fused, (int(p7[0]), int(p7[1])), 2, (0,0,255), 2)
                cv2.line(img_fused, (int(p1[0]),int(p1[1])), (int(p2[0]),int(p2[1])), (0,0,255), 3, 8)
                cv2.line(img_fused, (int(p3[0]),int(p3[1])), (int(p4[0]),int(p4[1])), (0,0,255), 3, 8)
                cv2.line(img_fused, (int(p5[0]),int(p5[1])), (int(p6[0]),int(p6[1])), (255,0,0), 2, 8)
                # cv2.line(img_fused, (int(p7[0]),int(p7[1])), (int(p8[0]),int(p8[1])), inv_color, 1, 8)

        # for obj_rects in grasps:
        #     for rect in obj_rects:
        #         cls_id = rect[-1]
        #         name = cls_list[int(cls_id)]
        #         center_x, center_y, width, height, theta, cls_id = rect
        #         box = ((center_x, center_y), (width, height), theta)
        #         box = cv2.boxPoints(box)
        #         box = np.int0(box)
        #         cv2.drawContours(img_fused, [box], 0, (255, 0, 0), 2)
                

        cv2.imwrite(os.path.join(target_dir, "result.png"), img_fused)


        ax = fig.add_subplot(2, 3, 1)
        ax.imshow((img_u8/255.)[...,::-1])
        ax.set_title('RGB')
        ax.axis('off')

        if depth is not None:
            ax = fig.add_subplot(2, 3, 2)
            ax.imshow(depth, cmap='gray')
            ax.set_title('Depth')
            ax.axis('off')

        ax = fig.add_subplot(2, 3, 3)
        ax.imshow(img_fused/255.)
        ax.set_title('Masks & Bboxes')
        ax.axis('off')


        if (pos_masks is not None) and (ang_masks is not None) and (wid_masks is not None):
            all_pos_mask = np.zeros((pos_masks.shape[1], pos_masks.shape[2]))
            all_ang_mask = np.zeros((pos_masks.shape[1], pos_masks.shape[2]))
            all_wid_mask = np.zeros((pos_masks.shape[1], pos_masks.shape[2]))

            # for pos_mask, ang_mask, wid_mask in zip(pos_masks, ang_masks, wid_masks):
            #     all_pos_mask += pos_mask
            #     all_ang_mask += ang_mask
            #     all_wid_mask += wid_mask
            all_pos_mask += pos_masks[-1]
            all_ang_mask += ang_masks[-1]
            all_wid_mask += wid_masks[-1]

            ax = fig.add_subplot(2, 3, 4)
            plot = ax.imshow(all_pos_mask, cmap='jet', vmin=0, vmax=1)
            ax.set_title('Quality')
            ax.axis('off')
            plt.colorbar(plot)

            ax = fig.add_subplot(2, 3, 5)
            plot = ax.imshow(all_ang_mask, cmap='rainbow', vmin=-np.pi / 2, vmax=np.pi / 2)
            ax.set_title('Angle')
            ax.axis('off')
            plt.colorbar(plot)

            ax = fig.add_subplot(2, 3, 6)
            plot = ax.imshow(all_wid_mask, cmap='jet', vmin=0, vmax=1)
            ax.set_title('Width')
            ax.axis('off')
            plt.colorbar(plot)

        if display:
            plt.show()
        elif target_dir is not None:
            print("Save visualizaitons")
            plt.savefig(os.path.join(target_dir, "overall_cogr_result.png"))
        else:
            print("Please specify the name of output file")
        
        plt.close()

    
    def _match_rects_and_objects(self, rects, bboxes, masks, labels, min_corners=2):
        object_rects = []
        object_bboxes = []
        object_masks = []
        object_labels = []

        for i in range(bboxes.shape[0]):
            box = bboxes[i]
            mask = masks[i]
            label = labels[i]
            tmp = []
            for rect in rects:
                center_x, center_y, w, h = rect[:4]
                rect_obj_id = rect[-1]

                # Grasp rect and bbox should have the same cls_id
                if int(rect_obj_id) == int(box[4]):
                    # Center of grasp rect in bbox
                    if mask[int(center_y), int(center_x)]:
                        tmp.append(rect)
                    # else:
                    #     # at least 1 corner of grasp rect in bbox
                    #     count = 0
                    #     for corner in rect_box:
                    #         if (corner[0] >= box[0] and corner[0] <= box[2]) and \
                    #             (corner[1] >= box[1] and corner[1] <= box[3]):
                                
                    #             count += 1
                        
                    #     if count > min_corners:
                    #         tmp.append(rect)
            if len(tmp) > 0:
                object_rects.append(tmp)
                object_bboxes.append(box)
                object_masks.append(mask)
                object_labels.append(label)
        
        return object_rects, np.array(object_bboxes), np.array(object_masks), np.array(object_labels)

    
    def _draw_grasp_rects(self, rects, width, height):
        pos_masks = []
        qua_masks = []
        ang_masks = []
        wid_masks = []

        for obj_rects in rects:
            pos_out = np.zeros((height, width))
            qua_out = np.zeros((height, width))
            ang_out = np.zeros((height, width))
            wid_out = np.zeros((height, width))
            for rect in obj_rects:
                center_x, center_y, w_rect, h_rect, theta, cls_id = rect
                width_factor = float(self.max_grasp_width[int(cls_id)-1])

                # Get 4 corners of rotated rect
                # Convert from our angle represent to opencv's
                r_rect = ((center_x, center_y), (w_rect/2, h_rect), -(theta+180))
                box = cv2.boxPoints(r_rect)
                box = np.int0(box)

                rr, cc = polygon(box[:, 0], box[:,1])

                mask_rr = rr < width
                rr = rr[mask_rr]
                cc = cc[mask_rr]

                mask_cc = cc < height
                cc = cc[mask_cc]
                rr = rr[mask_cc]


                pos_out[cc, rr] = 1.0
                qua_out[cc, rr] += 1.0
                ang_out[cc, rr] = theta * np.pi / 180
                # Adopt width normalize accoding to class 
                wid_out[cc, rr] = np.clip(w_rect, 0.0, width_factor) / width_factor

            # Preprocessing quality mask
            qua_out = 1 / (1 + np.exp(-qua_out))
            qua_out = qua_out * pos_out
            smooth_factor = 1e-7

            qua_out = np.clip(qua_out, smooth_factor, 1-smooth_factor)


            pos_masks.append(pos_out)
            qua_masks.append(qua_out)
            ang_masks.append(ang_out)
            wid_masks.append(wid_out)
        
        return np.array(pos_masks), np.array(qua_masks), np.array(ang_masks), np.array(wid_masks) 



    def _load_split(self):
        with open(os.path.join(self.root_path, "data_split", self.split_name + ".txt"), "r") as fid:
            images = [x.strip().split(',') for x in fid.readlines()]

        return images


    def _load_rgb(self, index):
        seq_path, img_f = self._data[index]
        img = cv2.imread(os.path.join(self.root_path, seq_path, "rgb", img_f), cv2.COLOR_BGR2RGB)

        return img

    
    def _load_depth(self, index, factor=1000.):
        seq_path, img_f = self._data[index]
        depth = cv2.imread(os.path.join(self.root_path, seq_path, "depth", img_f), cv2.IMREAD_UNCHANGED) / factor

        depth = 1 - (depth / np.max(depth))

        return np.expand_dims(depth, -1)

    def _load_annos(self, index):
        seq_path, img_f = self._data[index]
        annos = np.load(os.path.join(os.path.join(self.root_path, seq_path, "annos"), "{}.npz".format(img_f[:-4])))

        return annos["bboxes"], annos["rects"], annos["labels"], annos["ins_masks"], annos["pos_masks"], annos["qua_masks"], annos["ang_masks"], annos["wid_masks"]


    
    def _load_sem_masks(self, index):

        seq_path, img_f = self._data[index]
        sem_mask = cv2.imread(os.path.join(self.root_path, seq_path, "seg_mask_labeled_combi", img_f), cv2.IMREAD_UNCHANGED)
        ins_mask = cv2.imread(os.path.join(self.root_path, seq_path, "seg_mask_instances_combi", img_f), cv2.IMREAD_UNCHANGED)

        labels     = []
        bboxes     = []
        ins_masks  = []

        props = regionprops(sem_mask)
        for prop in props:
            cls_id = prop.label
            
            # Get binary mask for each semantic class
            bin_mask = (sem_mask == cls_id).astype('int8')
            # Get corresponding instance mask
            cls_ins_mask = (ins_mask * bin_mask)

            # Get regions for each instance
            ins_props = regionprops(cls_ins_mask)
            for ins in ins_props:
                labels.append(cls_id)
                bboxes.append([ins.bbox[1], ins.bbox[0], ins.bbox[3], ins.bbox[2], cls_id])
                mask = (cls_ins_mask == ins.label).astype('int8').astype('float32')
                ins_masks.append(mask)
        
        bboxes = np.array(bboxes).astype('float32')
        labels = np.array(labels)
        ins_masks  = np.array(ins_masks)

        return bboxes, sem_mask, ins_masks, labels
    

    def _load_per_class_grasps(self, index):
        seq_path, img_f = self._data[index]
        img_n = img_f[:-4]
        anno_path = os.path.join(self.root_path, seq_path, "Annotations_per_class", img_n)


        grasps_list = []
        for cls_id in os.listdir(anno_path):
            grasp_path = os.path.join(anno_path, cls_id, img_n+".txt")
            with open(grasp_path, 'r') as f:
                points_list = []
                for count, line in enumerate(f):
                    line = line.rstrip()
                    [x, y] = line.split(' ')

                    x = float(x)
                    y = float(y)

                    pt = (x, y)
                    points_list.append(pt)

                    if len(points_list) == 4:
                        p1, p2, p3, p4 = points_list
                        center_x = (p1[0] + p3[0]) / 2
                        center_y = (p1[1] + p3[1]) / 2
                        width  = np.sqrt((p1[0] - p4[0]) * (p1[0] - p4[0]) + (p1[1] - p4[1]) * (p1[1] - p4[1]))
                        height = np.sqrt((p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1]))
                        
                        # @NOTE
                        # Along x+ is 0 degree, increase by rotating anti-clockwise
                        # If you want to use opencv boxPoints & drawContours to visualize grasps
                        # Remember to take -theta as param :-)
                        theta = np.arctan2(p4[0] - p1[0], p4[1] - p1[1]) * 180 / np.pi
                        if theta > 0:
                            theta = theta-90
                        else:
                            theta = theta+90


                        grasps_list.append([center_x, center_y, width, height, theta, int(cls_id)])
                        points_list = []
            
        return grasps_list
    


    def __getitem__(self, index):
        rgb = self._load_rgb(index)
        height, width, _ = rgb.shape
        depth = self._load_depth(index)
        #  bboxes, rects, labels, ins_masks, pos_masks, qua_masks, ang_masks, wid_masks = self._load_annos(index)


        bboxes, sem_mask, ins_masks, labels = self._load_sem_masks(index)
        rects = self._load_per_class_grasps(index)

        # self.save_visualize_grasps(rgb, rects,ins_masks, bboxes, labels, target_dir="results/grasps/{:04d}".format(index))

        # @NOTE
        # Uncomment to see all the data & annos
        # self.show_data(rgb, depth, np.array(bboxes), np.array(labels), np.array(ins_masks), rects)

        ins_rects, bboxes, ins_masks, labels = self._match_rects_and_objects(rects, bboxes, ins_masks, labels)

            
        # @NOTE
        # Uncomment to see the matching results
        # self.show_data(rgb, depth, np.array([bboxes[0]]), np.array(labels), np.array([ins_masks[0]]), ins_rects[0])


        # self.save_visualize_grasps(rgb, ins_rects[-1][:3],np.array([ins_masks[-1]]), np.array([bboxes[-1]]), np.array([labels[-1]]), crop=crop, target_dir="results/grasps/{:04d}".format(index))

        pos_masks, qua_masks, ang_masks, wid_masks = self._draw_grasp_rects(ins_rects, width, height)

        
        # self.save_grasp_masks(pos_masks, qua_masks, ang_masks, wid_masks, sin_masks, cos_masks, target_dir="results/grasps/{:04d}".format(index))

        # self.save_to_npz(index, bboxes, rects, labels, ins_masks, pos_masks, qua_masks, ang_masks, wid_masks)
       
        # # @NOTE
        # # Uncomment to see the grasping heat map
        # self.show_data(rgb, depth, np.array(bboxes), np.array(labels), np.array(ins_masks), rects, pos_masks,  ang_masks, wid_masks, tgt_file="data_with_annos_1.png")

        if self.mode == "train":
            img, depth, ins_masks, pos_masks, qua_masks, ang_masks, wid_masks, bboxes, labels = self.transform(
                rgb, depth, ins_masks, pos_masks, qua_masks, ang_masks, wid_masks, bboxes[:, :4], labels
            )

            if len(depth.shape) == 2:
                depth = np.expand_dims(depth, -1)
            # self.show_data(img, depth, np.array([bboxes[2]]), np.array([labels[2]]), ins_masks=np.array([ins_masks[2]]), pos_masks=np.array([pos_masks[2]]), ang_masks=np.array([ang_masks[2]]), wid_masks=np.array([wid_masks[2]]))

            rgbd = np.concatenate([img, depth], axis=-1).transpose((2,0,1))
            bboxes = np.concatenate([bboxes, labels.reshape(-1,1)], axis=-1)

            # Test using 0 - pi / pi
            # ang_sin_masks = np.sin(2 * ang_masks)
            # ang_cos_masks = 1 - (np.cos(2 * ang_masks) + 1.) / 2.

            sin_masks = np.sin(2 * ang_masks)
            cos_masks = np.cos(2 * ang_masks)


            return rgbd, bboxes, ins_masks, pos_masks, qua_masks, sin_masks, cos_masks, wid_masks

        elif self.mode == "test":
            img, depth, ins_masks, pos_masks, qua_masks, ang_masks, wid_masks, bboxes, labels = self.transform(
                rgb, depth, ins_masks, pos_masks, qua_masks, ang_masks, wid_masks, bboxes[:, :4], labels
            )
            

            if len(depth.shape) == 2:
                depth = np.expand_dims(depth, -1)
            # self.show_data(img, depth, np.array([bboxes[2]]), np.array([labels[2]]), ins_masks=np.array([ins_masks[2]]), pos_masks=np.array([pos_masks[2]]), ang_masks=np.array([ang_masks[2]]), wid_masks=np.array([wid_masks[2]]))

            rgbd = np.concatenate([img, depth], axis=-1).transpose((2,0,1))
            bboxes = np.concatenate([bboxes, labels.reshape(-1,1)], axis=-1)

            # Test using 0 - pi / pi
            # ang_sin_masks = np.sin(2 * ang_masks)
            # ang_cos_masks = 1 - (np.cos(2 * ang_masks) + 1.) / 2.

            # Normalize angle mask to [-1,1]
            sin_masks = np.sin(2 * ang_masks)
            cos_masks = np.cos(2 * ang_masks)

            # ang_sin_masks: [0, 1]
            # ang_cos_masks: [-1,1]
            # @NOTE
            # Should we normalize ang_cos_masks to [0, 1]?

            # self.show_data(img, depth, np.array([bboxes[2]]), np.array([labels[2]]), ins_masks=np.array([ins_masks[2]]), pos_masks=np.array([pos_masks[2]]), ang_masks=np.array([ang_sin_masks[2]]), wid_masks=np.array([wid_masks[2]]), tgt_file="data_with_annos-1.py")

            return rgbd, bboxes, ins_rects, ins_masks, pos_masks, qua_masks, ang_masks, wid_masks
        



if __name__ == "__main__":
    from tqdm import tqdm

    train_dataset = OCIDGraspDataset(
        "/home/puzek/sdb/dataset/OCID_grasp",
        "training_0",
        mode="train",
        transform=partial(gr_val_aug, 544)
    )
    print(len(train_dataset))
    train_dataset[8]

    
