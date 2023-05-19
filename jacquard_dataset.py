import os
import cv2
import glob
import random
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

from utils.gr_augmentation import gr_train_aug_jacquard, gr_val_aug_jacquard

random.seed(1026)


class JacquardGraspDataset(data.Dataset):
    def __init__(self, root_path, origin_size=1024, target_size=544, transform=None, multi_obj_aug=False, mode="train"):
        super(JacquardGraspDataset, self).__init__()
        self.root_path = root_path
        self.origin_size = origin_size
        self.target_size = target_size
        self.transform = transform
        self.mode = mode
        self.multi_obj_aug = multi_obj_aug
        self.bg_mean = np.array([180, 180, 180])

        self.depth_mean = 1.5
        self.depth_std = 0.041

        self.sample_range = [[np.array([86, 86]), 100], [np.array([358, 86]), 100], [np.array([86, 358]), 100], [np.array([358, 358]), 100]]

    
        print("Loading data...")
        if self.multi_obj_aug:
            print("Enable multi-object data augment!")
        self.grasps = self._load_split()
        self.images = [f.replace("grasps.txt", "RGB.png") for f in self.grasps]
        self.depths = [f.replace("grasps.txt", "perfect_depth.tiff") for f in self.grasps]
        self.masks  = [f.replace("grasps.txt", "mask.png") for f in self.grasps]
        print("Done!")



    
    def __len__(self):
        assert len(self.grasps) == len(self.images) == len(self.depths) == len(self.masks)

        return len(self.grasps)


    def _load_split(self):
        if self.mode == "train":
            with open(os.path.join(self.root_path, "train_set.txt"), 'r') as f:
                grasps = f.readlines()
        elif self.mode == "test":
            with open(os.path.join(self.root_path, "val_set.txt"), 'r') as f:
                grasps = f.readlines()
        
        return grasps
    

    def _show_results(self, img, depth, bboxes, labels, mask, grasps, pos_masks, ang_masks, wid_masks, tgt_file=None):
        print(labels)
        fig = plt.figure(figsize=(20, 20))
        
        img_grasp = img.astype('uint8')
        img_bbox = img.astype('uint8')

        for bbox in bboxes:
            cv2.rectangle(img_bbox, (int(bbox[0]), int(bbox[1])),
                          (int(bbox[2]), int(bbox[3])), (255,0,0), 1)
        
        if grasps is not None:
            for grasp in grasps:
                center_x, center_y, width, height, theta = grasp[:5]
                box = ((center_x, center_y), (width, height), -theta)
                box = cv2.boxPoints(box)
                box = np.int0(box)
                cv2.drawContours(img_grasp, [box], 0, (255,0,0), 2)
        
        ax = fig.add_subplot(2, 5, 1)
        ax.imshow((img.astype('uint8')/255.)[...,::-1])
        ax.set_title('RGB')
        ax.axis('off')

        ax = fig.add_subplot(2, 5, 2)
        ax.imshow((img_grasp/255.)[...,::-1])
        ax.set_title('Grasp rectangles')
        ax.axis('off')

        ax = fig.add_subplot(2, 5, 3)
        ax.imshow((img_bbox/255.)[...,::-1])
        ax.set_title('bounding boxes')
        ax.axis('off')

        ax = fig.add_subplot(2, 5, 4)
        ax.imshow(depth, cmap='gray')
        ax.set_title('Depth')
        ax.axis('off')

        ax = fig.add_subplot(2, 5, 5)
        ax.imshow(mask[0])
        ax.set_title('Mask')
        ax.axis('off')

        all_pos_mask = np.zeros((pos_masks.shape[1], pos_masks.shape[2]))
        all_ang_mask = np.zeros((pos_masks.shape[1], pos_masks.shape[2]))
        all_wid_mask = np.zeros((pos_masks.shape[1], pos_masks.shape[2]))


        for (pos_mask, ang_mask, wid_mask) in zip(pos_masks, ang_masks, wid_masks):
            all_pos_mask += pos_mask
            all_ang_mask += ang_mask
            all_wid_mask += wid_mask
        
        ax = fig.add_subplot(2, 5, 6)
        plot = ax.imshow(all_pos_mask, cmap='jet', vmin=0, vmax=1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        ax.set_title('Grasp Position')
        ax.axis('off')
        plt.colorbar(plot, cax=cax)

        ax = fig.add_subplot(2, 5, 9)
        plot = ax.imshow(all_ang_mask, cmap='jet', vmin=0, vmax=1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        ax.set_title('Grasp Angle')
        ax.axis('off')
        plt.colorbar(plot, cax=cax)


        ax = fig.add_subplot(2, 5, 10)
        plot = ax.imshow(all_wid_mask, cmap='jet', vmin=0, vmax=1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        ax.set_title('Gripper Width')
        ax.axis('off')
        plt.colorbar(plot, cax=cax)

        if tgt_file is not None:
            plt.savefig(tgt_file)
            plt.close()
        else:
            plt.savefig("jacquard_vis.png")
            plt.close()


    
    def _show_data(self, img, depth, masks, bboxes, grasps=None, pos_masks=None, qua_masks=None, ang_masks=None, wid_masks=None, tgt_file=None):

        img_grasp = img.astype('uint8')
        img_bbox = img.astype('uint8')

        fig = plt.figure(figsize=(20, 20))
        
        for bbox in bboxes:
            cv2.rectangle(img_bbox, (int(bbox[0]), int(bbox[1])),
                          (int(bbox[2]), int(bbox[3])), (255,0,0), 1)
            break
            
        # cv2.imwrite("test_jacquard.png", img_u8)

        # cv2.imwrite("test_jacquard_mask.png", mask*255)

        if grasps is not None:
            for grasp in grasps:
                center_x, center_y, width, height, theta = grasp[:5]
                box = ((center_x, center_y), (width, height), -theta)
                box = cv2.boxPoints(box)
                box = np.int0(box)
                cv2.drawContours(img_grasp, [box], 0, (255,0,0), 2)
        
        # cv2.imwrite("test_jacquard_gr.png", img_u8)

        ax = fig.add_subplot(2, 5, 1)
        ax.imshow((img.astype('uint8')/255.)[...,::-1])
        ax.set_title('RGB')
        ax.axis('off')

        ax = fig.add_subplot(2, 5, 2)
        ax.imshow((img_grasp/255.)[...,::-1])
        ax.set_title('Grasp rectangles')
        ax.axis('off')

        ax = fig.add_subplot(2, 5, 3)
        ax.imshow((img_bbox/255.)[...,::-1])
        ax.set_title('bounding boxes')
        ax.axis('off')

        ax = fig.add_subplot(2, 5, 4)
        ax.imshow(depth, cmap='gray')
        ax.set_title('Depth')
        ax.axis('off')

        all_mask = np.zeros((pos_masks.shape[1], pos_masks.shape[2]))
        for mask in masks:
            all_mask += mask
            break

        ax = fig.add_subplot(2, 5, 5)
        ax.imshow(all_mask)
        ax.set_title('Mask')
        ax.axis('off')


        all_pos_mask = np.zeros((pos_masks.shape[1], pos_masks.shape[2]))
        all_qua_mask = np.zeros((pos_masks.shape[1], pos_masks.shape[2]))
        all_wid_mask = np.zeros((pos_masks.shape[1], pos_masks.shape[2]))
        all_ang_mask = np.zeros((pos_masks.shape[1], pos_masks.shape[2]))

        

        for (pos_mask, qua_mask, ang_mask, wid_mask) in zip(pos_masks, qua_masks, ang_masks, wid_masks):
            all_pos_mask += pos_mask
            all_qua_mask += qua_mask
            all_ang_mask += ang_mask
            all_wid_mask += wid_mask
            break

        all_sin_mask = np.sin(all_ang_mask * 2)
        all_cos_mask = np.cos(all_ang_mask * 2)

        ax = fig.add_subplot(2, 5, 6)
        plot = ax.imshow(all_pos_mask, cmap='jet', vmin=0, vmax=1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        ax.set_title('Grasp Position')
        ax.axis('off')
        plt.colorbar(plot, cax=cax)

        ax = fig.add_subplot(2, 5, 7)
        plot = ax.imshow(all_qua_mask, cmap='jet', vmin=0, vmax=1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        ax.set_title('Grasp Quality')
        ax.axis('off')
        plt.colorbar(plot, cax=cax)

        ax = fig.add_subplot(2, 5, 8)
        plot = ax.imshow(all_sin_mask, cmap='jet', vmin=-1, vmax=1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        ax.set_title('sin(angle)')
        ax.axis('off')
        plt.colorbar(plot, cax=cax)

        ax = fig.add_subplot(2, 5, 9)
        plot = ax.imshow(all_cos_mask, cmap='jet', vmin=0, vmax=1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        ax.set_title('cos(angle)')
        ax.axis('off')
        plt.colorbar(plot, cax=cax)


        ax = fig.add_subplot(2, 5, 10)
        plot = ax.imshow(all_wid_mask, cmap='jet', vmin=0, vmax=1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        ax.set_title('Gripper Width')
        ax.axis('off')
        plt.colorbar(plot, cax=cax)


        if tgt_file is not None:
            plt.savefig(tgt_file)
            plt.close()
        else:
            plt.savefig("jacquard_vis.png")
            plt.close()


    def _load_image(self, index):
        file_name = self.images[index].strip("\n")
        f = os.path.join(self.root_path, self.images[index].strip("\n"))
        img = cv2.imread(f, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.target_size, self.target_size))
        return file_name, img
    

    def _load_depth(self, index, factor=1., normalize=False):
        f = os.path.join(self.root_path, self.depths[index].strip("\n"))
        depth = cv2.imread(f, cv2.IMREAD_UNCHANGED) / factor
        depth = cv2.resize(depth, (self.target_size, self.target_size))

        if normalize:
            depth = 1 - (depth / np.max(depth))

        return depth
    

    def _load_mask(self, index):
        f = os.path.join(self.root_path, self.masks[index].strip("\n"))
        mask = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        mask = cv2.resize(mask, (self.target_size, self.target_size))
        mask = (mask > 0.4).astype("uint8")

        props = regionprops(mask)
        
        cls_id = props[0].label
        bin_mask = (mask == cls_id).astype("uint8")
        bbox = props[0].bbox
        
        return np.array([bin_mask]), np.array([[bbox[1], bbox[0], bbox[3], bbox[2]]]).astype("float"), np.array([1])

    
    def _load_grasps(self, index):
        scale = float(self.target_size/self.origin_size)

        fname = os.path.join(self.root_path, self.grasps[index].strip("\n"))
        grs = []
        with open(fname) as f:
            for l in f:
                x, y, theta, w, h = [float(v) for v in l[:-1].split(';')]
                # index based on row, column (y,x), and the Jacquard dataset's angles are flipped around an axis.
                grs.append([x*scale, y*scale, w*scale, h*scale, -theta, 1])
        
        return np.asarray(grs)


    def _draw_grasp_rects(self, rects, width, height, bboxes):
        pos_masks = []
        qua_masks = []
        ang_masks = []
        wid_masks = []

        factor = min(bboxes[3]-bboxes[1], bboxes[2]-bboxes[0])

        for obj_rects in rects:
            pos_out = np.zeros((height, width))
            qua_out = np.zeros((height, width))
            ang_out = np.zeros((height, width))
            wid_out = np.zeros((height, width))
            for rect in obj_rects:
                center_x, center_y, w_rect, h_rect, theta = rect[:5]
                # Get 4 corners of rotated rect
                # Convert from our angle represent to opencv's
                r_rect = ((center_x, center_y), (w_rect/3, h_rect), -theta)
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
                wid_out[cc, rr] = np.clip(w_rect, 0.0, factor) / factor

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


    # def _rotateAndScale(self, img, depth, msk, all_boxes):

    #     (oldY, oldX, chan) = img.shape  # note: numpy uses (y,x) convention but most OpenCV functions use (x,y)

    #     theta = float(np.random.randint(360) - 1)
    #     dx = np.random.randint(101) - 51
    #     dy = np.random.randint(101) - 51

    #     M = cv2.getRotationMatrix2D(center=(oldX / 2, oldY / 2), angle=theta,
    #                                 scale=1.0)  # rotate about center of image.

    #     # choose a new image size.
    #     newX, newY = oldX, oldY
    #     # include this if you want to prevent corners being cut off
    #     r = np.deg2rad(theta)
    #     newX, newY = (abs(np.sin(r) * newY) + abs(np.cos(r) * newX), abs(np.sin(r) * newX) + abs(np.cos(r) * newY))

    #     # Find the translation that moves the result to the center of that region.
    #     (tx, ty) = ((newX - oldX) / 2, (newY - oldY) / 2)
    #     M[0, 2] += tx
    #     M[1, 2] += ty

    #     imgRotate = cv2.warpAffine(img, M, dsize=(int(newX), int(newY)))
    #     mskRotate = cv2.warpAffine(msk, M, dsize=(int(newX), int(newY)))
    #     depRotate = cv2.warpAffine(depth, M, dsize=(int(newX), int(newY)))

    #     imgRotateCrop = imgRotate[
    #                     int(imgRotate.shape[0] / 2 - (im_size[0] / 2)) - dx:int(
    #                         imgRotate.shape[0] / 2 + (im_size[0] / 2)) - dx,
    #                     int(imgRotate.shape[1] / 2 - (im_size[1] / 2)) - dy:int(
    #                         imgRotate.shape[1] / 2 + (im_size[1] / 2)) - dy, :]
    #     mskRotateCrop = mskRotate[
    #                     int(mskRotate.shape[0] / 2 - (im_size[0] / 2)) - dx:int(
    #                         mskRotate.shape[0] / 2 + (im_size[0] / 2)) - dx,
    #                     int(mskRotate.shape[1] / 2 - (im_size[1] / 2)) - dy:int(
    #                         mskRotate.shape[1] / 2 + (im_size[1] / 2)) - dy]
    #     depRotateCrop = depRotate[
    #                     int(mskRotate.shape[0] / 2 - (im_size[0] / 2)) - dx:int(
    #                         mskRotate.shape[0] / 2 + (im_size[0] / 2)) - dx,
    #                     int(mskRotate.shape[1] / 2 - (im_size[1] / 2)) - dy:int(
    #                         mskRotate.shape[1] / 2 + (im_size[1] / 2)) - dy]

    #     bbsInShift = np.zeros_like(all_boxes)
    #     bbsInShift[:, 0, :] = all_boxes[:, 0, :] - (im_size[1] / 2)
    #     bbsInShift[:, 1, :] = all_boxes[:, 1, :] - (im_size[0] / 2)
    #     R = np.array([[np.cos(theta / 180 * np.pi), -np.sin(theta / 180 * np.pi)],
    #                   [np.sin(theta / 180 * np.pi), np.cos(theta / 180 * np.pi)]])
    #     R_all = np.expand_dims(R, axis=0)  #
    #     R_all = np.repeat(R_all, all_boxes.shape[0], axis=0)
    #     bbsInShift = np.swapaxes(bbsInShift, 1, 2)

    #     bbsRotated = np.dot(bbsInShift, R_all.T)
    #     bbsRotated = bbsRotated[:, :, :, 0]
    #     bbsRotated = np.swapaxes(bbsRotated, 1, 2)
    #     bbsInShiftBack = np.asarray(bbsRotated)
    #     bbsInShiftBack[:, 0, :] = (bbsRotated[:, 0, :] + (im_size[1] / 2) + dy)
    #     bbsInShiftBack[:, 1, :] = (bbsRotated[:, 1, :] + (im_size[0] / 2) + dx)

    #     return imgRotateCrop, mskRotateCrop, depRotateCrop, bbsInShiftBack


    def _filter_grasp_rects(self, grasps, mask):
        # @TODO
        # Filter our grasp rects which center falls out of mask of object.
        return 0           


    def _multi_obj_aug(self, index, num_obj, image_size=544):
        random.shuffle(self.sample_range)
        if num_obj == 1:
            f, rgb               = self._load_image(index)
            depth                = self._load_depth(index, normalize=True)
            mask, bboxes, labels = self._load_mask(index)
            # grasp angle in degree
            grasps               = self._load_grasps(index)

            pos_masks, qua_masks, ang_masks, wid_masks = self._draw_grasp_rects([grasps], self.target_size, self.target_size, bboxes[0])

            return rgb, depth, mask, bboxes, labels, pos_masks, qua_masks, ang_masks, wid_masks
        else:
            idx_list = random.sample(range(len(self.grasps)), num_obj-1)
            idx_list.append(index)

            all_image = np.zeros((image_size,image_size,3))
            all_image[:, :, ] = self.bg_mean
            random_noise = 3 * np.random.normal(size=(image_size,image_size,3))
            # all_image = all_image + random_noise
            all_depth = np.zeros((image_size,image_size))
            all_depth[:,:] = self.depth_mean

            all_pos_masks = []
            all_qua_masks = []
            all_ang_masks = []
            all_wid_masks = []

            all_bboxes = []
            all_masks = []
            all_labels = []

            for i, idx in enumerate(idx_list):
                f, rgb               = self._load_image(idx)
                depth                = self._load_depth(idx)
                mask, bboxes, labels = self._load_mask(idx)
                # grasp angle in degree
                grasps               = self._load_grasps(idx)

                new_pos_masks = np.zeros((image_size, image_size))
                new_qua_masks = np.zeros((image_size, image_size))
                new_ang_masks = np.zeros((image_size, image_size))
                new_wid_masks = np.zeros((image_size, image_size))
                new_masks = np.zeros((image_size, image_size))

                pos_masks, qua_masks, ang_masks, wid_masks = self._draw_grasp_rects([grasps], self.target_size, self.target_size, bboxes[0])

                mask = np.squeeze(mask)
                depth = np.squeeze(depth)

                pixel_idx_x, pixel_idx_y = np.where(mask==1)

                rgb_value = rgb[pixel_idx_x, pixel_idx_y]
                dep_value = depth[pixel_idx_x, pixel_idx_y]
                pos_value = np.squeeze(pos_masks)[pixel_idx_x, pixel_idx_y]
                qua_value = np.squeeze(qua_masks)[pixel_idx_x, pixel_idx_y]
                ang_value = np.squeeze(ang_masks)[pixel_idx_x, pixel_idx_y]
                wid_value = np.squeeze(wid_masks)[pixel_idx_x, pixel_idx_y]

                lt_x = np.min(pixel_idx_x)
                lt_y = np.min(pixel_idx_y)

                pixel_idx_x = pixel_idx_x - lt_x
                pixel_idx_y = pixel_idx_y - lt_y

                bboxes = [[bbox[0]-lt_y, bbox[1]-lt_x, bbox[2]-lt_y, bbox[3]-lt_x] for bbox in bboxes]

                # obj_pos = (np.random.random_sample(size=2) * np.array([int(image_size * 0.75), int(image_size * 0.75)])).astype("int")
                obj_pos = (np.random.random_sample(size=2) * self.sample_range[i][1] + self.sample_range[i][0]).astype("int")

                pixel_idx_x = pixel_idx_x + obj_pos[0]
                pixel_idx_y = pixel_idx_y + obj_pos[1]

                valid_pixel_mask = np.logical_and((pixel_idx_x < 544), (pixel_idx_y < 544))

                pixel_idx_x = pixel_idx_x[valid_pixel_mask]
                pixel_idx_y = pixel_idx_y[valid_pixel_mask]
                
                new_pos_masks[pixel_idx_x, pixel_idx_y] = pos_value[valid_pixel_mask]
                new_qua_masks[pixel_idx_x, pixel_idx_y] = qua_value[valid_pixel_mask]
                new_ang_masks[pixel_idx_x, pixel_idx_y] = ang_value[valid_pixel_mask]
                new_wid_masks[pixel_idx_x, pixel_idx_y] = wid_value[valid_pixel_mask]
                new_masks[pixel_idx_x, pixel_idx_y] = 1

                all_image[pixel_idx_x, pixel_idx_y] = rgb_value[valid_pixel_mask]
                all_depth[pixel_idx_x, pixel_idx_y] = dep_value[valid_pixel_mask]
                all_pos_masks.append(new_pos_masks)
                all_qua_masks.append(new_qua_masks)
                all_ang_masks.append(new_ang_masks)
                all_wid_masks.append(new_wid_masks)
                all_masks.append(new_masks)

                bboxes = [np.clip([bbox[0]+obj_pos[1], bbox[1]+obj_pos[0], bbox[2]+obj_pos[1], bbox[3]+obj_pos[0]], 0, image_size-2) for bbox in bboxes]

                all_bboxes.extend(bboxes)
                all_labels.append(1)
            

            all_pos_masks = np.array(all_pos_masks)
            all_qua_masks = np.array(all_qua_masks)
            all_ang_masks = np.array(all_ang_masks)
            all_wid_masks = np.array(all_wid_masks)
            all_masks = np.array(all_masks)
            all_bboxes = np.array(all_bboxes)
            all_labels = np.array(all_labels)

            all_depth = 1 - (all_depth / np.max(all_depth))
            all_image = all_image.astype("uint8")

            # self._show_data(all_image, all_depth, all_masks, all_bboxes, pos_masks=all_pos_masks, qua_masks=all_qua_masks, ang_masks=all_ang_masks, wid_masks=all_wid_masks)
            
            return all_image, all_depth, all_masks, all_bboxes, all_labels, all_pos_masks, all_qua_masks, all_ang_masks, all_wid_masks

            # self._show_data(all_image, all_depth, all_masks, all_bboxes, pos_masks=all_pos_masks, qua_masks=all_qua_masks, ang_masks=all_ang_masks, wid_masks=all_wid_masks)

            # cv2.imwrite("test_multi_obj.png", all_image.astype("uint8"))
            # cv2.imwrite("test_multi_obj_dep.png", (all_depth*255).astype("uint8"))

                # Standardlize the pixel idx -min_x, -min_y
                # Get corresponding depth value, RGB, value

                # center_x, center_y = int((bboxes[0][0]+bboxes[0][2])/2), int((bboxes[0][1]+bboxes[0][3])/2) 

                # new_bbox = [bboxes[0][0]-center_x, bboxes[0][1]-center_y, bboxes[0][2]-center_x, bboxes[0][3]-center_y]
                # new_grasps = [[grasp[0]-center_x, grasp[1]-center_y, grasp[2], grasp[3], grasp[4], grasp[5]] for grasp in grasps]



    
    def __getitem__(self, index):
        if self.multi_obj_aug:
            num_obj = random.randint(2, 4)
            # num_obj = 3
            rgb, depth, mask, bboxes, labels, pos_masks, qua_masks, ang_masks, wid_masks = self._multi_obj_aug(index, num_obj)
        else:
            f, rgb               = self._load_image(index)
            depth                = self._load_depth(index, normalize=True)
            mask, bboxes, labels = self._load_mask(index)
            # grasp angle in degree
            grasps               = self._load_grasps(index)

            pos_masks, qua_masks, ang_masks, wid_masks = self._draw_grasp_rects([grasps], self.target_size, self.target_size, bboxes[0])


        # self._show_data(rgb, depth, mask, bboxes, grasps, pos_masks, qua_masks, ang_masks, wid_masks)

        if self.mode == "train":
            img, depth, ins_masks, pos_masks, qua_masks, ang_masks, wid_masks, bboxes, labels = self.transform(
                rgb, depth, mask, pos_masks, qua_masks, ang_masks, wid_masks, bboxes[:, :4], labels
            )

            if len(depth.shape) == 2:
                depth = np.expand_dims(depth, -1)

            rgbd = np.concatenate([img, depth], axis=-1).transpose((2,0,1))
            bboxes = np.concatenate([bboxes, labels.reshape(-1,1)], axis=-1)

            sin_masks = np.sin(2 * ang_masks)
            cos_masks = np.cos(2 * ang_masks)

            self._show_data(np.ascontiguousarray(img, dtype=np.int8)*255, depth, ins_masks, bboxes[:,:4]*544, pos_masks=pos_masks, qua_masks=qua_masks, ang_masks=ang_masks, wid_masks=wid_masks)

            return rgbd, bboxes, ins_masks, pos_masks, qua_masks, sin_masks, cos_masks, wid_masks

        elif self.mode == "test":
            img, depth, ins_masks, pos_masks, qua_masks, ang_masks, wid_masks, bboxes, labels = self.transform(
                rgb, depth, mask, pos_masks, qua_masks, ang_masks, wid_masks, bboxes[:, :4], labels
            )

            if len(depth.shape) == 2:
                depth = np.expand_dims(depth, -1)
            # self.show_data(img, depth, np.array([bboxes[2]]), np.array([labels[2]]), ins_masks=np.array([ins_masks[2]]), pos_masks=np.array([pos_masks[2]]), ang_masks=np.array([ang_masks[2]]), wid_masks=np.array([wid_masks[2]]))

            rgbd = np.concatenate([img, depth], axis=-1).transpose((2,0,1))
            bboxes = np.concatenate([bboxes, labels.reshape(-1,1)], axis=-1)

            # Normalize angle mask to [-1,1]
            sin_masks = np.sin(2 * ang_masks)
            cos_masks = np.cos(2 * ang_masks)

            # sself._show_data(np.ascontiguousarray(img, dtype=np.int8)*255, depth, np.array([ins_masks]), bboxes[:,:4], pos_masks=np.array([pos_masks]), qua_masks=np.array([qua_masks]), ang_masks=np.array([ang_masks]), wid_masks=np.array([wid_masks]))


            return f, rgbd, bboxes, grasps, ins_masks, pos_masks, qua_masks, ang_masks, wid_masks
        

        


if __name__ == "__main__":
    from tqdm import tqdm
    import matplotlib.pyplot as plt

    dataset = JacquardGraspDataset(
        "/home/puzek/sdb/dataset/JACQUARD/jacquard", 
        origin_size=1024, 
        target_size=544, 
        transform=gr_train_aug_jacquard, 
        mode="train",
        multi_obj_aug=True
    )
    bg_value = []
    pbar = tqdm(range(len(dataset)))

    dataset[1000]

    # for i in pbar:
        
    #     val = dataset[i]
    #     bg_value.append(val)
    #     if i % 10 == 0:
    #         print(val)
    
    

    # bg_value = np.concatenate(bg_value, axis=0)

    # print(bg_value.shape)

    # max_width = 0
    # mean_width = 0
    # num_grasp = 0

    # width_data = []
    # angle_data = []
    # total_count = 0
    # min_count = 0
    # max_count = 0

    # over_max = []
    # over_min = []

    # pbar = tqdm(range(len(dataset)))
    # for i in pbar:
    #     bboxes, grasps = dataset[i]
    #     bbox = bboxes[0]
    #     width = bbox[2] - bbox[0]
    #     height = bbox[3] - bbox[1]

    #     min_len = min(width, height)
    #     max_len = max(width, height)
    #     num_grasp += len(grasps)

    #     for grasp in grasps:
    #         width_data.append(grasp[2])
    #         angle_data.append(grasp[-1])
    #         total_count += 1
    #         if grasp[2] < min_len:
    #             min_count += 1
    #             max_count += 1
    #         elif grasp[2] < max_len:
    #             over_min.append((grasp[2] - min_len)/float(min_len)*100)
    #             max_count += 1
    #         else:
    #             over_max.append((grasp[2] - max_len)/float(max_len)*100)
    
    # print(min_count, total_count)
    # print(max_count, total_count)
    # print(float(min_count / float(total_count)))
    # print(float(max_count / float(total_count)))

    # print(len(over_min))
    # print(len(over_max))

    
    # np.savez(
    #     "data.npz", 
    #     angles=np.array(angle_data), 
    #     widths=np.array(width_data), 
    #     min_count=min_count, 
    #     max_count=max_count, 
    #     total_count=total_count, 
    #     over_min=over_min, 
    #     over_max=over_max
    # )
    

    # data = np.load("data.npz")
    # print(data.files)
    # angles = data["angles"]
    # widths = data["widths"]
    # over_min = data["over_min"]
    # over_max = data["over_max"]

    # ax = plt.subplot(2,2,1)
    # ax.hist(angles, bins=int(180/5), color="cyan", edgecolor="black")
    # ax.set_title("Angles Distribution")
    # ax.set_xlabel("angle(degree)")
    # ax.set_ylabel("Numbers")


    # ax = plt.subplot(2,2,2)
    # ax.hist(widths, bins=int(300/5), color="coral", edgecolor="black")
    # ax.set_title("Widths Distribution")
    # ax.set_xlabel("width(pixel)")
    # ax.set_ylabel("Numbers")

    # ax = plt.subplot(2,2,3)
    # ax.hist(over_min, bins=int(100/2), color="coral", edgecolor="black")
    # ax.set_title("Over min edge of bbox Distribution")
    # ax.set_xlabel("Over")
    # ax.set_ylabel("Pixels")

    # ax = plt.subplot(2,2,4)
    # ax.hist(over_max, bins=int(100/2), color="coral", edgecolor="black")
    # ax.set_title("Over max edge of bbox Distribution")
    # ax.set_xlabel("Over")
    # ax.set_ylabel("Pixels")

    # plt.tight_layout()
    # plt.show()






