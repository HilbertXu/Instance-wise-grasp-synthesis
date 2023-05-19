import os
import cv2
import xmltodict
import numpy as np
from skimage.measure import regionprops
from skimage.draw import polygon


import torch
import torch.utils.data as data


import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable



class OSGDDataset(data.Dataset):
    def __init__(self, root_path, transform=None, mode="train") -> None:
        super(OSGDDataset, self).__init__()

        self.root_path = root_path
        self.transform = transform
        self.mode = mode
        self.target_size = 544

        if mode == "train":
            img_set = os.path.join(root_path, "ImageSets/Main/trainval.txt")
            with open(img_set, 'r') as f:
                self.data = f.readlines()
        elif mode == "test":
            img_set = os.path.join(root_path, "ImageSets/Main/test.txt")
            with open(img_set, 'r') as f:
                self.data = f.readlines()
            

        self.cls_to_label = {
            'background':0,
            'knife': 1,
            'pen': 2,
            'hammer': 3,
            'fork': 4,
            'spatula': 5,
            'wrench': 6, 
            'pliers': 7,
            'screwdriver': 8,
            'spoon': 9,
            'toothbrush': 10
        }

        self.actions = ['hand_over', 'cut', 'write', 'hammer', 'fork', 'shovel', 'wrench', 'pinch', 'screw', 'ladle', 'brush']


        self.per_class_max_width = [192.85486771144772, 98.40731680114037, 150.6154042583799, 108.55873986004075, 194.11594473424896, 185.88437266214711, 258.69866640523145, 131.5294643796591, 167.32304085211933, 75.8023746329889]

    def __len__(self):
        return len(self.data)

    
    def visualize_data(self, depth, bboxes, obj_index, all_grasps, pos_masks, qua_masks, ang_masks, wid_masks, aff_masks, tgt_obj=[]):
        h, w, c = depth.shape

        depth = depth * 255

        tgt_idx = 3
        target_dir = "./osgd/"

        for idx, bbox, obj_grasps in zip(obj_index, bboxes, all_grasps):
            if idx == tgt_idx:
                cv2.rectangle(depth, (int(bbox[0]), int(bbox[1])),
                            (int(bbox[2]), int(bbox[3])), (0,0,255), 1)
                
                for grasp in obj_grasps:
                    grasp_conf, idx, act = grasp
                    center_x, center_y, width, height, theta = grasp_conf
                    print(theta)

                    box = ((center_x, center_y), (width, height),  90-theta)
                    box = cv2.boxPoints(box)
                    box = np.int0(box)
                    cv2.drawContours(depth, [box], 0, (255,0,0), 2)
        

        for idx, pos_mask, qua_mask, ang_mask, wid_mask, aff_masks in zip(obj_index, pos_masks, qua_masks, ang_masks, wid_masks, aff_masks):
            if idx == tgt_idx:
                plt.figure()
                ax = plt.gca()
                im = ax.imshow(pos_mask, cmap='jet', vmin=0, vmax=1)
                plt.axis("off")
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = plt.colorbar(im, cax=cax)
                for t in cbar.ax.get_yticklabels():
                    t.set_fontsize(20)
                plt.savefig(target_dir+'pos_mask.png', bbox_inches='tight',pad_inches = 0)
                plt.cla()
                plt.clf()

                plt.figure()
                ax = plt.gca()
                im = ax.imshow(qua_mask, cmap='jet', vmin=0, vmax=1)
                plt.axis("off")
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = plt.colorbar(im, cax=cax)
                for t in cbar.ax.get_yticklabels():
                    t.set_fontsize(20)
                plt.savefig(target_dir+'qua_mask.png', bbox_inches='tight',pad_inches = 0)
                plt.cla()
                plt.clf()


                plt.figure()
                ax = plt.gca()
                im = ax.imshow(ang_mask, cmap='rainbow', vmin=-np.pi / 2, vmax=np.pi / 2)
                plt.axis("off")
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = plt.colorbar(im, cax=cax)
                for t in cbar.ax.get_yticklabels():
                    t.set_fontsize(20)
                plt.savefig(target_dir+'ang_mask.png', bbox_inches='tight',pad_inches = 0)
                plt.cla()
                plt.clf()

                plt.figure()
                ax = plt.gca()
                im = ax.imshow(wid_mask, cmap='jet', vmin=0, vmax=1)
                plt.axis("off")
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = plt.colorbar(im, cax=cax)
                for t in cbar.ax.get_yticklabels():
                    t.set_fontsize(20)
                plt.savefig(target_dir+'wid_mask.png', bbox_inches='tight',pad_inches = 0)
                plt.cla()
                plt.clf()

                for i in range(aff_masks.shape[0]):
                    plt.figure()
                    ax = plt.gca()
                    im = ax.imshow(aff_masks[i], cmap='jet', vmin=0, vmax=1)
                    plt.axis("off")
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    cbar = plt.colorbar(im, cax=cax)
                    for t in cbar.ax.get_yticklabels():
                        t.set_fontsize(20)
                    
                    aff_name = self.actions[i]

                    plt.savefig(target_dir+'{}_mask.png'.format(aff_name), bbox_inches='tight',pad_inches = 0)
                    plt.cla()
                    plt.clf()

        
        cv2.imwrite("osgd/test_depth.png", depth)


    def load_annos(self, index, scale=1.0):
        annos_f = os.path.join(self.root_path, "Annotations/{:06d}.xml".format(index))
        with open(annos_f, 'r') as f:
            data = f.read()
            dict_data = xmltodict.parse(data)

        names = []
        obj_indexs = []
        bboxes = []
        
        for obj in dict_data['annotations']['object']:
            name = obj['name']
            label = self.cls_to_label[name]
            idx = int(obj['index']) - 1
            bbox = [int(obj['bndbox']['xmin'])/500., int(obj['bndbox']['ymin'])/500., int(obj['bndbox']['xmax'])/500., int(obj['bndbox']['ymax'])/500., label]

            names.append(name)
            obj_indexs.append(idx)
            bboxes.append(bbox)
        
        return names, np.asarray(obj_indexs), np.asarray(bboxes)

    

    def load_depth(self, index):
        depth_f = os.path.join(self.root_path, "JPEGImages/{:06d}.jpg".format(index))
        depth = 1 - (cv2.imread(depth_f, cv2.IMREAD_UNCHANGED) / 255.)

        # cv2.imwrite("test_depth.png", depth * 255)

        return depth

    
    def load_grasp(self, index, scale = 1.0):
        grasp_f = os.path.join(self.root_path, "Grasps/{:06d}.txt".format(index))
        with open(grasp_f, 'r') as f:
            data = f.readlines()
        grasps = []
        for line in data:
            p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y, idx, act = line.split(" ")
            center_x = (float(p1x)+float(p3x)) / 2.
            center_y = (float(p1y)+float(p3y)) / 2.
            width = np.sqrt((float(p1x) - float(p4x)) * (float(p1x) - float(p4x)) + (float(p1y) - float(p4y)) * (float(p1y) - float(p4y)))
            height = np.sqrt((float(p1x) - float(p2x)) * (float(p1x) - float(p2x)) + (float(p1y) - float(p2y)) * (float(p1y) - float(p2y)))
            theta = np.arctan2(float(p4x) - float(p1x), float(p4y) - float(p1y)) * 180 / np.pi

            # if theta > 90:
            #     theta = theta - 180
            
            # if theta < -90:
            #     theta = theta + 180
            
            grasps.append((
                [
                    center_x*scale, center_y*scale, 
                    width*scale, height*scale,
                    theta
                ],
                idx, act.strip()
            ))
        
        return grasps

    
    def draw_grasp_masks(self, names, grasps, height, width):
        num_act = len(self.actions)
        pos_masks = []
        qua_masks = []
        ang_masks = []
        wid_masks = []
        aff_masks = []
        loss_mask = []

        for name, obj_grasps in zip(names, grasps):
            pos_out = np.zeros((height, width)) + 1e-4
            qua_out = np.zeros((height, width)) + 1e-4
            ang_out = np.zeros((height, width)) + 1e-4
            wid_out = np.zeros((height, width)) + 1e-4
            aff_out = np.zeros((num_act, height, width)) + 1e-4

            tmp = [1e-4 for i in range(11)]

            for grasp in obj_grasps:
                conf, idx, act = grasp
                width_factor = float(self.per_class_max_width[self.cls_to_label[name]-1])
                center_x, center_y, w_rect, h_rect, theta = conf


                
                r_rect = ((center_x, center_y), (w_rect/2, h_rect), 90-theta)
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

                act_idx = self.actions.index(act)
                tmp[act_idx] = 1.0
                aff_out[act_idx, cc, rr] = 1.0
            
            # Preprocessing quality mask
            qua_out = 1 / (1 + np.exp(-qua_out))
            qua_out = qua_out * pos_out
            smooth_factor = 1e-7

            qua_out = np.clip(qua_out, smooth_factor, 1-smooth_factor)

            pos_masks.append(pos_out)
            qua_masks.append(qua_out)
            ang_masks.append(ang_out)
            wid_masks.append(wid_out)
            aff_masks.append(aff_out)
            loss_mask.append(tmp)
    
        return np.clip(np.array(pos_masks), 1e-4, 1.0), np.clip(np.array(qua_masks), 1e-4, 1.0), np.array(ang_masks), np.clip(np.array(wid_masks), 1e-4, 1.0), np.clip(np.array(aff_masks), 1e-4, 1.0), np.clip(np.array(loss_mask), 1e-4, 1.0)

        

    def __getitem__(self, index):
        names, obj_index, bboxes = self.load_annos(index)
        depth = self.load_depth(index)
        depth = cv2.resize(depth, (self.target_size, self.target_size))

        all_grasps = self.load_grasp(index, scale = 544 / 500.)

        assert len(obj_index) == len(bboxes)

        # self.visualize_data(depth, bboxes, obj_index, all_grasps)

        # Assign grasp to each object
        per_instance_grasp = [[] for i in range(len(obj_index))]
        labels = []
        for i in range(len(obj_index)):
            name = names[i]
            labels.append(self.cls_to_label[name])
            obj_idx = obj_index[i]
            for grasp in all_grasps:
                g_idx = grasp[1]

                if int(g_idx) == int(obj_idx):
                    per_instance_grasp[i].append(grasp)


        pos_masks, qua_masks, ang_masks, wid_masks, aff_masks, loss_masks = self.draw_grasp_masks(names, per_instance_grasp, self.target_size, self.target_size)

        # self.visualize_data(depth, bboxes[:, :-1]*544, obj_index, per_instance_grasp, pos_masks, pos_masks, ang_masks, wid_masks, aff_masks)
        
        sin_masks = np.sin(2 * ang_masks)
        cos_masks = np.cos(2 * ang_masks)
        loss_masks = np.array(loss_masks)


        return np.expand_dims(depth[:,:,0], axis=0), bboxes, per_instance_grasp, pos_masks, qua_masks, sin_masks, cos_masks, wid_masks, aff_masks, loss_masks


if __name__ == "__main__":
    dataset = OSGDDataset(root_path="/home/puzek/sdb/dataset/OSGD_IMG/taskoritv0", transform=None, mode="train")

    depth, bboxes, per_instance_grasp, pos_masks, qua_masks, sin_masks, cos_masks, wid_masks, aff_masks, loss_masks = dataset[0]
    
    # max_pos = -999
    # min_pos = 999
    # min_cls = 999
    # max_cls = -999

    # from tqdm import tqdm

    # pbar = tqdm(range(len(dataset)))


    # for i in pbar:
    #     depth, bboxes, per_instance_grasp, pos_masks, qua_masks, sin_masks, cos_masks, wid_masks, aff_masks, loss_masks = dataset[i]
    #     cls_labels = bboxes[:,-1]

    #     if np.max(cls_labels) > max_cls:
    #         max_cls = np.max(cls_labels)
    #     if np.min(cls_labels) < min_cls:
    #         min_cls = np.min(cls_labels)
    #     if np.max(pos_masks) > max_pos:
    #         max_pos = np.max(pos_masks)
    #     if np.min(pos_masks) < min_pos:
    #         min_pos = np.min(pos_masks)
    
    # print(max_pos, min_pos, max_cls, min_cls)



        
    
    
