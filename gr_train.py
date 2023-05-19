import time
from functools import partial
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.utils.data as data
from tensorboardX import SummaryWriter
import argparse
import datetime
import re

from utils import timer
from modules.gr_yolact import GraspYolact
from config import get_config
from utils.coco import train_collate
from utils.common_utils import save_best, save_latest
from gr_eval import evaluate

from graspnet import GraspNetDetection
from ocid_grasp import OCIDGraspDataset
from utils.gr_augmentation import gr_train_aug, gr_val_aug

parser = argparse.ArgumentParser(description='SSG Training Script')
parser.add_argument('--local_rank', type=int, default=None)
parser.add_argument('--cfg', default='res101_coco', help='The configuration name to use.')
parser.add_argument('--train_bs', type=int, default=8, help='total training batch size')
parser.add_argument('--img_size', default=544, type=int, help='The image size for training.')
parser.add_argument('--resume', default=None, type=str, help='The path of the weight file to resume training with.')
parser.add_argument('--val_interval', default=4000, type=int,
                    help='The validation interval during training, pass -1 to disable.')
parser.add_argument('--val_num', default=-1, type=int, help='The number of images for test, set to -1 for all.')
parser.add_argument('--traditional_nms', default=False, action='store_true', help='Whether to use traditional nms.')
parser.add_argument('--coco_api', action='store_true', help='Whether to use cocoapi to evaluate results.')

# for numpy randomness
# import numpy as np
# np.random.seed(10)
# # for randomness in image augmentation
# import random
# random.seed(10)
# # every PyTorch thing can be fixed with these two lines
# torch.manual_seed(10)
# torch.cuda.manual_seed_all(10)

args = parser.parse_args()
cfg = get_config(args, mode='train')
cfg_name = cfg.__class__.__name__

net = GraspYolact(cfg)
net.train()

if args.resume:
    with torch.no_grad():
        import torch.nn as nn
        net.backbone.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    net.load_weights(torch.load(args.resume), cfg.cuda)
    
    start_step = int(cfg.weight.split('.pth')[0].split('_')[-1])
    print("Resume from {}".format(args.resume))

else:
    net.backbone.init_backbone(cfg.weight)
    start_step = 0

    with torch.no_grad():
        import torch.nn as nn
        # Add extra depth channel for net.backbone
        weight = net.backbone.conv1.weight.clone()
        net.backbone.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        net.backbone.conv1.weight[:,:3] = weight

dataset = cfg.dataset
eval_dataset = cfg.val_dataset


if 'res' in cfg.__class__.__name__:
    optimizer = optim.SGD(net.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=5e-4)
    # optimizer = optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=0.05)
elif cfg.__class__.__name__ == 'swin_tiny_coco':
    optimizer = optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=0.05)
else:
    raise ValueError('Unrecognized cfg.')

train_sampler = None
main_gpu = False
num_gpu = 0
if cfg.cuda:
    cudnn.benchmark = True
    cudnn.fastest = True
    main_gpu = dist.get_rank() == 0
    num_gpu = dist.get_world_size()

    net = DDP(net.cuda(), [args.local_rank], output_device=args.local_rank, broadcast_buffers=True)
    train_sampler = DistributedSampler(dataset, shuffle=True)

# shuffle must be False if sampler is specified
data_loader = data.DataLoader(dataset, cfg.bs_per_gpu, num_workers=cfg.bs_per_gpu // 2, shuffle=(train_sampler is None),
                              collate_fn=train_collate, pin_memory=False, sampler=train_sampler)
# data_loader = data.DataLoader(dataset, cfg.bs_per_gpu, num_workers=0, shuffle=False,
#                               collate_fn=train_collate, pin_memory=True)

epoch_seed = 0
map_tables = []
training = True
timer.reset()
step = start_step
val_step = start_step
max_rate = 0.0
writer = SummaryWriter('tensorboard_log/{}'.format(cfg.summary_comment))

if main_gpu:
    print(f'Number of all parameters: {sum([p.numel() for p in net.parameters()])}\n')

try:  # try-except can shut down all processes after Ctrl + C.
    while training:
        if train_sampler:
            epoch_seed += 1
            train_sampler.set_epoch(epoch_seed)
        
        for (images, targets, masks, pos_masks, qua_masks, sin_masks, cos_masks, wid_masks) in data_loader:
            if cfg.warmup_until > 0 and step <= cfg.warmup_until:  # warm up learning rate.
                for param_group in optimizer.param_groups:
                    param_group['lr'] = (cfg.lr - cfg.warmup_init) * (step / cfg.warmup_until) + cfg.warmup_init

            if step in cfg.lr_steps:  # learning rate decay.
                for param_group in optimizer.param_groups:
                    param_group['lr'] = cfg.lr * 0.1 ** cfg.lr_steps.index(step)

            if cfg.cuda:
                images = images.cuda().detach()
                targets = [ann.cuda().detach() for ann in targets]
                masks = [mask.cuda().detach() for mask in masks]
                pos_masks = [pos_mask.cuda().detach() for pos_mask in pos_masks]
                qua_masks = [qua_mask.cuda().detach() for qua_mask in qua_masks]
                sin_masks = [sin_mask.cuda().detach() for sin_mask in sin_masks]
                cos_masks = [cos_mask.cuda().detach() for cos_mask in cos_masks]
                wid_masks = [wid_mask.cuda().detach() for wid_mask in wid_masks]


            with timer.counter('for+loss'):
                loss_c, loss_b, loss_m, loss_g_pos, loss_g_qua, loss_g_ang, loss_g_wid, loss_s = net(images, targets, masks, pos_masks, qua_masks, sin_masks, cos_masks, wid_masks)

                if cfg.cuda:
                    # use .all_reduce() to get the summed loss from all GPUs
                    all_loss = torch.stack([loss_c, loss_b, loss_m, loss_g_pos, loss_g_qua, loss_g_ang, loss_g_wid, loss_s], dim=0)
                    dist.all_reduce(all_loss)

            with timer.counter('backward'):
                loss_total = loss_c + loss_b + loss_m + loss_g_pos + loss_g_qua + loss_g_ang + loss_g_wid + loss_s
                optimizer.zero_grad()
                loss_total.backward()

            with timer.counter('update'):
                optimizer.step()

            time_this = time.time()
            if step > start_step:
                batch_time = time_this - time_last
                timer.add_batch_time(batch_time)
            time_last = time_this

            if step % 10 == 0 and step != start_step:
                if (not cfg.cuda) or main_gpu:
                    cur_lr = optimizer.param_groups[0]['lr']
                    time_name = ['batch', 'data', 'for+loss', 'backward', 'update']
                    t_t, t_d, t_fl, t_b, t_u = timer.get_times(time_name)
                    seconds = (cfg.lr_steps[-1] - step) * t_t
                    eta = str(datetime.timedelta(seconds=seconds)).split('.')[0]

                    # Get the mean loss across all GPUS for printing, seems need to call .item(), not sure
                    l_c = all_loss[0].item() / num_gpu if main_gpu else loss_c.item()
                    l_b = all_loss[1].item() / num_gpu if main_gpu else loss_b.item()
                    l_m = all_loss[2].item() / num_gpu if main_gpu else loss_m.item()
                    l_g_pos = all_loss[3].item() / num_gpu if main_gpu else loss_m.item()
                    l_g_qua = all_loss[4].item() / num_gpu if main_gpu else loss_m.item()
                    l_g_ang = all_loss[5].item() / num_gpu if main_gpu else loss_m.item()
                    l_g_wid = all_loss[6].item() / num_gpu if main_gpu else loss_m.item()
                    l_s = all_loss[7].item() / num_gpu if main_gpu else loss_s.item()


                    writer.add_scalar('loss/class', l_c, global_step=step)
                    writer.add_scalar('loss/box', l_b, global_step=step)
                    writer.add_scalar('loss/mask', l_m, global_step=step)
                    writer.add_scalar('loss/grasp_pos', l_g_pos, global_step=step)
                    writer.add_scalar('loss/grasp_qua', l_g_qua, global_step=step)
                    writer.add_scalar('loss/grasp_ang', l_g_ang, global_step=step)
                    writer.add_scalar('loss/grasp_wid', l_g_wid, global_step=step)
                    writer.add_scalar('loss/semantic', l_s, global_step=step)
                    writer.add_scalar('loss/total', loss_total, global_step=step)

                    print(f'[Co:Gr-v2]step: {step} | lr: {cur_lr:.2e} | l_class: {l_c:.3f} | l_box: {l_b:.3f} | '
                          f'l_mask: {l_m:.3f} | l_g_pos: {l_g_pos:.3f} | l_g_qua: {l_g_qua:.3f} | l_g_ang: {l_g_ang:.3f} | l_g_wid: {l_g_wid:.3f} | l_semantic: {l_s:.3f} | t_t: {t_t:.3f} | t_d: {t_d:.3f} | '
                          f't_fl: {t_fl:.3f} | t_b: {t_b:.3f} | t_u: {t_u:.3f} | ETA: {eta}')

            if step % 2000 == 0 and step != start_step:
                torch.save(net.module.state_dict(), 'weights/{}/latest_CoGrv2_{}.pth'.format(cfg.weight_dir, step))

            if args.val_interval > 0 and step % args.val_interval == 0 and step != start_step and eval_dataset is not None:
                if (not cfg.cuda) or main_gpu:
                    val_step = step
                    net.eval()
                    class_rate, overrall_rate = evaluate(net.module, eval_dataset, cfg)
                    net.train()
                    timer.reset()  # training timer and val timer share the same Obj, so reset it to avoid conflict

                    writer.add_scalar('grasping_successful_rate', overrall_rate, global_step=step)

                    if overrall_rate > max_rate:
                        torch.save(net.module.state_dict(), "weights/{}/CoGrv2_{:.2f}.pth".format(cfg.weight_dir, overrall_rate*100))
                        max_rate = overrall_rate

            if ((not cfg.cuda) or main_gpu) and step == val_step + 1:
                timer.start()  # the first iteration after validation should not be included

            step += 1
            if step >= cfg.lr_steps[-1]:
                training = False

                if (not cfg.cuda) or main_gpu:
                    torch.save(net.module.state_dict(), 'weights/{}/latest_CoGrv2_{}.pth'.format(cfg.weight_dir, step))

                    print('\nValidation results during training:\n')
                    for table in map_tables:
                        print(table, '\n')

                    print(f'Training completed.')

                break

except KeyboardInterrupt:
    if (not cfg.cuda) or main_gpu:
        torch.save(net.module.state_dict(), 'weights/{}/latest_CoGrv2_{}.pth'.format(cfg.weight_dir, step))

        print('\nValidation results during training:\n')
        for table in map_tables:
            print(table, '\n')
