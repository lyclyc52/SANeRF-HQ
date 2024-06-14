
import os
import glob
import tqdm
import imageio
import wandb
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import cv2
import torch.distributed as dist
import matplotlib.pyplot as plt

from rich.console import Console
from torch_ema import ExponentialMovingAverage
from nerf.utils import Cache
from nerf.utils import overlay_mask, overlay_mask_heatmap, overlay_mask_composition, overlay_mask_only, overlay_point
from nerf.utils import get_rays, get_incoherent_mask
class Trainer(object):
    def __init__(self,
                 name,  # name of this experiment
                 opt,  # extra conf
                 model,  # network
                 criterion=None,  # loss function, if None, assume inline implementation in train_step
                 optimizer=None,  # optimizer
                 ema_decay=None,  # if use EMA, set the decay
                 lr_scheduler=None,  # scheduler
                 # metrics for evaluation, if None, use val_loss to measure performance, else use the first metric.
                 metrics=[],
                 local_rank=0,  # which GPU am I
                 world_size=1,  # total num of GPUs
                 # device to use, usually setting to None is OK. (auto choose device)
                 device=None,
                 mute=False,  # whether to mute all print
                 fp16=False,  # amp optimize level
                 eval_interval=1,  # eval once every $ epoch
                 # save once every $ epoch (independently from eval)
                 save_interval=1,
                 max_keep_ckpt=2,  # max num of saved ckpts in disk
                 workspace='workspace',  # workspace to save logs & ckpts
                 best_mode='min',  # the smaller/larger result, the better
                 use_loss_as_metric=True,  # use loss as the first metric
                 report_metric_at_train=False,  # also report metrics at training
                 use_checkpoint="latest",  # which ckpt to use at init time
                 use_tensorboardX=False,  # whether to use tensorboard for logging
                 # whether to call scheduler.step() after every train step
                 scheduler_update_every_step=False,
                 sam_predictor=None,
                 ):

        self.opt = opt
        self.name = name
        self.mute = mute
        self.metrics = metrics
        self.local_rank = local_rank
        self.world_size = world_size
        self.workspace = workspace
        self.ema_decay = ema_decay
        self.fp16 = fp16
        self.best_mode = best_mode
        self.use_loss_as_metric = use_loss_as_metric
        self.report_metric_at_train = report_metric_at_train
        self.max_keep_ckpt = max_keep_ckpt
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.use_checkpoint = use_checkpoint
        self.track_trajectory = False
        
        if self.opt.trajectory_root is not None:
            os.makedirs(self.opt.trajectory_root, exist_ok=True)
            
        self.trajectories = []
        self.use_tensorboardX = use_tensorboardX
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.scheduler_update_every_step = scheduler_update_every_step
        self.device = device if device is not None else torch.device(
            f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        self.console = Console()
        
        self.sam_predictor = sam_predictor
        self.sam_output_dim = 256

        self.point_3d = None
        self.input_labels = None
        
        if self.opt.use_point and self.opt.point_file is not None:
            with open(self.opt.point_file) as f:
                point_json = json.load(f)
            self.point_3d = torch.tensor(point_json['points'])
            self.input_labels = torch.ones(self.point_3d.shape[0])
            # which point should have negative labels
            for i in point_json['negative_labels']:
                self.input_labels[i] = 0
            # Valid views must contain the crucial points.
            # Sometimes it will be useful when the object has multiple componets. The crucial point can help to ensure that 
            # most parts of the objects are visible. It may help you to filter some invalid views. We add this function at the 
            # beginning but barely used this finally.
            self.crucial_point_label = torch.zeros(self.point_3d.shape[0])
            self.crucial_point_count = len(point_json['crucial_point_index'])
            # which point should have negative labels
            for i in point_json['crucial_point_index']:
                self.crucial_point_label[i] = 1
            # valid views must contain points larger than this threshold
            self.valid_threshold = point_json['valid_threshold']
            if self.valid_threshold == -1:
                self.valid_threshold = int(self.point_3d.shape[0] * 0.8) + 1
            self.point_3d = self.point_3d.to(self.device)
            self.input_labels = self.input_labels.to(self.device)
            self.crucial_point_label = self.crucial_point_label.to(self.device)
        # for GUI
        self.last_masks = None
        # for cache
        self.cache = Cache(self.opt.cache_size)

        model.to(self.device)
        if self.world_size > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank])
        self.model = model

        if isinstance(criterion, nn.Module):
            criterion.to(self.device)
        self.criterion = criterion

        self.color_map = np.multiply([
            plt.cm.get_cmap('gist_ncar', 100)((i * 7 + 5) % 100)[:3] for i in range(100)
        ], 1)
        self.color_map = torch.from_numpy(
            self.color_map).to(self.device).to(torch.float)

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        if ema_decay is not None:
            self.ema = ExponentialMovingAverage(
                self.model.parameters(), decay=ema_decay)
        else:
            self.ema = None


        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        # variable init
        self.epoch = 0
        self.global_step = 0
        self.local_step = 0
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [],  # metrics[0], or valid_loss
            "checkpoints": [],  # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
        }

        # auto fix
        if len(metrics) == 0 or self.use_loss_as_metric:
            self.best_mode = 'min'

        # workspace prepare
        self.log_ptr = None
        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)
            self.log_path = os.path.join(workspace, f"log_{self.name}.txt")
            self.log_ptr = open(self.log_path, "a+")

            self.ckpt_path = os.path.join(self.workspace, 'checkpoints')
            self.best_path = f"{self.ckpt_path}/{self.name}.pth"
            os.makedirs(self.ckpt_path, exist_ok=True)
            self.sam_cache_path = os.path.join(self.workspace, 'sam_cache')
            if self.opt.with_sam and self.opt.feature_container == 'cache':
                os.makedirs(self.sam_cache_path, exist_ok=True)
            

        self.log(
            f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | {"fp16" if self.fp16 else "fp32"} | {self.workspace}')
        self.log(
            f'[INFO] #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')

        self.log(opt)
        self.log(self.model)

        if self.workspace is not None:

            if self.use_checkpoint == "scratch":
                self.log("[INFO] Training from scratch ...")
            elif self.use_checkpoint == "latest":
                self.log("[INFO] Loading latest checkpoint ...")
                self.load_checkpoint()
            elif self.use_checkpoint == "latest_model":
                self.log("[INFO] Loading latest checkpoint (model only)...")
                self.load_checkpoint(model_only=True)
            elif self.use_checkpoint == "best":
                if os.path.exists(self.best_path):
                    self.log("[INFO] Loading best checkpoint ...")
                    self.load_checkpoint(self.best_path)
                else:
                    self.log(
                        f"[INFO] {self.best_path} not found, loading latest ...")
                    self.load_checkpoint()
            else:  # path to ckpt
                self.log(f"[INFO] Loading {self.use_checkpoint} ...")
                self.load_checkpoint(self.use_checkpoint)

    def __del__(self):
        if self.log_ptr:
            self.log_ptr.close()

    def log(self, *args, **kwargs):
        if self.local_rank == 0:
            if not self.mute:
                # print(*args)
                self.console.print(*args, **kwargs)
            if self.log_ptr:
                print(*args, file=self.log_ptr)
                self.log_ptr.flush()  # write immediately to file


    def save_trajectory(self):
        save_dict = {}
        os.makedirs(self.opt.trajectory_root, exist_ok=True)
        trajectories = [self.trajectories[i].cpu().numpy().tolist() for i in range(len(self.trajectories))]
        save_dict['trajectory'] = trajectories
        save_dict['fx'], save_dict['fy'], save_dict['cx'], save_dict['cy'] = self.intrinsics[0].cpu().numpy().tolist()
        if self.point_3d is not None:
            save_dict['points'] = self.point_3d.cpu().numpy().tolist()
        else:
            save_dict['points'] = None
        camera_trajectory_files = os.listdir(self.opt.trajectory_root)
        camera_trajectory_files.sort()
        if len(camera_trajectory_files) == 0:
            print('No trajectory files found, saving to 0000.json')
            last_idx = 0
        else:
            last_idx = int(camera_trajectory_files[-1].split('.')[0]) + 1
            print(f'Last trajectory file index: {last_idx}')
        with open(f'{self.opt.trajectory_root}/{last_idx:04}.json', 'w') as f:
            json.dump(save_dict, f, indent=4)
        print('Saving trajectories Done!')
        self.trajectories = []
        return

    def save_3d_points(self):
        save_dict = {}
        save_dict['points'] = self.point_3d.cpu().numpy().tolist()
        save_dict['negative_labels'] = []
        for i in range(len(self.input_labels)):
            if self.input_labels[i] == 0: 
                save_dict['negative_labels'].append(i)
        save_dict['valid_threshold'] = int(-1),
        save_dict['crucial_point_index'] = []

        with open(os.path.join(self.opt.workspace, 'sam_points.json'), 'w') as f:
            json.dump(save_dict, f, indent=4)
        print('Saving 3D points Done!')
        return
    def ray_pair_rgb_loss(self, rgb, inst_masks, gt_flattened, incoherent, use_pred_logistics = False):
        '''
        Args:
            local_*** : [num of local samples, local patch size ^2, -1]
            use_pred_logistics
        '''
        # random sample some points
        # weights = torch.ones(rgb.shape[1]).expand(rgb.shape[0], -1).to(rgb.device)
        weights = 1. - incoherent[..., 0].to(torch.float32)
        weights = (weights > 0.8).to(torch.float32)
        invalid_inds =  weights.sum(-1) == 0

        invalid_inds_size = invalid_inds.sum()
        weights[invalid_inds] = torch.ones(rgb.shape[1]).expand(invalid_inds_size, -1).to(rgb.device) 
        num_sample = self.opt.ray_pair_rgb_num_sample
    
        sample_index = torch.multinomial(weights, num_samples=num_sample, replacement=False)
        
        
        col_ids = torch.arange(rgb.shape[0], dtype=torch.int64).to(rgb.device)
        rgb_sample = rgb[col_ids[:, None], sample_index][..., None, :]

        sample_mask = inst_masks[col_ids[:, None], sample_index][..., None, :].detach()
        sample_mask_gt = gt_flattened[col_ids[:, None], sample_index].detach()
        if not use_pred_logistics:
            sample_mask_arg =  torch.argmax(sample_mask, -1)
            sample_mask = torch.zeros_like(sample_mask, device=sample_mask.device)
            sample_mask= sample_mask.scatter_(-1, sample_mask_arg[..., None], 1)


        # gt_masks
        rgb = rgb[:, None, ...]
        inst_masks = inst_masks[:, None, ...]


        # calculate indices that has similar rgb values
        color_similarity_map = torch.norm(rgb-rgb_sample, dim=-1)
        similarity_map = color_similarity_map < self.opt.ray_pair_rgb_threshold

        pred_masks_similarity = F.cosine_similarity(inst_masks, sample_mask, dim=-1)
        pred_masks_similarity = torch.exp(- self.opt.ray_pair_rgb_exp_weight * pred_masks_similarity - self.opt.epsilon)         
        pred_masks_similarity = (similarity_map * pred_masks_similarity).sum(-1)
        mean_weight = similarity_map.sum(-1)
        rgb_loss = (pred_masks_similarity / mean_weight).mean()

        return rgb_loss
    
    def label_regularization(self, depth, pred_masks):
        '''
        depth: [B, N]
        pred_masks: [B, N, num_instances]
        '''
        pred_masks = pred_masks.view(-1, self.opt.patch_size, self.opt.patch_size,
                                     self.opt.n_inst).permute(0, 3, 1, 2).contiguous()  # [B, num_instances, patch_size, patch_size]

        diff_x = pred_masks[:, :, :, 1:] - pred_masks[:, :, :, :-1]
        diff_y = pred_masks[:, :, 1:, :] - pred_masks[:, :, :-1, :]

        # [B, patch_size, patch_size]
        depth = depth.view(-1, self.opt.patch_size, self.opt.patch_size)

        depth_diff_x = depth[:, :, 1:] - depth[:, :, :-1]
        depth_diff_y = depth[:, 1:, :] - depth[:, :-1, :]
        weight_x = torch.exp(-(depth_diff_x * depth_diff_x)
                             ).unsqueeze(1).expand_as(diff_x)
        weight_y = torch.exp(-(depth_diff_y * depth_diff_y)
                             ).unsqueeze(1).expand_as(diff_y)

        diff_x = diff_x * diff_x * weight_x
        diff_y = diff_y * diff_y * weight_y

        smoothness_loss = torch.sum(
            diff_x) / torch.sum(weight_x) + torch.sum(diff_y) / torch.sum(weight_y)

        return smoothness_loss

    def train_step(self, data):

        # use cache instead of novel poses
        use_cache = self.opt.with_sam and \
            self.opt.cache_size > 0 and \
            self.cache.full() and \
            self.global_step % self.opt.cache_interval != 0

        # override data
        if use_cache:
            data = self.cache.get()

        rays_o = data['rays_o']  # [N, 3]
        rays_d = data['rays_d']  # [N, 3]
        index = data['index']  # [1/N]
        # [1/N, 2] or None
        cam_near_far = data['cam_near_far'] if 'cam_near_far' in data else None
        H, W = data['H'], data['W']

        N = rays_o.shape[0]
        if self.opt.background == 'random':
            # [N, 3], pixel-wise random.
            bg_color = torch.rand(N, 3, device=self.device)
        else:  # white / last_sample
            bg_color = 1

        # rgb training
        if not self.opt.with_sam and not self.opt.with_mask:
            images = data['images']  # [N, 3/4]
            C = images.shape[-1]
            if C == 4:
                gt_rgb = images[..., :3] * images[..., 3:] + \
                    bg_color * (1 - images[..., 3:])
            else:
                gt_rgb = images

            update_proposal = (not self.opt.with_sam) and (
                self.global_step <= 3000 or self.global_step % 5 == 0)

            outputs = self.model.render(rays_o, rays_d, staged=False, index=index, bg_color=bg_color,
                                        perturb=True, cam_near_far=cam_near_far, update_proposal=update_proposal, return_feats=0)
            pred_rgb = outputs['image']

            loss = self.criterion(pred_rgb, gt_rgb).mean()

            # extra loss
            if 'proposal_loss' in outputs and self.opt.lambda_proposal > 0:
                loss = loss + self.opt.lambda_proposal * \
                    outputs['proposal_loss']

            if 'distort_loss' in outputs and self.opt.lambda_distort > 0:
                loss = loss + self.opt.lambda_distort * outputs['distort_loss']

            if self.opt.lambda_entropy > 0:
                w = outputs['weights_sum'].clamp(1e-5, 1 - 1e-5)
                entropy = - w * torch.log2(w) - (1 - w) * torch.log2(1 - w)
                loss = loss + self.opt.lambda_entropy * (entropy.mean())

            # adaptive num_rays
            if self.opt.adaptive_num_rays:
                self.opt.num_rays = int(
                    round((self.opt.num_points / outputs['num_points']) * self.opt.num_rays))

            return pred_rgb, gt_rgb, loss

        elif self.opt.with_mask:
            gt_mask = data['masks'].to(torch.long)  # [B, N], N=H*W?
            B, N = gt_mask.shape
            # num_instances = torch.unique(gt_masks).shape[0]

            bg_color = 1
            outputs = self.model.render(rays_o, rays_d, staged=False, index=index, bg_color=bg_color,
                                        perturb=False, cam_near_far=cam_near_far, update_proposal=False, 
                                        return_rgb=0, return_feats=0, return_mask=1)

            # [B, N, num_instances]
            inst_masks = outputs['instance_mask_logits']

            # [B*N, num_instances]
            
            gt_masks_flattened = gt_mask.view(-1)  # [B*N]
            labeled = gt_masks_flattened != -1  # only compute loss for labeled pixels

            inst_masks = torch.softmax(
                inst_masks, dim=-1)  # [B, N, num_instances + k]
            # pred_masks = torch.stack([inst_masks[..., :-1].sum(-1), inst_masks[..., -1]], -1)
            pred_masks = inst_masks
            pred_masks_flattened = pred_masks.view(-1, self.opt.n_inst)
            pred_masks_flattened = torch.clamp(pred_masks_flattened, min=self.opt.epsilon, max=1-self.opt.epsilon)
            global_pred_masks_flattened = pred_masks_flattened[:self.opt.num_rays]
            global_gt_masks_flattened = gt_masks_flattened[:self.opt.num_rays]
            if labeled.sum() > 0:
                loss = -torch.log(torch.gather(global_pred_masks_flattened, -1, global_gt_masks_flattened[..., None]))
            else:
                # no
                loss = torch.tensor(0).to(
                    pred_masks_flattened.dtype).to(self.device)
            
            if self.error_map is not None:
                index = data['index'] # [B]
                inds = data['inds_coarse']# [B]
                global_inst_masks = inst_masks[:self.opt.num_rays]
                if isinstance(index, list):
                    # take out, this is an advanced indexing and the copy is unavoidable.
                    error_map = self.error_map[index] # [B, H * W]

                    sample_mask_gt = torch.zeros_like(global_pred_masks_flattened, device=global_pred_masks_flattened.device)
                    sample_mask_gt= sample_mask_gt.scatter_(-1, global_gt_masks_flattened[..., None], 1)

                    pred_masks_similarity = F.cosine_similarity(global_inst_masks, sample_mask_gt, dim=-1)
                    error = torch.exp(- self.opt.ray_pair_rgb_exp_weight * pred_masks_similarity - self.opt.epsilon) 

                    
                    # ema update
                    ema_error = 0.1 * error_map.gather(1, inds) + 0.9 * error
                    error_map.scatter_(1, inds, ema_error)

                    # put back
                    self.error_map[index] = error_map

                else:
                    sample_mask_gt = torch.zeros_like(global_pred_masks_flattened, device=global_pred_masks_flattened.device)
                    sample_mask_gt= sample_mask_gt.scatter_(-1, global_gt_masks_flattened[..., None], 1)
                    
                    pred_masks_similarity = F.cosine_similarity(global_inst_masks, sample_mask_gt, dim=-1)
                    error = torch.exp(- self.opt.ray_pair_rgb_exp_weight * pred_masks_similarity - self.opt.epsilon) 

                    ema_error = 0.1 * self.error_map[index, inds] + 0.9 * error
                    self.error_map[index, inds] = ema_error
                
                
                

                
                # np.save(f'./debug/error_{self.epoch}.npy', self.error_map.detach().cpu().numpy())
                # put back
                # self.error_map[index] = error_map
            loss = loss.mean()
                
            if self.opt.label_regularization_weight > 0:
                loss = loss + self.label_regularization(
                    outputs['depth'].detach(), pred_masks) * self.opt.label_regularization_weight
                
                
            if self.opt.ray_pair_rgb_loss_weight > 0 and self.global_step > self.opt.ray_pair_rgb_iter:
                if self.opt.mixed_sampling:
                    local_inst_masks = inst_masks[self.opt.num_rays:]
                    # data['']
                    local_inst_masks = local_inst_masks.view(self.opt.num_local_sample, self.opt.local_sample_patch_size*self.opt.local_sample_patch_size, -1)
                    
                    local_rgb = outputs['image'][self.opt.num_rays:]
                    local_rgb = local_rgb.view(self.opt.num_local_sample, self.opt.local_sample_patch_size*self.opt.local_sample_patch_size, -1)
                    local_gt_flattened = gt_masks_flattened[self.opt.num_rays:]
                    local_gt_flattened = local_gt_flattened.view(self.opt.num_local_sample, self.opt.local_sample_patch_size*self.opt.local_sample_patch_size, -1)

                    local_error_maps = data['error_maps'][self.opt.num_rays:]
                    local_error_maps = local_error_maps.view(self.opt.num_local_sample, self.opt.local_sample_patch_size*self.opt.local_sample_patch_size, -1)

                    loss = loss + self.ray_pair_rgb_loss(local_rgb, local_inst_masks, local_gt_flattened, local_error_maps, use_pred_logistics = self.opt.ray_pair_rgb_use_pred_logistics) \
                                                                * self.opt.ray_pair_rgb_loss_weight
                else:
                    loss = loss + self.ray_pair_rgb_loss(outputs['image'].detach()[None, ...], inst_masks[None, ...], 
                                                            gt_masks_flattened[None, ...], use_pred_logistics = self.opt.ray_pair_rgb_use_pred_logistics) \
                                                                * self.opt.ray_pair_rgb_loss_weight
                    
                
            pred_masks = pred_masks.argmax(dim=-1)  # [B, N]
                

            return pred_masks, gt_mask, loss

        elif self.opt.with_sam:
            with torch.no_grad():
                if use_cache:
                    gt_samvit = data['gt_samvit']
                else:
                    # render high-res RGB
                    outputs = self.model.render(rays_o, rays_d, staged=True, index=index, bg_color=bg_color,
                                                perturb=True, cam_near_far=cam_near_far, update_proposal=False, return_feats=0)
                    pred_rgb = outputs['image'].reshape(H, W, 3)

                    # encode SAM ground truth
                    image = (pred_rgb.detach().cpu().numpy()
                                * 255).astype(np.uint8)
                    self.sam_predictor.set_image(image)
                    # [1, 256, 64, 64]
                    
                    gt_samvit = self.sam_predictor.features
                    if self.opt.sam_type == 'sam_hq':
                        gt_interm_features = self.sam_predictor.interm_features

                    # write to cache
                    if self.opt.cache_size > 0:
                        data['gt_samvit'] = gt_samvit
                        self.cache.insert(data)

            # always use 64x64 features as SAM default to 1024x1024
            h, w = data['h'], data['w']
            rays_o_hw = data['rays_o_lr']
            rays_d_hw = data['rays_d_lr']
            outputs = self.model.render(rays_o_hw, rays_d_hw, staged=False, index=index, bg_color=bg_color,
                                        perturb=False, cam_near_far=cam_near_far, return_feats=1, H=h, W=w)
            output_dim = self.sam_output_dim

            pred_samvit = outputs['samvit'].reshape(
                1, h, w, output_dim).permute(0, 3, 1, 2).contiguous()
            # print(gt_samvit.shape)
            # print(pred_samvit.shape)
            # exit()
            pred_samvit = F.interpolate(
                pred_samvit, gt_samvit.shape[2:], mode='bilinear')

            # loss
            
            loss = self.criterion(pred_samvit, gt_samvit).mean()


            if self.opt.sam_type == 'sam_hq':
                gt_samvit = [gt_samvit, gt_interm_features]
            return pred_samvit, gt_samvit, loss
        else:
            raise NotImplementedError("Not implemented other types.")
    def post_train_step(self):
        # unscale grad before modifying it!
        # ref: https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-clipping
        self.scaler.unscale_(self.optimizer)

        # the new inplace TV loss
        if self.opt.lambda_tv > 0:
            self.model.apply_total_variation(self.opt.lambda_tv)

        if self.opt.lambda_wd > 0:
            self.model.apply_weight_decay(self.opt.lambda_wd)

    def eval_step(self, data):
        rays_o = data['rays_o']  # [N, 3]
        rays_d = data['rays_d']  # [N, 3]
        index = data['index']  # [1/N]
        # [1/N, 2] or None
        cam_near_far = data['cam_near_far'] if 'cam_near_far' in data else None
        H, W = data['H'], data['W']
        bg_color = 1

        # full resolution RGBD query, do not query feats!
        outputs = self.model.render(rays_o, rays_d, staged=True, index=index, bg_color=bg_color, perturb=False,
                                    cam_near_far=cam_near_far, return_feats=0, return_mask=self.opt.with_mask)

        pred_rgb = outputs['image'].reshape(H, W, 3)
        pred_depth = outputs['depth'].reshape(H, W)

        if not self.opt.with_sam and not self.opt.with_mask:
            images = data['images']  # [H, W, 3/4]
            C = images.shape[-1]
            if C == 4:
                gt_rgb = images[..., :3] * images[..., 3:] + \
                    bg_color * (1 - images[..., 3:])
            else:
                gt_rgb = images

            loss = self.criterion(pred_rgb, gt_rgb).mean()

            return pred_rgb, pred_depth, None, gt_rgb, loss

        elif self.opt.with_mask and not self.opt.with_sam:
                gt_mask = data['masks'].to(torch.long)

                gt_mask_flattened = gt_mask.view(-1)  # [B*H*W]

                labeled = gt_mask_flattened != -1  # only compute loss for labeled pixels
                
                inst_mask = outputs['instance_mask_logits'].reshape(H, W, self.opt.n_inst)
            
                if self.opt.n_inst > 1:
                    inst_mask = torch.softmax(inst_mask, dim=-1)                
                    # pred_mask = torch.stack([inst_mask[..., :-1].sum(-1), inst_mask[..., -1]], -1)
                    pred_mask = inst_mask
                else:
                    pred_mask = torch.sigmoid(inst_mask)

                pred_mask_flattened = pred_mask.view(-1, self.opt.n_inst)
 
                pred_mask_flattened = torch.clamp(pred_mask_flattened, min=self.opt.epsilon, max=1-self.opt.epsilon)

                
                if not data['use_default_intrinsics']:
                    if labeled.sum() > 0:
                        loss = -torch.log(torch.gather(pred_mask_flattened[labeled], -1, gt_mask_flattened[labeled][..., None]))
                    else:
                        loss = torch.tensor(0).to(
                            pred_mask_flattened.dtype).to(self.device)
                    
                    loss = loss.mean()

                    if self.opt.label_regularization_weight > 0:
                        loss = loss + self.label_regularization(
                            outputs['depth'].detach(), pred_mask) * self.opt.label_regularization_weight
                else:
                    loss = torch.tensor(0.)

                # [B, H, W, num_instances]
                # if self.opt.n_inst > 1: 
                #     pred_mask = torch.softmax(pred_mask, dim=-1)
                # else:
                #     pred_masks = torch.sigmoid(pred_masks)
                # pred_mask = pred_mask.argmax(dim=-1) # [B, H, W]

                # pred_seg = overlay_mask(pred_rgb, pred_mask)

                # gt_seg = overlay_mask(pred_rgb, gt_mask)

                return pred_rgb, pred_depth, pred_mask, gt_mask, loss

        elif self.opt.with_sam and not self.opt.with_mask:
            # encode SAM ground truth
            image = (pred_rgb.detach().cpu().numpy()
                    * 255).astype(np.uint8)
            self.sam_predictor.set_image(image)
            gt_samvit = self.sam_predictor.features
            # gt_interm_samvit = self.sam_predictor.interm_features
            
            # always use 64x64 features as SAM default to 1024x1024
            h, w = data['h'], data['w']
            rays_o_hw = data['rays_o_lr']
            rays_d_hw = data['rays_d_lr']
            outputs = self.model.render(rays_o_hw, rays_d_hw, staged=False, index=index, bg_color=bg_color,
                                        perturb=False, cam_near_far=cam_near_far, return_feats=1, H=h, W=w)

            output_dim = self.sam_output_dim
            pred_samvit = outputs['samvit'].reshape(
                1, h, w, output_dim).permute(0, 3, 1, 2).contiguous()
            pred_samvit = F.interpolate(
                pred_samvit, gt_samvit.shape[2:], mode='bilinear')

            # report feature loss
            loss = self.criterion(pred_samvit, gt_samvit).mean()

            # TODO: grid point samples to evaluate IoU...
            if self.opt.use_point:
                masks, _, point_coords, low_res_masks = self.sam_predict(
                    H, W, pred_samvit)

                pred_seg = overlay_mask(pred_rgb, masks[0])
                pred_seg = overlay_point(pred_seg, point_coords)

                # gt_masks, point_coords, low_res_masks = self.sam_predict(H, W, gt_samvit, point_coords, image=(pred_rgb.detach().cpu().numpy() * 255).astype(np.uint8))
                gt_masks, point_coords, low_res_masks = self.sam_predict(
                    H, W, gt_samvit, point_coords)  # use gt feature to debug

                gt_seg = overlay_mask(pred_rgb, gt_masks[0])
                gt_seg = overlay_point(gt_seg, point_coords)
                return pred_seg, pred_depth, pred_samvit, gt_seg, loss
            else:
                return pred_rgb, pred_depth, pred_samvit, gt_samvit, loss

        else:
            raise NotImplementedError("Not implemented other types.")
    def test_step(self, data, bg_color=None, perturb=False, point_coords=None, point_labels=None):
    
        rays_o = data['rays_o']  # [N, 3]
        rays_d = data['rays_d']  # [N, 3]
        index = data['index']  # [1/N]
        H, W = data['H'], data['W']

        # [1/N, 2] or None
        cam_near_far = data['cam_near_far'] if 'cam_near_far' in data else None

        if bg_color is not None:
            bg_color = bg_color.to(self.device)
        # full resolution RGBD query, do not query feats!
        
        outputs = self.model.render(rays_o, rays_d, staged=True, index=index, bg_color=bg_color,
                                    perturb=perturb, cam_near_far=cam_near_far, return_feats=False, return_mask=self.opt.with_mask)

        pred_rgb = outputs['image'].reshape(H, W, 3)
        pred_depth = outputs['depth'].reshape(H, W)
        if self.opt.render_mesh:
            predicted_mesh = outputs['mesh_image'].reshape(H, W, 3)
            pred_rgb = predicted_mesh
            

        if self.opt.val_save_root is not None:
            os.makedirs(self.opt.val_save_root, exist_ok=True)
            img_name = data["index"][0]
            if data['img_names'] is not None:
                img_name = data['img_names'][0]
            if self.opt.render_mesh:
                save_mesh_rgb = predicted_mesh.detach().cpu().numpy()
                save_mesh_rgb = (save_mesh_rgb * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(self.opt.val_save_root, f'{img_name}_mesh_rgb.png'), cv2.cvtColor(save_mesh_rgb, cv2.COLOR_RGB2BGR))
            else:
                save_pred_rgb = pred_rgb.detach().cpu().numpy()
                save_pred_rgb = (save_pred_rgb * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(self.opt.val_save_root, f'{img_name}_rgb.png'), cv2.cvtColor(save_pred_rgb, cv2.COLOR_RGB2BGR))
           
        if self.opt.with_mask:
            inst_mask = outputs['instance_mask_logits'].reshape(
                H, W, self.opt.n_inst)
            
            if self.opt.n_inst > 1:
                inst_mask = torch.softmax(inst_mask, dim=-1)                
                # pred_mask = torch.stack([inst_mask[..., :-1].sum(-1), inst_mask[..., -1]], -1)
                pred_mask = inst_mask
            else:
                pred_mask = torch.sigmoid(inst_mask)

            if self.opt.render_mask_type == 'heatmap':
                if self.opt.render_mask_instance_id >= 0 and self.opt.render_mask_instance_id < self.opt.n_inst:
                    instance_mask = pred_mask[...,
                                              self.opt.render_mask_instance_id]
                    instance_id = self.opt.render_mask_instance_id
                else:
                    instance_mask, _ = torch.max(pred_mask, -1)
                    instance_id = pred_mask.argmax(-1)

                pred_rgb = overlay_mask_heatmap(
                    instance_mask, instance_id, color_map=self.color_map)
            elif self.opt.render_mask_type == 'composition':
                instance_mask, _ = torch.max(pred_mask, -1)
                instance_id = pred_mask.argmax(-1)

                if self.opt.render_mask_instance_id >= 0 and self.opt.render_mask_instance_id < self.opt.n_inst:
                    render_id = self.opt.render_mask_instance_id
                else:
                    render_id = -1

                pred_rgb = overlay_mask_composition(
                    pred_rgb, instance_id, color_map=self.color_map, render_id=render_id)
            elif self.opt.render_mask_type == 'mask':
                instance_mask, _ = torch.max(pred_mask, -1)
                instance_id = pred_mask.argmax(-1)

                if self.opt.render_mask_instance_id >= 0 and self.opt.render_mask_instance_id < self.opt.n_inst:
                    render_id = self.opt.render_mask_instance_id
                else:
                    render_id = -1
                

                pred_mask = instance_id == render_id
                pred_mask = pred_mask.to(torch.float32)
                # print(pred_mask.shape)
          
                pred_rgb = pred_rgb * pred_mask[..., None] + (1- pred_mask[..., None]) * bg_color
                
        if self.opt.val_save_root is not None:
            save_pred_rgb = pred_rgb.detach().cpu().numpy()
            save_pred_rgb = (save_pred_rgb * 255).astype(np.uint8)
            img_name = data["index"][0]
            if data['img_names'] is not None:
                img_name = data['img_names'][0]
            np.save(os.path.join(self.opt.val_save_root, f'{img_name}_mask.npy'), pred_mask.detach().cpu().numpy())
            cv2.imwrite(os.path.join(self.opt.val_save_root, f'{img_name}_mask_vis.png'), cv2.cvtColor(save_pred_rgb, cv2.COLOR_RGB2BGR))

                    
        if self.opt.with_sam:
            h, w = data['h'], data['w']
            rays_o_hw = data['rays_o_lr']
            rays_d_hw = data['rays_d_lr']
            outputs = self.model.render(rays_o_hw, rays_d_hw, staged=False, index=index, bg_color=bg_color,
                                        perturb=False, cam_near_far=cam_near_far, return_feats=1, H=h, W=w)
            
            output_dim = self.sam_output_dim

            pred_samvit = outputs['samvit'].reshape(
                1, h, w, output_dim).permute(0, 3, 1, 2).contiguous()

            # remember new point_3d
        if point_coords is not None:
            rays_o = rays_o.view(H, W, 3)
            rays_d = rays_d.view(H, W, 3)
            point_depth = pred_depth[point_coords[:,1], point_coords[:, 0]]
            point_rays_o = rays_o[point_coords[:, 1], point_coords[:, 0]]
            point_rays_d = rays_d[point_coords[:, 1], point_coords[:, 0]]
            point_3d = point_rays_o + point_rays_d * \
                point_depth.unsqueeze(-1)  # [1, 3]
            point_labels = torch.from_numpy(point_labels).to(self.device)
            # update current selected points
            if self.point_3d is None:
                self.point_3d = point_3d
                self.input_labels = point_labels
            else:
                dist = (self.point_3d - point_3d).norm(dim=-1)
                dist_thresh = 0.01
                if dist.min() > dist_thresh:
                    # add if not close to any existed point
                    # print(f'[INFO] add new point {point_3d}')
                    self.point_3d = torch.cat(
                        [self.point_3d, point_3d], dim=0)
                    self.input_labels = torch.cat(
                        [self.input_labels, point_labels], dim=0)
                else:
                    # remove existed point if too close
                    # print(f'[INFO] remove old point mask {dist <= dist_thresh}')
                    keep_mask = dist > dist_thresh
                    if keep_mask.any():
                        self.point_3d = self.point_3d[keep_mask]
                        self.input_labels = self.input_labels[keep_mask]
                    else:
                        self.point_3d = None
                        self.input_labels = None

        # get remembered points coords first
        inputs_point_coords = None
        if self.point_3d is not None:
            point_3d = torch.cat([self.point_3d, torch.ones_like(
                self.point_3d[:, :1])], axis=-1)  # [N, 4]
            inputs_point_labels = self.input_labels
            w2c = torch.inverse(data['poses'][0])  # [4, 4]
            point_3d_cam = point_3d @ w2c.T  # [N, 4]
            intrinsics = data['intrinsics'][0]  # [4]
            fx, fy, cx, cy = intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3]
            inputs_point_coords = torch.stack([
                W - (fx * point_3d_cam[:, 0] / point_3d_cam[:, 2] + cx),
                fy * point_3d_cam[:, 1] / point_3d_cam[:, 2] + cy,
            ], dim=-1).long()  # [N, 2]
            # mask out-of-screen coords
            screen_mask = (inputs_point_coords[:, 0] >= 0) & (inputs_point_coords[:, 0] < W) & (
                inputs_point_coords[:, 1] >= 0) & (inputs_point_coords[:, 1] < H)
            if screen_mask.any():
                inputs_point_coords = inputs_point_coords[screen_mask]
                inputs_point_labels = inputs_point_labels[screen_mask]
                # depth test to reject those occluded point_coords
                point_depth = - point_3d_cam[screen_mask, 2]
                observed_depth = pred_depth[inputs_point_coords[:,
                                                                1], inputs_point_coords[:, 0]]
                unoccluded_mask = (
                    point_depth - observed_depth).abs() <= 0.05
                if unoccluded_mask.any():
                    inputs_point_coords = inputs_point_coords[unoccluded_mask].detach().cpu().numpy()
                    inputs_point_labels = inputs_point_labels[unoccluded_mask].detach().cpu().numpy()
                else:
                    inputs_point_coords = None
            else:
                inputs_point_coords = None

        if inputs_point_coords is not None:
            resize_ratio = 1024 / W if W > H else 1024 / H
            point_coords = (inputs_point_coords.astype(np.float32)
                            * resize_ratio).astype(np.int32)
            
            original_point_coords = (point_coords / resize_ratio).astype(np.int32)
            
            if self.opt.with_sam:
                masks, _, outputs_point_coords, low_res_masks = self.sam_predict(
                H, W, pred_samvit, inputs_point_coords, inputs_point_labels)

                pred_rgb = overlay_mask(pred_rgb, masks[0])
                

            pred_rgb = overlay_point(pred_rgb, original_point_coords, inputs_point_labels = inputs_point_labels)
   
        if self.opt.return_extra:
            if self.opt.with_sam:
                return pred_rgb, pred_depth, pred_samvit
            elif self.opt.with_mask:
                return pred_rgb, pred_depth, pred_mask
        else:
            return pred_rgb, pred_depth


    def decode_step(self, data, bg_color=None, perturb=False, point_coords=None):
        rays_o = data['rays_o']  # [N, 3]
        rays_d = data['rays_d']  # [N, 3]
        index = data['index']  # [1/N]
        H, W = data['H'], data['W']

        cam_near_far = data['cam_near_far'] if 'cam_near_far' in data else None

        if bg_color is not None:
            bg_color = bg_color.to(self.device)
        # full resolution RGBD query, do not query feats!
        
        outputs = self.model.render(rays_o, rays_d, staged=True, index=index, bg_color=bg_color,
                                    perturb=perturb, cam_near_far=cam_near_far, return_feats=False, return_mask=self.opt.with_mask)

        pred_rgb = outputs['image'].reshape(H, W, 3)
        pred_depth = outputs['depth'].reshape(H, W)
        if self.opt.render_mesh:
            predicted_mesh = outputs['mesh_image'].reshape(H, W, 3)
            pred_rgb = predicted_mesh

        h, w = data['h'], data['w']
        rays_o_hw = data['rays_o_lr']
        rays_d_hw = data['rays_d_lr']
        outputs = self.model.render(rays_o_hw, rays_d_hw, staged=False, index=index, bg_color=bg_color,
                                    perturb=False, cam_near_far=cam_near_far, return_feats=1, H=h, W=w)
        
        output_dim = self.sam_output_dim
        # sam feature
        if self.opt.feature_container == 'cache':
            pred_samvit = np.load(os.path.join(self.opt.workspace,'sam_cache', f'{data["img_names"][0]}.npy'))
            pred_samvit = torch.from_numpy(pred_samvit).to(self.device).unsqueeze(0)
        elif self.opt.feature_container == 'distill':
            pred_samvit = outputs['samvit'].reshape(
                1, h, w, output_dim).permute(0, 3, 1, 2).contiguous()

        # point prompts
        point_3d = torch.cat([self.point_3d, torch.ones_like(
            self.point_3d[:, :1])], axis=-1)  # [N, 4]
        inputs_point_labels = self.input_labels
        inputs_crucial_point_label = self.crucial_point_label        

        w2c = torch.inverse(data['poses'][0])  # [4, 4]
        point_3d_cam = point_3d @ w2c.T  # [N, 4]
        intrinsics = data['intrinsics'][0]  # [4]
        fx, fy, cx, cy = intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3]
        inputs_point_coords = torch.stack([
            W - (fx * point_3d_cam[:, 0] / point_3d_cam[:, 2] + cx),
            fy * point_3d_cam[:, 1] / point_3d_cam[:, 2] + cy,
        ], dim=-1).long()  # [N, 2]
        # mask out-of-screen coords
        screen_mask = (inputs_point_coords[:, 0] >= 0) & (inputs_point_coords[:, 0] < W) & (
            inputs_point_coords[:, 1] >= 0) & (inputs_point_coords[:, 1] < H)
        if screen_mask.any():
            inputs_point_coords = inputs_point_coords[screen_mask]
            inputs_point_labels = inputs_point_labels[screen_mask]
            inputs_crucial_point_label = inputs_crucial_point_label[screen_mask]
            
            # depth test to reject those occluded point_coords
            point_depth = - point_3d_cam[screen_mask, 2]
            observed_depth = pred_depth[inputs_point_coords[:,
                                                            1], inputs_point_coords[:, 0]]
            unoccluded_mask = (
                point_depth - observed_depth).abs() <= 0.05
            if unoccluded_mask.any():
                inputs_point_coords = inputs_point_coords[unoccluded_mask].detach().cpu().numpy()
                inputs_point_labels = inputs_point_labels[unoccluded_mask].detach().cpu().numpy()
                inputs_crucial_point_label = inputs_crucial_point_label[unoccluded_mask]
            else:
                inputs_point_coords = None
        else:
            inputs_point_coords = None

        
        is_valid = (inputs_point_coords is not None) and \
                (inputs_crucial_point_label.sum() >= self.crucial_point_count) and \
                (inputs_crucial_point_label.size(0) >= self.valid_threshold)
        if inputs_point_coords is not None:
            resize_ratio = 1024 / W if W > H else 1024 / H
            point_coords = (inputs_point_coords.astype(np.float32)
                            * resize_ratio).astype(np.int32)
            original_point_coords = (point_coords / resize_ratio).astype(np.int32)
            masks, scores, outputs_point_coords, low_res_masks = self.sam_predict(H, W, pred_samvit, inputs_point_coords, inputs_point_labels = inputs_point_labels)

            max_score = 0
            index = 0
            for j, (mask, score) in enumerate(zip(masks, scores)):
                if score > max_score:
                    max_score = score
                    index = j
            
            pred_masks = masks[index:index + 1]
            
            pred_rgb = overlay_mask(pred_rgb, pred_masks[0])
            pred_rgb = overlay_point(pred_rgb, original_point_coords, inputs_point_labels = inputs_point_labels)
        else:    
            pred_masks = torch.zeros_like(pred_rgb)
   
        
        return pred_rgb, pred_depth, pred_masks, is_valid

    def sam_predict(self, H, W, features, point_coords=None, mask_input=None, image=None, inputs_point_labels = None):
        # H/W: original image size
        # features: [1, 256, h, w]
        # point_coords: [N, 2] np.ndarray, int32
        # image: np.ndarray [H, W, 3], uint8, debug use, if provided, override with GT feature

        resize_ratio = 1024 / W if W > H else 1024 / H
        input_size = (int(H * resize_ratio), int(W * resize_ratio))

        if image is not None:
            self.sam_predictor.set_image(image)
        else:
            # mimic set_image
            self.sam_predictor.reset_image()
            self.sam_predictor.original_size = (H, W)
            self.sam_predictor.input_size = input_size

            h, w = features.shape[2:]
            resize_ratio_feat = 64 / w if w > h else 64 / h
            features = F.interpolate(features, (int(
                h * resize_ratio_feat), int(w * resize_ratio_feat)), mode='bilinear', align_corners=False)
            features = F.pad(
                features, (0, 64 - features.shape[3], 0, 64 - features.shape[2]), mode='constant', value=0)
            self.sam_predictor.features = features
            self.sam_predictor.is_image_set = True

        if point_coords is None:
            # random single point if not provided
            border_h = int(input_size[0] * 0.2)
            border_w = int(input_size[1] * 0.2)
            point_coords = np.array([[
                np.random.randint(0 + border_h, input_size[1] - border_h),
                np.random.randint(0 + border_w, input_size[0] - border_w)
            ]])
        else:
            # scale to input size
            point_coords = (point_coords.astype(np.float32)
                            * resize_ratio).astype(np.int32)

        # use last mask as a prior if provided
        # NOTE: seems not useful, still need the point inputs...
        if mask_input is not None:
            mask_input_torch = torch.as_tensor(
                mask_input, dtype=torch.float, device=self.device)
            mask_input_torch = mask_input_torch[None, :, :, :]
        else:
            mask_input_torch = None

        
        coords_torch = torch.as_tensor(
            point_coords, dtype=torch.float, device=self.device)
        if inputs_point_labels is None:
            point_labels = np.ones_like(point_coords[:, 0])  # [N]
            labels_torch = torch.as_tensor(
                point_labels, dtype=torch.int, device=self.device)
        else:
            labels_torch = torch.tensor(inputs_point_labels,device=self.device)
        coords_torch, labels_torch = coords_torch[None,
                                                  :, :], labels_torch[None, :]

        # decode 
        self.sam_predictor.interm_features = None
        masks, iou_predictions, low_res_masks = self.sam_predictor.predict_torch(
            coords_torch, labels_torch,
            mask_input=mask_input_torch,
            multimask_output=True,
        )

        original_point_coords = (point_coords / resize_ratio).astype(np.int32)
        # [N, H, W], [N], [N, 2], [N, 256, 256]
        return masks[0], iou_predictions[0], original_point_coords, low_res_masks

    # ------------------------------
    def store_sam_feautres(self, loader):
        self.log(f"[INFO] store SAM features to {self.sam_cache_path}")
        for data in tqdm.tqdm(loader):
            with torch.cuda.amp.autocast(enabled=self.opt.fp16):
                preds, preds_depth, preds_extra, truths, loss = self.eval_step(data)
            for ind in range(len(data['img_names'])):
                img_name = data['img_names'][ind]
                save_feature_path = os.path.join(self.sam_cache_path, img_name + '.npy')
                np.save(save_feature_path, truths[ind].detach().cpu().numpy())
                
        return 
        
        
    def train(self, train_loader, valid_loader, max_epochs):
        start_t = time.time()
        # get a ref to error_map
        
        self.error_map = train_loader._data.error_map if self.opt.error_map else None
        if self.opt.feature_container == 'cache' and self.opt.with_sam:
            self.store_sam_feautres(valid_loader)
        else:
            for epoch in range(self.epoch + 1, max_epochs + 1):
                self.epoch = epoch
                train_loader.epoch = epoch
                self.train_one_epoch(train_loader)
                self.save_interval = 1

                if (self.epoch % self.save_interval == 0 or self.epoch == max_epochs) and self.workspace is not None and self.local_rank == 0:
                    self.save_checkpoint(full=True, best=False, remove_old=True)

                if self.epoch % self.eval_interval == 0:
                    self.evaluate_one_epoch(valid_loader)
                    self.save_checkpoint(full=False, best=True)

        end_t = time.time()
        self.log(f"[INFO] training takes {(end_t - start_t)/ 60:.6f} minutes.")


    def evaluate(self, loader, name=None):
        self.evaluate_one_epoch(loader, name)


    def test(self, loader, save_path=None, name=None, write_video=True):

        if save_path is None:
            save_path = os.path.join(self.workspace, 'results')

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        os.makedirs(save_path, exist_ok=True)

        self.log(f"==> Start Test, save results to {save_path}")

        pbar = tqdm.tqdm(total=len(loader) * loader.batch_size,
                         bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        self.model.eval()

        if write_video:
            all_preds = []
            all_preds_depth = []

        with torch.no_grad():
            pose_dict = {}
            for i, data in enumerate(loader):

                with torch.cuda.amp.autocast(enabled=self.opt.fp16):
                    if self.opt.return_extra:
                        preds, preds_depth, pred_extra = self.test_step(data)
                    else:
                        preds, preds_depth = self.test_step(data)
                pred = preds.detach().cpu().numpy()
                # print(pred.max(), pred.min())
                # exit()
                pred = (pred * 255).astype(np.uint8)

                pred_depth = preds_depth.detach().cpu().numpy()
                # pred_depth = (pred_depth - pred_depth.min()) / \
                #     (pred_depth.max() - pred_depth.min() + 1e-6)
                # pred_depth = (pred_depth * 255).astype(np.uint8)
                file_name = data['img_names'][0] if data['img_names'] is not None else f'{name}_{i:04d}'
                if self.opt.val_save_root is None:
                    
                    if write_video:
                        all_preds.append(pred)
                        all_preds_depth.append(pred_depth)
                    else:
                        cv2.imwrite(os.path.join(save_path, f'{file_name}_rgb.png'), cv2.cvtColor(
                            pred, cv2.COLOR_RGB2BGR))
                        np.save(os.path.join(
                            save_path, f'{file_name}_depth.npy'), pred_depth)
                        if self.opt.return_extra:
                            pred_extra = pred_extra.detach().cpu().numpy()
                            ending = 'sam' if self.opt.with_sam else 'mask'
                            np.save(os.path.join(save_path, f'{file_name}_{ending}.npy'), pred_extra)
                            if self.opt.with_mask:
                                instance_id = pred_extra.argmax(-1)
                                masks = overlay_mask_only(instance_id, self.color_map).detach().cpu().numpy()
                                
                                cv2.imwrite(os.path.join(save_path, f'{file_name}_mask.png'), cv2.cvtColor(
                                                            masks, cv2.COLOR_RGB2BGR))
                        pose_dict[f'{file_name}'] = data['poses'][0].detach().cpu().numpy().tolist()
                pbar.update(loader.batch_size)
        if self.opt.val_save_root is None:
            with open(os.path.join(save_path, 'pose_dir.json'), "w+") as f:
                json.dump(pose_dict, f, indent=4)
            
        if write_video:
            all_preds = np.stack(all_preds, axis=0)  # [N, H, W, 3]
            all_preds_depth = np.stack(all_preds_depth, axis=0)  # [N, H, W]

            # fix ffmpeg not divisible by 2
            all_preds = np.pad(all_preds, ((0, 0), (0, 1 if all_preds.shape[1] % 2 != 0 else 0), (
                0, 1 if all_preds.shape[2] % 2 != 0 else 0), (0, 0)))
            all_preds_depth = np.pad(all_preds_depth, ((
                0, 0), (0, 1 if all_preds_depth.shape[1] % 2 != 0 else 0), (0, 1 if all_preds_depth.shape[2] % 2 != 0 else 0)))

            imageio.mimwrite(os.path.join(
                save_path, f'{name}_rgb.mp4'), all_preds, fps=24, quality=8, macro_block_size=1)
            imageio.mimwrite(os.path.join(
                save_path, f'{name}_depth.mp4'), all_preds_depth, fps=24, quality=8, macro_block_size=1)

        self.log(f"==> Finished Test.")

    # [GUI] just train for 16 steps, without any other overhead that may slow down rendering.
    def train_gui(self, train_loader, step=16):

        self.model.train()

        total_loss = torch.tensor([0], dtype=torch.float32, device=self.device)

        loader = iter(train_loader)

        for _ in range(step):

            # mimic an infinite loop dataloader (in case the total dataset is smaller than step)
            try:
                data = next(loader)
            except StopIteration:
                loader = iter(train_loader)
                data = next(loader)

            self.global_step += 1

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.opt.fp16):
                preds, truths, loss_net = self.train_step(data)

            loss = loss_net
            self.scaler.scale(loss).backward()

            self.post_train_step()  # for TV loss...

            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            total_loss += loss_net.detach()

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss.item() / step

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        outputs = {
            'loss': average_loss,
            'lr': self.optimizer.param_groups[0]['lr'],
        }

        return outputs

    # [GUI] test on a single image

    def test_gui(self, pose, intrinsics, W, H, bg_color=None, spp=1, downscale=1, user_inputs=None):

        # render resolution (may need downscale to for better frame rate)
        rH = int(H * downscale)
        rW = int(W * downscale)
        intrinsics = intrinsics * downscale

        pose = torch.from_numpy(pose).unsqueeze(0).to(self.device)
        # pose = torch.tensor([[[ 4.7486e-01,  2.2481e-01,  8.5087e-01,  2.4316e-01],
        #  [ 8.8006e-01, -1.2130e-01, -4.5910e-01, -1.1018e-01],
        #  [-3.3307e-16,  9.6682e-01, -2.5545e-01, -2.0741e-01],
        #  [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]]]).to(self.device)
        intrinsics = torch.from_numpy(intrinsics).unsqueeze(0).to(self.device)

        rays = get_rays(pose, intrinsics, rH, rW, -1, device=self.device)

        scale = 16 * rH // 1024 if rH > rW else 16 * rW // 1024
        rays_lr = get_rays(pose, intrinsics / scale, rH //
                           scale, rW // scale, -1, device=self.device)

        data = {
            'poses': pose,
            'intrinsics': intrinsics,
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            'rays_o_lr': rays_lr['rays_o'],
            'rays_d_lr': rays_lr['rays_d'],
            'H': rH,
            'W': rW,
            'h': rH // scale,
            'w': rW // scale,
            'index': [0],
        }

        if user_inputs is not None:
            point_coords = user_inputs['point_coords']
            point_labels = user_inputs['point_labels']
        else:
            point_coords = None
            point_labels = None

        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()


        if self.track_trajectory:
            self.intrinsics = intrinsics
            self.trajectories.append(pose)


        with torch.no_grad():
            # here spp is used as perturb random seed! (but not perturb the first sample)
            with torch.cuda.amp.autocast(enabled=self.opt.fp16):
                preds, preds_depth = self.test_step(
                    data, bg_color=bg_color, perturb=False if spp == 1 else spp, point_coords=point_coords, point_labels=point_labels)

        if self.ema is not None:
            self.ema.restore()

        # interpolation to the original resolution
        if downscale != 1:
            preds = F.interpolate(preds.unsqueeze(0).permute(0, 3, 1, 2), size=(
                H, W), mode='nearest').permute(0, 2, 3, 1).squeeze(0).contiguous()
            preds_depth = F.interpolate(preds_depth.unsqueeze(0).unsqueeze(
                1), size=(H, W), mode='nearest').squeeze(0).squeeze(1)

        pred = preds.detach().cpu().numpy()
        pred_depth = preds_depth.detach().cpu().numpy()

        outputs = {
            'image': pred,
            'depth': pred_depth,
        }

        return outputs

    def render_depth(self, data):
        rays_o = data['rays_o']  # [N, 3]
        rays_d = data['rays_d']  # [N, 3]
        index = data['index']  # [1/N]
        # [1/N, 2] or None
        cam_near_far = data['cam_near_far'] if 'cam_near_far' in data else None
        H, W = data['H'], data['W']
        bg_color = 1

        # full resolution RGBD query, do not query feats!
        outputs = self.model.render(rays_o, rays_d, staged=True, index=index, bg_color=bg_color, perturb=False,
                                    cam_near_far=cam_near_far, return_feats=0, return_mask=self.opt.with_mask)

        depth_map = outputs['depth'].reshape(H, W)
            
        return depth_map

    def render_mask(self, data):
        rays_o = data['rays_o']  # [N, 3]
        rays_d = data['rays_d']  # [N, 3]
        index = data['index']  # [1/N]
        # [1/N, 2] or None
        cam_near_far = data['cam_near_far'] if 'cam_near_far' in data else None
        H, W = data['H'], data['W']
        bg_color = 1

        # full resolution RGBD query, do not query feats!
        outputs = self.model.render(rays_o, rays_d, staged=True, index=index, bg_color=bg_color, perturb=False,
                                    cam_near_far=cam_near_far, return_feats=0, return_mask=self.opt.with_mask)

        inst_mask = outputs['instance_mask_logits'].reshape(
                H, W, self.opt.n_inst)
            
        if self.opt.n_inst > 1:
            inst_mask = torch.softmax(inst_mask, dim=-1)          
            pred_mask = inst_mask      
            # pred_mask = torch.stack([inst_mask[..., :-1].sum(-1), inst_mask[..., -1]], -1)
        else:
            pred_mask = torch.sigmoid(inst_mask)
        return pred_mask
    
    
    def update_ground_truth(self, loader, rendered_masks):

        H, W = loader._data.confident_masks.shape[1:3]
        rendered_masks = rendered_masks[:, None, ...]
        rendered_masks = F.interpolate(rendered_masks, (H, W), mode='bilinear')

       
        # loader._data.confident_masks = loader._data.confident_masks * 0.7 + rendered_masks.permute(0,2,3,1) * 0.3
        # loader._data.masks = (loader._data.confident_masks >= 0.4).to(torch.float32)
        
        confident_map = loader._data.confident_masks * 0.3 + rendered_masks.permute(0,2,3,1) * 0.7
        loader._data.masks = (confident_map >= 0.4).to(torch.float32)
        # np.save('debug/render.npy', rendered_masks.detach().cpu().numpy())
        # np.save('debug/masks_{}.npy', loader._data.masks.detach().cpu().numpy())
        

    def update_depth(self, loader):
        
        self.model.eval()
        rendered_depth_list = []
        with torch.no_grad():
            for index  in range(len(loader._data.poses)):
                data = loader._data.collate_depth(index)
                depth = self.render_depth(data)
                rendered_depth_list.append(depth)
                
        rendered_depths = torch.stack(rendered_depth_list, 0).detach()

        
        np.save(f'debug/depth.npy', rendered_depths[0].detach().cpu().numpy())
        loader._data.depth = rendered_depths
        self.model.train()
    
    
    def update_error_map(self, loader):
        self.model.eval()
        rendered_mask_list = []
        with torch.no_grad():
            for index  in range(len(loader._data.poses)):
                data = loader._data.collate_depth(index)
                mask = self.render_mask(data)
                rendered_mask_list.append(mask)
        rendered_masks_softmax = torch.stack(rendered_mask_list, 0).detach()
        batch_size = rendered_masks_softmax.shape[0]
        rendered_masks = rendered_masks_softmax.reshape(batch_size, -1, rendered_masks_softmax.shape[-1])
        
        gt_masks = loader._data.masks.clone().to(torch.float32)[..., 0]
        gt_masks = gt_masks[:, None, ...]
        
        mask_small = F.interpolate(gt_masks, (self.opt.error_map_size, self.opt.error_map_size), mode='bilinear')
        mask_small = mask_small.round().to(torch.int64)
        
        gt_masks_flatten = mask_small.reshape(batch_size, -1)
        
        gt_masks_vectors = torch.zeros_like(rendered_masks, device=rendered_masks.device)
        # print(gt_masks_flatten.dtype)
        gt_masks_vectors= gt_masks_vectors.scatter_(-1, gt_masks_flatten[..., None], 1)
                        
                        
        pred_masks_similarity = F.cosine_similarity(gt_masks_vectors, rendered_masks, dim=-1)
        error_map = torch.exp(- self.opt.ray_pair_rgb_exp_weight * pred_masks_similarity - self.opt.epsilon) 
        # np.save('debug/errors.npy', error_map[0].detach().cpu().numpy())
        loader._data.error_map = error_map
        self.model.train()

    def generate_depth(self, loader):
        
        self.model.eval()
        rendered_mask_list = []
        with torch.no_grad():
            for index  in range(len(loader._data.poses)):
                data = loader._data.collate_mask(index)
                mask = self.render_mask(data)
                rendered_mask_list.append(mask)
        rendered_masks_softmax = torch.stack(rendered_mask_list, 0).detach()
        rendered_masks = rendered_masks_softmax.argmax(-1)[:, None, ...]

        self.model.train()


    def train_one_epoch(self, loader):
        
        self.log(
            f"==> Start Training Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']:.6f} ...")

        total_loss = 0
        if self.local_rank == 0 and self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()

        self.model.train()

        # distributedSampler: must call set_epoch() to shuffle indices across multiple epochs
        # ref: https://pytorch.org/docs/stable/data.html
        if self.world_size > 1:
            loader.sampler.set_epoch(self.epoch)
        loader._data.epoch = self.epoch
        loader._data.global_step = self.global_step + 1
        
        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size,
                             bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        self.local_step = 0


        for data in loader:
            self.local_step += 1
            self.global_step += 1
            loader._data.global_step += 1
            self.optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=self.opt.fp16):
                preds, truths, loss_net = self.train_step(data)

            if (self.global_step + 1) % self.opt.ray_pair_rgb_iter == 0 and self.global_step != 0 and self.opt.error_map:
                print("Update error map and start to use RGB loss...")
                self.update_error_map(loader)

            
            loss = loss_net
            self.scaler.scale(loss).backward()

            self.post_train_step()  # for TV loss...

            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            
            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            loss_val = loss_net.item()
            total_loss += loss_val

            if self.local_rank == 0:
                if self.report_metric_at_train:
                    for metric in self.metrics:
                        metric.update(preds, truths)

                if self.scheduler_update_every_step:
                    pbar.set_description(
                        f"loss={loss_val:.6f} ({total_loss/self.local_step:.6f}), lr={self.optimizer.param_groups[0]['lr']:.6f}")
                else:
                    pbar.set_description(
                        f"loss={loss_val:.6f} ({total_loss/self.local_step:.6f})")
                pbar.update(loader.batch_size)
                
            

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if self.report_metric_at_train:
                for metric in self.metrics:
                    self.log(metric.report(), style="red")
                    metric.clear()

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        self.log(f"==> Finished Epoch {self.epoch}, loss={average_loss:.6f}.")
        

        
    def evaluate_one_epoch(self, loader, name=None):
        self.log(f"++> Evaluate at epoch {self.epoch} ...")

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        total_loss = 0
        if self.local_rank == 0:
            for metric in self.metrics:
                metric.clear()

        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size,
                             bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        with torch.no_grad():
            self.local_step = 0

            for data in loader:
                self.local_step += 1

                image_name = data['img_names'][0]

                with torch.cuda.amp.autocast(enabled=self.opt.fp16):
                    preds, preds_depth, preds_extra, truths, loss = self.eval_step(data)

                # all_gather/reduce the statistics (NCCL only support all_*)
                if self.world_size > 1:
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    loss = loss / self.world_size

                    preds_list = [torch.zeros_like(preds).to(self.device) for _ in range(
                        self.world_size)]  # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_list, preds)
                    preds = torch.cat(preds_list, dim=0)

                    preds_depth_list = [torch.zeros_like(preds_depth).to(
                        self.device) for _ in range(self.world_size)]  # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_depth_list, preds_depth)
                    preds_depth = torch.cat(preds_depth_list, dim=0)

                    if self.opt.with_sam or self.opt.with_mask:
                        preds_extra_list = [torch.zeros_like(preds_extra).to(
                            self.device) for _ in range(self.world_size)]  # [[B, ...], [B, ...], ...]
                        dist.all_gather(preds_extra_list, preds_depth)
                        preds_extra = torch.cat(preds_extra_list, dim=0)

                    truths_list = [torch.zeros_like(truths).to(self.device) for _ in range(
                        self.world_size)]  # [[B, ...], [B, ...], ...]
                    dist.all_gather(truths_list, truths)
                    truths = torch.cat(truths_list, dim=0)

                loss_val = loss.item()
                total_loss += loss_val

                if self.local_rank == 0:
                    for metric in self.metrics:
                        if self.opt.with_sam or self.opt.with_mask:
                            metric.update(preds_extra, truths)
                        else:
                            metric.update(preds, truths)

                    # save image
                    save_path = os.path.join(
                        self.workspace, 'validation', f'{name}_{image_name}_rgb.png')
                    save_path_depth = os.path.join(
                        self.workspace, 'validation', f'{name}_{image_name}_depth.npy')

                    save_path_error = None
                    save_path_gt = None
                    if self.opt.with_sam:
                        save_path_extra = os.path.join(
                            self.workspace, 'validation', f'{name}_{image_name}_sam.npy')
                    elif self.opt.with_mask:
                        save_path_extra = os.path.join(
                            self.workspace, 'validation', f'{name}_{image_name}_mask.npy')
                    if not self.opt.with_sam and not self.opt.with_mask:
                        save_path_gt = os.path.join(
                            self.workspace, 'validation', f'{name}_{image_name}_gt.png')
                    # self.log(f"==> Saving validation image to {save_path}")
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)

                    pred = preds.detach().cpu().numpy()
                    pred = (pred * 255).astype(np.uint8)
                    pred_depth = preds_depth.detach().cpu().numpy()
                    truth = truths.detach().cpu().numpy()
                    truth = (truth * 255).astype(np.uint8)
                    
                    cv2.imwrite(save_path, cv2.cvtColor(
                        pred, cv2.COLOR_RGB2BGR))
                    if save_path_gt is not None:
                        cv2.imwrite(save_path_gt, cv2.cvtColor(
                            truth, cv2.COLOR_RGB2BGR))
                    if save_path_error is not None:
                        error = np.abs(truth.astype(np.float32) - pred.astype(np.float32)).mean(-1).astype(np.uint8)
                        cv2.imwrite(save_path_error, error)
                    np.save(save_path_depth, pred_depth)

                    if self.opt.with_sam or self.opt.with_mask:
                        pred_extra = preds_extra.detach().cpu().numpy()
                        np.save(save_path_extra, pred_extra)
                        if self.opt.with_mask:
                            instance_id = pred_extra.argmax(-1)
                            masks = overlay_mask_only(instance_id, self.color_map, render_id=1).detach().cpu().numpy()
                            cv2.imwrite(os.path.join(self.workspace, 'validation', f'{name}_{image_name}_mask.png'), cv2.cvtColor(
                                                         masks * 255, cv2.COLOR_RGB2BGR))
                    pbar.set_description(
                        f"loss={loss_val:.6f} ({total_loss/self.local_step:.6f})")
                    pbar.update(loader.batch_size)

        average_loss = total_loss / self.local_step
        self.stats["valid_loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if not self.use_loss_as_metric and len(self.metrics) > 0:
                result = self.metrics[0].measure()
                # if max mode, use -result
                self.stats["results"].append(
                    result if self.best_mode == 'min' else - result)
            else:
                # if no metric, choose best by min loss
                self.stats["results"].append(average_loss)

            for metric in self.metrics:
                self.log(metric.report(), style="blue")
                metric.clear()

        if self.ema is not None:
            self.ema.restore()

        self.log(
            f"++> Evaluate epoch {self.epoch} Finished, loss = {average_loss:.6f}")

    def save_checkpoint(self, name=None, full=False, best=False, remove_old=True):

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        state = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'stats': self.stats,
        }

        if full:
            state['optimizer'] = self.optimizer.state_dict()
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
            state['scaler'] = self.scaler.state_dict()
            if self.ema is not None:
                state['ema'] = self.ema.state_dict()

        if not best:

            state['model'] = self.model.state_dict()

            file_path = f"{name}.pth"

            if remove_old:
                self.stats["checkpoints"].append(file_path)

                if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
                    old_ckpt = os.path.join(
                        self.ckpt_path, self.stats["checkpoints"].pop(0))
                    if os.path.exists(old_ckpt):
                        os.remove(old_ckpt)

            torch.save(state, os.path.join(self.ckpt_path, file_path))

        else:
            if len(self.stats["results"]) > 0:
                if self.stats["best_result"] is None or self.stats["results"][-1] < self.stats["best_result"]:
                    self.log(
                        f"[INFO] New best result: {self.stats['best_result']} --> {self.stats['results'][-1]}")
                    self.stats["best_result"] = self.stats["results"][-1]

                    # save ema results
                    if self.ema is not None:
                        self.ema.store()
                        self.ema.copy_to()

                    state['model'] = self.model.state_dict()

                    # we don't consider continued training from the best ckpt, so we discard the unneeded density_grid to save some storage (especially important for dnerf)
                    # if 'density_grid' in state['model']:
                    #     del state['model']['density_grid']

                    if self.ema is not None:
                        self.ema.restore()

                    torch.save(state, self.best_path)
            else:
                self.log(
                    f"[WARN] no evaluated results found, skip saving best checkpoint.")

    def decode(self, loader):
        save_path = os.path.join(self.workspace, 'object_masks')
        os.makedirs(save_path, exist_ok=True)
        self.log(f"==> Start Test, save results to {save_path}")
        self.model.eval()
        assert self.point_3d is not None, 'Please provide 3d points for decoding'
        assert self.opt.use_point, 'Only support point decoding now'
        assert self.opt.with_sam, 'Only support SAM decoding now'

        with torch.no_grad():
            pose_dict = {}
            valid_dict = {}
            for i, data in tqdm.tqdm(enumerate(loader)):
                with torch.cuda.amp.autocast(enabled=self.opt.fp16):
                    preds, preds_depth, preds_mask, is_valid = self.decode_step(data)
                pred = preds.detach().cpu().numpy()
                pred = (pred * 255).astype(np.uint8)
                pred_depth = preds_depth.detach().cpu().numpy()
                file_name = data['img_names'][0]
                cv2.imwrite(os.path.join(save_path, f'{file_name}_rgb.png'), cv2.cvtColor(
                    pred, cv2.COLOR_RGB2BGR))
                np.save(os.path.join(
                    save_path, f'{file_name}_depth.npy'), pred_depth)
                preds_mask = preds_mask.detach().cpu().numpy()
                np.save(os.path.join(save_path, f'{file_name}_obj_mask.npy'), preds_mask)

                valid_dict[file_name] = int(is_valid)
            with open(os.path.join(save_path, 'valid_dict.json'), "w+") as f:
                json.dump(valid_dict, f, indent=4)
        self.log(f"==> Finished Decoding.")


    def load_checkpoint(self, checkpoint=None, model_only=False):

        if checkpoint is None:  # load latest
            checkpoint_list = sorted(glob.glob(f'{self.ckpt_path}/*.pth'))

            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log(
                    "[WARN] No checkpoint found, abort loading latest model.")
                return
        print(checkpoint)
        checkpoint_dict = torch.load(checkpoint, map_location=self.device)

        if 'model' not in checkpoint_dict:
            self.model.load_state_dict(checkpoint_dict)
            self.log("[INFO] loaded model.")
            return

        missing_keys, unexpected_keys = self.model.load_state_dict(
            checkpoint_dict['model'], strict=False)
        self.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")

        if self.ema is not None and 'ema' in checkpoint_dict:
            try:
                self.ema.load_state_dict(checkpoint_dict['ema'])
                self.log("[INFO] loaded EMA.")
            except:
                self.log("[WARN] failed to loaded EMA.")

        if model_only:
            return

        self.stats = checkpoint_dict['stats']
        self.epoch = checkpoint_dict['epoch']
        self.global_step = checkpoint_dict['global_step']
        self.log(
            f"[INFO] load at epoch {self.epoch}, global step {self.global_step}")

        if self.optimizer and 'optimizer' in checkpoint_dict:
            try:
                self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
                self.log("[INFO] loaded optimizer.")
            except:
                self.log("[WARN] Failed to load optimizer.")

        if self.lr_scheduler and 'lr_scheduler' in checkpoint_dict:
            try:
                self.lr_scheduler.load_state_dict(
                    checkpoint_dict['lr_scheduler'])
                self.log("[INFO] loaded scheduler.")
            except:
                self.log("[WARN] Failed to load scheduler.")

        if self.scaler and 'scaler' in checkpoint_dict:
            try:
                self.scaler.load_state_dict(checkpoint_dict['scaler'])
                self.log("[INFO] loaded scaler.")
            except:
                self.log("[WARN] Failed to load scaler.")
