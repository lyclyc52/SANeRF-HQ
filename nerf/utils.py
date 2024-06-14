import os
import random
import numpy as np
import torch
import torch.nn.functional as F



import trimesh


from packaging import version as pver




def affinity_matrix(X):
    X_norm = F.normalize(X, dim=1)
    A = torch.mm(X_norm, X_norm.t())
    return A


def overlay_mask(image, mask, alpha=0.7, color=[1, 0, 0]):
    # image: [H, W, 3]
    # mask: [H, W]
    over_image = image.clone()
    over_image[mask] = torch.tensor(
        color, device=image.device, dtype=image.dtype)
    return image * alpha + over_image * (1 - alpha)


def overlay_mask_only(instance_id, color_map=None, render_id=-1):
    H, W = instance_id.shape
    instance_id = instance_id.reshape(H*W)
    

    if render_id == -1:
        instance_id_flatten = instance_id.reshape(H*W)
        color_mask = color_map[instance_id_flatten]
        color_mask = color_mask.reshape(H, W, -1)
    else:
        mask = instance_id == render_id
        mask = mask.reshape(H*W).astype(int)
        color_mask = color_map[mask]
        color_mask = color_mask.reshape(H, W, -1)
    return color_mask


def overlay_mask_composition(image, instance_id, color_map=None, render_id=-1, alpha=0.7):
    H, W = instance_id.shape
    instance_id_flatten = instance_id.reshape(H*W)
    # print(instance_id_flatten.max())
    # print(color_map.shape)
    # exit()
    color_mask = color_map[instance_id_flatten]
    color_mask = color_mask.reshape(H, W, -1)
    if render_id != -1:
        mask = instance_id != render_id
        color_mask[mask] = image[mask]
    return image * alpha + color_mask * (1 - alpha)


def overlay_mask_heatmap(mask, instance_id, color_map=None, alpha=0.7):
    # image: [H, W, 3]
    # mask: [H, W]

    if isinstance(instance_id, int):
        instance_id = torch.ones_like(mask) * instance_id
        instance_id = instance_id.to(color_map.device).to(torch.long)
    H, W = instance_id.shape
    instance_id = instance_id.reshape(H*W)
    color_mask = color_map[instance_id]
    color_mask = color_mask.reshape(H, W, -1)

    output = color_mask * mask[..., None]

    return output


def overlay_point(image, points, radius=2, color=[1, 0, 0], inputs_point_labels = None):
    # image: [H, W, 3]
    # points: [1, 2]
    mask = torch.zeros_like(image[:, :, 0]).bool()

        
    if inputs_point_labels is None:
        for point in points:
            mask[point[1]-radius:point[1]+radius,
                point[0]-radius:point[0]+radius] = True
        image[mask] = torch.tensor(color, device=image.device, dtype=image.dtype)
    else:
        for ind in range(len(points)):
            mask = torch.zeros_like(image[:, :, 0]).bool()
            mask[points[ind][1]-radius:points[ind][1]+radius,
                points[ind][0]-radius:points[ind][0]+radius] = True
            cur_color = [0, 1, 0] if inputs_point_labels[ind] == 0 else [1, 0, 0]
            image[mask] = torch.tensor(cur_color, device=image.device, dtype=image.dtype)
    return image



def scale_img_nhwc(x, size, mag='bilinear', min='bilinear'):
    assert (x.shape[1] >= size[0] and x.shape[2] >= size[1]) or (x.shape[1] < size[0]
                                                                 and x.shape[2] < size[1]), "Trying to magnify image in one dimension and minify in the other"
    y = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
    if x.shape[1] > size[0] and x.shape[2] > size[1]:  # Minification, previous size was bigger
        y = torch.nn.functional.interpolate(y, size, mode=min)
    else:  # Magnification
        if mag == 'bilinear' or mag == 'bicubic':
            y = torch.nn.functional.interpolate(
                y, size, mode=mag, align_corners=True)
        else:
            y = torch.nn.functional.interpolate(y, size, mode=mag)
    return y.permute(0, 2, 3, 1).contiguous()  # NCHW -> NHWC


def scale_img_hwc(x, size, mag='bilinear', min='bilinear'):
    return scale_img_nhwc(x[None, ...], size, mag, min)[0]


def scale_img_nhw(x, size, mag='bilinear', min='bilinear'):
    return scale_img_nhwc(x[..., None], size, mag, min)[..., 0]


def scale_img_hw(x, size, mag='bilinear', min='bilinear'):
    return scale_img_nhwc(x[None, ..., None], size, mag, min)[0, ..., 0]


def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')



def project_to_3d(pts, pose, intrinsics, depth):
    '''
    Args:
        pts: Nx2
        pose: 4x4
        intrinsics: fx, fy, cx, cy
        depth: HxW
    '''
    pose[:,1] = -pose[:, 1]
    pose[:,2] = -pose[:, 2]
    
    pts = torch.from_numpy(pts)
    pose = torch.tensor(pose)
    fx, fy, cx, cy = intrinsics
    zs = torch.ones_like(pts[..., 0])
    xs = (pts[..., 0] - cx) / fx * zs
    ys = (pts[..., 1]  - cy) / fy * zs
    directions = torch.stack((xs, ys, zs), dim=-1)
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    
    pts_z = depth[pts[..., 1], pts[..., 0]] 
    directions = directions * pts_z
    
    rays_d = directions @ pose[:3, :3].transpose(1,0) # (N, 3)
    rays_o = pose[:3, 3] # [3]
    rays_o = rays_o[None, :]
    return rays_o + rays_d



def sample_points_by_errors(H, W, incoherent_mask, incoherent_mask_size):
    inds_coarse_center = torch.multinomial(incoherent_mask.to(torch.float32), 1)
    
    inds_x, inds_y = inds_coarse_center // incoherent_mask_size, inds_coarse_center % incoherent_mask_size

    sx, sy = H / incoherent_mask_size, W / incoherent_mask_size
    mask_point_x = inds_x * sx
    mask_point_y = inds_y * sy
    
    return (mask_point_x, mask_point_y)
                



@torch.cuda.amp.autocast(enabled=False)
def get_rays(poses, intrinsics, H, W, N=-1, patch_size=1, coords=None, device='cpu', incoherent_mask=None, 
             include_incoherent_region=False, incoherent_mask_size = 128, random_sample=False):
    ''' get rays
    Args:
        poses: [N/1, 4, 4], cam2world
        intrinsics: [N/1, 4] tensor or [4] ndarray
        H, W, N: int
    Returns:
        rays_o, rays_d: [N, 3]
        i, j: [N]
    '''

    if isinstance(intrinsics, np.ndarray):
        fx, fy, cx, cy = intrinsics
    else:
        fx, fy, cx, cy = intrinsics[:, 0], intrinsics[:,
                                                      1], intrinsics[:, 2], intrinsics[:, 3]

    i, j = custom_meshgrid(torch.linspace(
        0, W-1, W, device=device), torch.linspace(0, H-1, H, device=device))  # float
    
    i = i.t().contiguous().view(-1) + 0.5
    j = j.t().contiguous().view(-1) + 0.5


    results = {}
    

    if N > 0:
        
        if coords is not None:
            inds = coords[:, 0] * W + coords[:, 1]
        elif patch_size > 1 and not random_sample:
            # when patch_size > 1, we sample patches randomly.
            if incoherent_mask is not None and include_incoherent_region:
                inds_coarse_center = torch.multinomial(incoherent_mask.to(torch.float32), 1)
                inds_x, inds_y = inds_coarse_center // incoherent_mask_size, inds_coarse_center % incoherent_mask_size
                # rand_indices = torch.multinomial(incoherent_mask.view(-1, H*W).to(torch.float32), 1)
                sx, sy = H / incoherent_mask_size, W / incoherent_mask_size
                mask_point_x = inds_x * sx
                mask_point_y = inds_y * sy

                # print(mask_point_x, mask_point_y)
                inds_x = torch.clamp(mask_point_x-patch_size//2, min=0, max=H-patch_size-1).long()
                inds_y = torch.clamp(mask_point_y-patch_size//2, min=0, max=W-patch_size-1).long()
            else:
                # random sample left-top cores.
                num_patch = N // (patch_size ** 2)
                inds_x = torch.randint(
                    0, H - patch_size, size=[num_patch], device=device)
                inds_y = torch.randint(
                    0, W - patch_size, size=[num_patch], device=device)
            inds = torch.stack([inds_x, inds_y], dim=-1)  # [num_sample, 1, 2]
            # create meshgrid for each patch
            pi, pj = custom_meshgrid(torch.arange(
                patch_size, device=device), torch.arange(patch_size, device=device))
            offsets = torch.stack(
                [pi.reshape(-1), pj.reshape(-1)], dim=-1)  # [p^2, 2]
            # inds = inds.unsqueeze(1) + offsets.unsqueeze(0)  
            inds = inds + offsets.unsqueeze(0)  # [num_sample, p^2, 2]
            inds = inds.view(-1, 2)  # [N, 2]
            inds = inds[..., 0] * W + inds[..., 1]  # [N], flatten
            
        
        elif patch_size == 1 and not random_sample:
            inds_coarse = torch.multinomial(incoherent_mask.to(torch.float32), N, replacement=False) # [B, N], but in [0, 128*128)
            B = poses.shape[0]
            # map to the original resolution with random perturb.
            inds_x, inds_y = inds_coarse // 128, inds_coarse % 128 # `//` will throw a warning in torch 1.10... anyway.
            sx, sy = H / incoherent_mask_size, W / incoherent_mask_size
            inds_x = (inds_x * sx + torch.rand(B, N, device=device) * sx).long().clamp(max=H - 1)
            inds_y = (inds_y * sy + torch.rand(B, N, device=device) * sy).long().clamp(max=W - 1)
            inds = inds_x * W + inds_y
            inds = inds[0]

            results['inds_coarse'] = inds_coarse # need this when updating error_map
        else:  # random sampling
            inds = torch.randint(
                0, H*W, size=[N], device=device)  # may duplicate

        i = torch.gather(i, -1, inds)
        j = torch.gather(j, -1, inds)

        results['i'] = i.long()
        results['j'] = j.long()
        
    else:
        inds = torch.arange(H*W, device=device)

    zs = -torch.ones_like(i)  # z is flipped
    xs = (i - cx) / fx
    ys = -(j - cy) / fy  # y is flipped
    directions = torch.stack((xs, ys, zs), dim=-1)  # [N, 3]
    # do not normalize to get actual depth, ref: https://github.com/dunbar12138/DSNeRF/issues/29
    # directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    # [N, 1, 3] @ [N, 3, 3] --> [N, 1, 3]
    

    rays_d = (directions.unsqueeze(1) @
              poses[:, :3, :3].transpose(-1, -2)).squeeze(1)

    rays_o = poses[:, :3, 3].expand_as(rays_d)  # [N, 3]

    results['rays_o'] = rays_o
    results['rays_d'] = rays_d
    
    
    
    

    # if incoherent_mask is not None and include_incoherent_region and patch_size > 1:
    if results.get('inds_coarse') == None:
        inds_x, inds_y = inds // W, inds % W
        sx_coarse, sy_coarse = incoherent_mask_size / H, incoherent_mask_size / W
        inds_coarse_x = (inds_x * sx_coarse).long()
        inds_coarse_y = (inds_y * sy_coarse).long()

        results['inds_coarse'] = (inds_coarse_x * incoherent_mask_size + inds_coarse_y).long()

    # visualize_rays(rays_o[0].detach().cpu().numpy(), rays_d[0].detach().cpu().numpy())

    return results



def get_incoherent_mask(input_masks, sfact=2, keep_size=True):
    mask = input_masks.float()
    w = input_masks.shape[-1]
    h = input_masks.shape[-2]
    mask_small = F.interpolate(mask, (h//sfact, w//sfact), mode='bilinear')
    mask_recover = F.interpolate(mask_small, (h, w), mode='bilinear')
    mask_residue = (mask - mask_recover).abs()
    mask_uncertain = F.interpolate(
        mask_residue, (h//sfact, w//sfact), mode='bilinear')
    mask_uncertain[mask_uncertain >= 0.01] = 1.
    
    if keep_size:
        mask_uncertain = F.interpolate(
            mask_uncertain, (h,w), mode='nearest')

    return mask_uncertain


def visualize_rays(rays_o, rays_d):

    axes = trimesh.creation.axis(axis_length=4)
    box = trimesh.primitives.Box(extents=(2, 2, 2)).as_outline()
    box.colors = np.array([[128, 128, 128]] * len(box.entities))
    objects = [axes, box]

    for i in range(0, rays_o.shape[0], 10):
        ro = rays_o[i]
        rd = rays_d[i]

        segs = np.array([[ro, ro + rd * 3]])
        segs = trimesh.load_path(segs)
        objects.append(segs)

    trimesh.Scene(objects).show()


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True

class Cache:
    def __init__(self, size=100):
        self.size = size
        self.data = {}
        self.key = 0

    def full(self):
        return len(self.data) == self.size

    def insert(self, x):
        self.data[self.key] = x
        self.key = (self.key + 1) % self.size

    def get(self, key=None):
        if key is None:
            key = random.randint(0, len(self.data) - 1)
        return self.data[key]


