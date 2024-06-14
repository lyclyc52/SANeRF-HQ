import os
import torch
import lpips
import numpy as np
try:
    from torchmetrics.functional import structural_similarity_index_measure as ssim
except:  # old versions
    from torchmetrics.functional import ssim

class PSNRMeter:
    def __init__(self):
        self.V = 0
        self.N = 0

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)

        return outputs

    def update(self, preds, truths):
        # [B, N, 3] or [B, H, W, 3], range[0, 1]
        preds, truths = self.prepare_inputs(preds, truths)

        # simplified since max_pixel_value is 1 here.
        psnr = -10 * np.log10(np.mean((preds - truths) ** 2))

        self.V += psnr
        self.N += 1

        return psnr

    def measure(self):
        if self.N == 0:
            return 0
        else:
            return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "PSNR"),
                          self.measure(), global_step)

    def report(self):
        return f'PSNR = {self.measure():.6f}'


class LPIPSMeter:
    def __init__(self, net='vgg', device=None):
        self.V = 0
        self.N = 0
        self.net = net

        self.device = device if device is not None else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.fn = lpips.LPIPS(net=net).eval().to(self.device)

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if len(inp.shape) == 3:
                inp = inp.unsqueeze(0)
            inp = inp.permute(0, 3, 1, 2).contiguous()  # [B, 3, H, W]
            inp = inp.to(self.device)
            outputs.append(inp)
        return outputs

    def update(self, preds, truths):
        # [H, W, 3] --> [B, 3, H, W], range in [0, 1]
        preds, truths = self.prepare_inputs(preds, truths)
        # normalize=True: [0, 1] to [-1, 1]
        v = self.fn(truths, preds, normalize=True).item()
        self.V += v
        self.N += 1

        return v

    def measure(self):
        if self.N > 0:
            return self.V / self.N
        else:
            return 0

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(
            prefix, f"LPIPS ({self.net})"), self.measure(), global_step)

    def report(self):
        return f'LPIPS ({self.net}) = {self.measure():.6f}'


class SSIMMeter:
    def __init__(self, device=None):
        self.V = 0
        self.N = 0

        self.device = device if device is not None else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if len(inp.shape) == 3:
                inp = inp.unsqueeze(0)
            inp = inp.permute(0, 3, 1, 2).contiguous()  # [B, 3, H, W]
            inp = inp.to(self.device)
            outputs.append(inp)
        return outputs

    def update(self, preds, truths):
        # [B, H, W, 3] --> [B, 3, H, W], range in [0, 1]
        preds, truths = self.prepare_inputs(preds, truths)

        v = ssim(preds, truths)

        self.V += v
        self.N += 1

    def measure(self):
        if self.N > 0:
            return self.V / self.N
        else:
            return 0

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "SSIM"),
                          self.measure(), global_step)

    def report(self):
        return f'SSIM = {self.measure():.6f}'


class MeanIoUMeter:
    def __init__(self):
        self.V = 0
        self.N = 0

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy().astype(np.int64)
            outputs.append(inp)

        return outputs

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(preds, truths) # [B, N] or [B, H, W], range: [0, num_classes-1]
        num_classes = max(preds.max() + 1, truths.max() + 1)
          
        ious = []
        for i in range(num_classes):
            intersection = np.logical_and(preds == i, truths == i).sum()
            union = np.logical_or(preds == i, truths == i).sum()

            if union > 0:
                ious.append(intersection / union)
        v = np.mean(ious)
        self.V += v
        self.N += 1
        return v

    def measure(self):
        if self.N > 0:
            return self.V / self.N
        else:
            return 0

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "mIoU"), self.measure(), global_step)

    def report(self):
        return f'mIoU = {self.measure():.6f}'
    
    def name(self):
        return 'mIoU'




class MSEMeter:
    def __init__(self):
        self.V = 0
        self.N = 0

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)

        return outputs

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(preds, truths)
        v = np.mean((preds - truths) ** 2)
        self.V += v
        self.N += 1

    def measure(self):
        if self.N > 0:
            return self.V / self.N
        else:
            return 0

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "MSE"), self.measure(), global_step)

    def report(self):
        return f'MSE = {self.measure():.6f}'
    
    def name(self):
        return 'MSE'
