"""RIFE 4.7 network (IFNet) for MpiRifeVFI (MPI-1342).

Ported from ComfyUI-Frame-Interpolation (vfi_models/rife/rife_arch.py), MIT licence,
Copyright (c) 2023 Fannovel16 (see rife/LICENSE-Frame-Interpolation), which itself derives from
Practical-RIFE by hzwer (MIT): https://github.com/hzwer/Practical-RIFE

Only the arch_ver "4.7" paths are kept: rife47.pth is the one weight Cubric Studio ships. The
state-dict key names are unchanged, so a rife47.pth that loads into the original loads here.
No ComfyUI import: the module is plain torch, so tests can load it alone.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

_backwarp_grid = {}


class ResConv(nn.Module):
    def __init__(self, c, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, dilation, dilation=dilation, groups=1)
        self.beta = nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        return self.relu(self.conv(x) * self.beta + x)


def warp(tenInput, tenFlow):
    # Upstream keys the grid on a module-level device (the ComfyUI device) while the key itself
    # names the flow's device; the flow always lives on that device there, so using the flow's own
    # device is the same thing and lets the net run on CPU in tests.
    device = tenFlow.device
    k = (str(device), str(tenFlow.size()))
    if k not in _backwarp_grid:
        tenHorizontal = (
            torch.linspace(-1.0, 1.0, tenFlow.shape[3], device=device)
            .view(1, 1, 1, tenFlow.shape[3])
            .expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
        )
        tenVertical = (
            torch.linspace(-1.0, 1.0, tenFlow.shape[2], device=device)
            .view(1, 1, tenFlow.shape[2], 1)
            .expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
        )
        _backwarp_grid[k] = torch.cat([tenHorizontal, tenVertical], 1).to(device)

    tenFlow = torch.cat(
        [
            tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
            tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0),
        ],
        1,
    )

    g = (_backwarp_grid[k] + tenFlow).permute(0, 2, 3, 1)

    if g.dtype != tenInput.dtype:
        g = g.to(tenInput.dtype)

    padding_mode = "border"
    if device.type == "mps":
        # https://github.com/pytorch/pytorch/issues/125098
        padding_mode = "zeros"
        g = g.clamp(-1, 1)
    return torch.nn.functional.grid_sample(
        input=tenInput,
        grid=g,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=True,
    )


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True,
        ),
        nn.LeakyReLU(0.2, True),
    )


class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64):
        super().__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c // 2, 3, 2, 1),
            conv(c // 2, c, 3, 2, 1),
        )
        self.convblock = nn.Sequential(*[ResConv(c) for _ in range(8)])
        self.lastconv = nn.Sequential(nn.ConvTranspose2d(c, 4 * 6, 4, 2, 1), nn.PixelShuffle(2))

    def forward(self, x, flow=None, scale=1):
        x = F.interpolate(x, scale_factor=1.0 / scale, mode="bilinear", align_corners=False)
        if flow is not None:
            flow = F.interpolate(flow, scale_factor=1.0 / scale, mode="bilinear", align_corners=False) * 1.0 / scale
            x = torch.cat((x, flow), 1)
        feat = self.conv0(x)
        feat = self.convblock(feat)
        tmp = self.lastconv(feat)
        tmp = F.interpolate(tmp, scale_factor=scale, mode="bilinear", align_corners=False)
        flow = tmp[:, :4] * scale
        mask = tmp[:, 4:5]
        return flow, mask


class IFNet(nn.Module):
    """RIFE 4.7 (also the 4.8 / 4.9 weights)."""

    def __init__(self):
        super().__init__()
        self.block0 = IFBlock(7 + 8, c=192)
        self.block1 = IFBlock(8 + 4 + 8, c=128)
        self.block2 = IFBlock(8 + 4 + 8, c=96)
        self.block3 = IFBlock(8 + 4 + 8, c=64)
        self.encode = nn.Sequential(nn.Conv2d(3, 16, 3, 2, 1), nn.ConvTranspose2d(16, 4, 4, 2, 1))

    def forward(
        self,
        img0,
        img1,
        timestep=0.5,
        scale_list=[16, 8, 4, 2, 1],
        training=True,
        fastmode=True,
        ensemble=False,
    ):
        """Same signature and positional order as upstream. MpiRifeVFI calls it exactly the way
        upstream's node does, so its fast_mode lands on `training` and its ensemble on `fastmode`
        (a no-op for 4.7): the node-level ensemble switch never turns ensembling on. Kept, because
        a port must not change what the saved Flow renders."""
        img0 = torch.clamp(img0, 0, 1)
        img1 = torch.clamp(img1, 0, 1)

        n, c, h, w = img0.shape
        ph = ((h - 1) // 64 + 1) * 64
        pw = ((w - 1) // 64 + 1) * 64
        padding = (0, pw - w, 0, ph - h)
        img0 = F.pad(img0, padding)
        img1 = F.pad(img1, padding)
        x = torch.cat((img0, img1), 1)

        if training == False:
            channel = x.shape[1] // 2
            img0 = x[:, :channel]
            img1 = x[:, channel:]
        if not torch.is_tensor(timestep):
            timestep = (x[:, :1].clone() * 0 + 1) * timestep
        else:
            timestep = timestep.repeat(1, 1, img0.shape[2], img0.shape[3])

        f0 = self.encode(img0[:, :3])
        f1 = self.encode(img1[:, :3])

        warped_img0 = img0
        warped_img1 = img1
        flow = None
        mask = None
        block = [self.block0, self.block1, self.block2, self.block3]

        for i in range(4):
            if flow is None:
                flow, mask = block[i](
                    torch.cat((img0[:, :3], img1[:, :3], f0, f1, timestep), 1),
                    None,
                    scale=scale_list[i],
                )
                if ensemble:
                    f_, m_ = block[i](
                        torch.cat((img1[:, :3], img0[:, :3], f1, f0, 1 - timestep), 1),
                        None,
                        scale=scale_list[i],
                    )
                    flow = (flow + torch.cat((f_[:, 2:4], f_[:, :2]), 1)) / 2
                    mask = (mask + (-m_)) / 2
            else:
                fd, m0 = block[i](
                    torch.cat(
                        (
                            warped_img0[:, :3],
                            warped_img1[:, :3],
                            warp(f0, flow[:, :2]),
                            warp(f1, flow[:, 2:4]),
                            timestep,
                            mask,
                        ),
                        1,
                    ),
                    flow,
                    scale=scale_list[i],
                )
                flow = flow + fd

                if ensemble:
                    wf0 = warp(f0, flow[:, :2])
                    wf1 = warp(f1, flow[:, 2:4])
                    f_, m_ = block[i](
                        torch.cat(
                            (warped_img1[:, :3], warped_img0[:, :3], wf1, wf0, 1 - timestep, -mask),
                            1,
                        ),
                        torch.cat((flow[:, 2:4], flow[:, :2]), 1),
                        scale=scale_list[i],
                    )
                    fd = (fd + torch.cat((f_[:, 2:4], f_[:, :2]), 1)) / 2
                    mask = (m0 + (-m_)) / 2
                else:
                    mask = m0

            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])

        mask = torch.sigmoid(mask)
        merged_final = warped_img0 * mask + warped_img1 * (1 - mask)
        return merged_final[:, :, :h, :w]
