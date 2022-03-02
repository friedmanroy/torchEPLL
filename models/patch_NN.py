import torch as tp
import torch.nn as nn


def sequential(*args):
    if len(args) == 1:
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


class ResBlock(nn.Module):
    def __init__(self, channels: int=64, kernel_size=3, stride=1, padding=1, bias=True,
                 transpose: bool=False, repeats: int=3):
        super(ResBlock, self).__init__()

        modules = []
        for _ in range(repeats):
            # add a convolution (or transpose convolution) layer
            if not transpose: modules.append(nn.Conv2d(in_channels=channels, out_channels=channels,
                                                       kernel_size=kernel_size, stride=stride, padding=padding,
                                                       bias=bias, padding_mode='reflect'))
            else: modules.append(nn.ConvTranspose2d(in_channels=channels, out_channels=channels,
                                                    kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
            # add ReLU activation
            modules.append(nn.ReLU())

            # add another convolution (or transpose convolution) layer
            if not transpose: modules.append(nn.Conv2d(in_channels=channels, out_channels=channels,
                                                       kernel_size=kernel_size, stride=stride, padding=padding,
                                                       bias=bias, padding_mode='reflect'))
            else: modules.append(nn.ConvTranspose2d(in_channels=channels, out_channels=channels,
                                                    kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))

        self.res = nn.Sequential(*modules)

    def forward(self, x):
        return x + self.res(x)


class SimpleResNet(nn.Module):
    def __init__(self, n_blocks: int=5, bias: bool=True, kernel_size: int=3):
        super(SimpleResNet, self).__init__()

        strt_chans = 16
        channels = [strt_chans]
        for i in range(1, n_blocks):
            if i < n_blocks//2: channels.append(channels[i-1]*2)
            else: channels.append(channels[i-1]/2)

        self.first = sequential(
            nn.Conv2d(in_channels=4, out_channels=strt_chans, bias=bias, stride=1, padding=1,
                      kernel_size=kernel_size, padding_mode='reflect'),
            nn.ReLU()
        )
        mods = []
        for i in range(n_blocks-1):
            mods.append(sequential(
                ResBlock(channels=int(channels[i]), kernel_size=kernel_size, bias=bias, repeats=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=int(channels[i]), out_channels=int(channels[i+1]), bias=bias, stride=1, padding=1,
                          kernel_size=kernel_size, padding_mode='reflect'),
                nn.ReLU(),
                ResBlock(channels=int(channels[i+1]), kernel_size=kernel_size, bias=bias, repeats=1),
                nn.ReLU()
            ))
        self.body = sequential(*mods)
        self.last = nn.Conv2d(in_channels=int(channels[-1]), out_channels=3, bias=bias, stride=1, padding=1,
                              kernel_size=kernel_size, padding_mode='reflect')

    def forward(self, x):
        return self.last(self.body(self.first(x)))

    def denoise(self, x, var: float):
        mp = tp.ones(x.shape[:-1], device=x.device)*var
        x = tp.concat([x-0.5, mp[..., None]], dim=-1).permute(0, -1, 1, 2)
        return self.forward(x).permute(0, 2, 3, 1) + 0.5
