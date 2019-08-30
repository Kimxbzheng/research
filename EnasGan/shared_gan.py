import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FactorizedReduction2(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(FactorizedReduction2, self).__init__()

        assert out_planes % 2 == 0, ("Need even number of filters when using this factorized reduction.")

        self.in_planes = in_planes
        self.out_planes = out_planes

        self.fr = nn.Sequential(
            nn.InstanceNorm2d(self.out_planes, affine=True),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        self.conv = nn.Conv2d(list(x.size())[1], self.out_planes, kernel_size=4, stride=2, padding=1)
        self.conv = self.conv.cuda()
        y = self.conv(x)
        out = self.fr(y)
        return out

class DENASLayer2(nn.Module):
    def __init__(self, layer_id, in_planes, out_planes):
        super(DENASLayer2, self).__init__()

        self.layer_id = layer_id
        self.in_planes = in_planes
        self.out_planes = out_planes

        self.branch_0 = ConvBranch2(in_planes, out_planes, kernel_size=3)
        self.branch_1 = ConvBranch2(in_planes, out_planes, kernel_size=5)

        self.bn = nn.InstanceNorm2d(self.out_planes, affine=True)

    def forward(self, x, prev_layers, sample_arc):
        layer_type = sample_arc[0]
        if self.layer_id > 0:
            skip_indices = sample_arc[1]
        else:
            skip_indices = []

        if layer_type == 0:
            out = self.branch_0(x)
        elif layer_type == 1:
            out = self.branch_1(x)
        else:
            raise ValueError("Unknown layer_type {}".format(layer_type))

        for i, skip in enumerate(skip_indices):
            if skip == 1:
                out += prev_layers[i]

        out = self.bn(out)
        return out


class DFixedLayer2(nn.Module):
    def __init__(self, layer_id, in_planes, out_planes, sample_arc):
        super(DFixedLayer2, self).__init__()

        self.layer_id = layer_id
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.sample_arc = sample_arc

        self.layer_type = sample_arc[0]
        if self.layer_id > 0:
            self.skip_indices = sample_arc[1]
        else:
            self.skip_indices = torch.zeros(1)

        if self.layer_type == 0:
            self.branch = ConvBranch2(in_planes, out_planes, kernel_size=3)
        elif self.layer_type == 1:
            self.branch = ConvBranch2(in_planes, out_planes, kernel_size=5)
        else:
            raise ValueError("Unknown layer_type {}".format(self.layer_type))

        # Use concatentation instead of addition in the fixed layer for some reason
        in_planes = int((torch.sum(self.skip_indices).item() + 1) * in_planes)
        self.dim_reduc = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm2d(self.out_planes, affine=True)
        )

    def forward(self, x, prev_layers, sample_arc):
        out = self.branch(x)

        res_layers = []
        for i, skip in enumerate(self.skip_indices):
            if skip == 1:
                res_layers.append(prev_layers[i])
        prev = res_layers + [out]
        prev = torch.cat(prev, dim=1)

        out = self.dim_reduc(prev)
        return out


class ConvBranch2(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        super(ConvBranch2, self).__init__()
        assert kernel_size in [3, 5], "Kernel size must be either 3 or 5"

        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size

        padding = (kernel_size - 1) // 2
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=padding),
            nn.InstanceNorm2d(self.out_planes, affine=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        out = self.out_conv(x)
        return out


class Discriminator3(nn.Module):
    def __init__(self, args):
        super(Discriminator3, self).__init__()
        print('Using ENAS Discriminator v2')

        self.num_layers = args.child_num_layers
        self.num_branches = args.child_num_branches
        self.out_filters = 2*args.z_dim
        self.fixed_arc = args.fixed_arc

        downsample_distance = self.num_layers // 3
        self.downsample_layers = [downsample_distance, 2 * downsample_distance]

        self.stem_conv = nn.Sequential(
            nn.Conv2d(args.channels, self.out_filters, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(self.out_filters, affine=True, track_running_stats=False),
            nn.LeakyReLU(0.2),
        )

        self.layers = nn.ModuleList([])
        self.downsampled_layers = nn.ModuleList([])

        for layer_id in range(self.num_layers):
            if self.fixed_arc == False:
                layer = DENASLayer2(layer_id, self.out_filters, self.out_filters)
            else:
                layer = DFixedLayer2(layer_id, self.out_filters, self.out_filters, self.fixed_arc[str(layer_id)])
            self.layers.append(layer)

            if layer_id in self.downsample_layers:
                for i in range(len(self.layers)):
                    self.downsampled_layers.append(FactorizedReduction2(self.out_filters, self.out_filters * 2))
                self.out_filters *= 2

        self.leaf_conv = nn.Conv2d(self.out_filters, 1, kernel_size=4, stride=1, padding=0)

    def forward(self, x, sample_arc):
        x = self.stem_conv(x)

        prev_layers = []
        downsample_count = 0
        for layer_id in range(self.num_layers):
            x = self.layers[layer_id](x, prev_layers, sample_arc[str(layer_id)])
            prev_layers.append(x)
            if layer_id in self.downsample_layers:
                for i, prev_layer in enumerate(prev_layers):
                    prev_layers[i] = self.downsampled_layers[downsample_count](prev_layer)
                    downsample_count += 1
                x = prev_layers[-1]

        out = self.leaf_conv(x)

        return out
        

class FactorizedExpansion2(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(FactorizedExpansion2, self).__init__()

        assert out_planes % 2 == 0, ("Need even number of filters when using this factorized reduction.")

        self.in_planes = in_planes
        self.out_planes = out_planes

        self.fr = nn.Sequential(
            nn.BatchNorm2d(self.out_planes),
            nn.ReLU()
        )

    def forward(self, x):
        self.deconv = nn.ConvTranspose2d(list(x.size())[1], self.out_planes, kernel_size=4, stride=2, padding=1)
        self.deconv = self.deconv.cuda()
        y = self.deconv(x)
        out = self.fr(y)
        return out

class GENASLayer2(nn.Module):
    def __init__(self, layer_id, in_planes, out_planes):
        super(GENASLayer2, self).__init__()

        self.layer_id = layer_id
        self.in_planes = in_planes
        self.out_planes = out_planes

        self.branch_0 = DeconvBranch2(in_planes, out_planes, kernel_size=3)
        self.branch_1 = DeconvBranch2(in_planes, out_planes, kernel_size=5)

        self.bn = nn.BatchNorm2d(self.out_planes)

    def forward(self, x, prev_layers, sample_arc):
        layer_type = sample_arc[0]
        if self.layer_id > 0:
            skip_indices = sample_arc[1]
        else:
            skip_indices = []

        if layer_type == 0:
            out = self.branch_0(x)
        elif layer_type == 1:
            out = self.branch_1(x)
        else:
            raise ValueError("Unknown layer_type {}".format(layer_type))

        for i, skip in enumerate(skip_indices):
            if skip == 1:
                out += prev_layers[i]

        out = self.bn(out)
        return out


class GFixedLayer2(nn.Module):
    def __init__(self, layer_id, in_planes, out_planes, sample_arc):
        super(GFixedLayer2, self).__init__()

        self.layer_id = layer_id
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.sample_arc = sample_arc

        self.layer_type = sample_arc[0]
        if self.layer_id > 0:
            self.skip_indices = sample_arc[1]
        else:
            self.skip_indices = torch.zeros(1)

        if self.layer_type == 0:
            self.branch = DeconvBranch2(in_planes, out_planes, kernel_size=3)
        elif self.layer_type == 1:
            self.branch = DeconvBranch2(in_planes, out_planes, kernel_size=5)
        else:
            raise ValueError("Unknown layer_type {}".format(self.layer_type))

        # Use concatentation instead of addition in the fixed layer for some reason
        in_planes = int((torch.sum(self.skip_indices).item() + 1) * in_planes)
        self.dim_reduc = nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(self.out_planes)
        )

    def forward(self, x, prev_layers, sample_arc):
        out = self.branch(x)

        res_layers = []
        for i, skip in enumerate(self.skip_indices):
            if skip == 1:
                res_layers.append(prev_layers[i])
        prev = res_layers + [out]
        prev = torch.cat(prev, dim=1)

        out = self.dim_reduc(prev)
        return out


class DeconvBranch2(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        super(DeconvBranch2, self).__init__()
        assert kernel_size in [3, 5], "Kernel size must be either 3 or 5"

        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size

        padding = (kernel_size - 1) // 2
        self.out_conv = nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(self.out_planes),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.out_conv(x)
        return out


class Generator3(nn.Module):
    def __init__(self, args):
        super(Generator3, self).__init__()
        print('Using ENAS Generator v2')

        self.num_layers = args.child_num_layers
        self.num_branches = args.child_num_branches
        self.out_filters = 8*args.z_dim
        self.fixed_arc = args.fixed_arc

        upsample_distance = self.num_layers // 3
        self.upsample_layers = [upsample_distance, 2 * upsample_distance]
        self.stem_deconv = nn.Sequential(
            nn.ConvTranspose2d(args.latent_dim, self.out_filters, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(self.out_filters),
            nn.ReLU()
        )

        self.layers = nn.ModuleList([])
        self.upsampled_layers = nn.ModuleList([])

        for layer_id in range(self.num_layers):
            if self.fixed_arc == False:
                layer = GENASLayer2(layer_id, self.out_filters, self.out_filters)
            else:
                layer = GFixedLayer2(layer_id, self.out_filters, self.out_filters, self.fixed_arc[str(layer_id)])
            self.layers.append(layer)

            if layer_id in self.upsample_layers:
                for i in range(len(self.layers)):
                    self.upsampled_layers.append(FactorizedExpansion2(self.out_filters, self.out_filters // 2))
                self.out_filters = self.out_filters // 2

        self.leaf_deconv = nn.ConvTranspose2d(self.out_filters, args.channels, kernel_size=4, stride=2, padding=1)
        self.generate = nn.Tanh() 

    def forward(self, x, sample_arc):
        x = self.stem_deconv(x)

        prev_layers = []
        upsample_count = 0
        for layer_id in range(self.num_layers):
            x = self.layers[layer_id](x, prev_layers, sample_arc[str(layer_id)])
            prev_layers.append(x)
            if layer_id in self.upsample_layers:
                for i, prev_layer in enumerate(prev_layers):
                    prev_layers[i] = self.upsampled_layers[upsample_count](prev_layer)
                    upsample_count += 1
                x = prev_layers[-1]

        x = self.leaf_deconv(x)
        out = self.generate(x)

        return out
