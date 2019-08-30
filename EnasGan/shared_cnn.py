import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

'''
Implementation Notes:
-Setting track_running_stats to True in BatchNorm layers seems to hurt validation
    and test performance for some reason, so here it is disabled even though it
    is used in the official implementation.
'''

class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        print('Using default discriminator')
        self.args = args
        self.img_shape = (args.channels, args.img_size, args.img_size)
        self.model = nn.Sequential(
            nn.Conv2d(args.channels, 2*args.z_dim, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(2*args.z_dim, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(2*args.z_dim, 4*args.z_dim, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(4*args.z_dim, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(4*args.z_dim, 8*args.z_dim, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(8*args.z_dim, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.output = nn.Sequential(
            nn.Conv2d(8*args.z_dim, 1, kernel_size=4, stride=1, padding=0),
        )
    def forward(self, img):
        x = self.model(img)
        return self.output(x)

class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        print('Using default generator')
        self.args = args
        self.img_shape = (args.channels, args.img_size, args.img_size)
        self.model = nn.Sequential(
            nn.ConvTranspose2d(args.latent_dim, 8*args.z_dim, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(8*args.z_dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(8*args.z_dim, 4*args.z_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(4*args.z_dim),
            
            nn.ConvTranspose2d(4*args.z_dim, 2*args.z_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(2*args.z_dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(2*args.z_dim, args.channels, kernel_size=4, stride=2, padding=1),
        )
        self.output = nn.Tanh()
    def forward(self, z):
        img = self.model(z)
        return self.output(img)

class DENASLayer(nn.Module):
    def __init__(self, layer_id, in_planes, out_planes):
        super(DENASLayer, self).__init__()

        self.layer_id = layer_id
        self.in_planes = int(in_planes)
        self.out_planes = int(out_planes)

        self.branch_0 = ConvBranch(self.in_planes, self.out_planes, 2, 2, 0)
        self.branch_1 = ConvBranch(self.in_planes, self.out_planes, 4, 2, 1)
        self.branch_2 = ConvBranch(self.in_planes, self.out_planes, 6, 2, 2)
        self.branch_3 = ConvBranch(self.in_planes, self.out_planes, 8, 2, 3)


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
        elif layer_type == 2:
            out = self.branch_2(x)
        elif layer_type == 3:
            out = self.branch_3(x)
        else:
            raise ValueError("Unknown layer_type {}".format(layer_type))

        #for i, skip in enumerate(skip_indices):
        #    if skip == 1:
        #        out += prev_layers[i]

        return out


class DFixedLayer(nn.Module):
    def __init__(self, layer_id, in_planes, out_planes, sample_arc):
        super(DFixedLayer, self).__init__()

        self.layer_id = layer_id
        self.in_planes = int(in_planes)
        self.out_planes = int(out_planes)
        self.sample_arc = sample_arc

        self.layer_type = sample_arc[0]
        if self.layer_id > 0:
            self.skip_indices = sample_arc[1]
        else:
            self.skip_indices = torch.zeros(1)

        if self.layer_type == 0:
            self.branch = ConvBranch(self.in_planes, self.out_planes, 2, 2, 0)
        elif self.layer_type == 1:
            self.branch = ConvBranch(self.in_planes, self.out_planes, 4, 2, 1)
        elif self.layer_type == 2:
            self.branch = ConvBranch(self.in_planes, self.out_planes, 6, 2, 2)
        elif self.layer_type == 3:
            self.branch = ConvBranch(self.in_planes, self.out_planes, 8, 2, 3)
        else:
            raise ValueError("Unknown layer_type {}".format(self.layer_type))

        # Use concatentation instead of addition in the fixed layer for some reason
        in_planes = int((torch.sum(self.skip_indices).item() + 1) * in_planes)
        self.dim_reduc = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(out_planes, track_running_stats=False)
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


class ConvBranch(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding):
        super(ConvBranch, self).__init__()
        self.in_planes = int(in_planes)
        self.out_planes = int(out_planes)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)

        self.out_conv = nn.Sequential(
            nn.Conv2d(self.in_planes, self.out_planes, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding),
            nn.InstanceNorm2d(self.out_planes, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        out = self.out_conv(x)
        return out

class Discriminator2(nn.Module):
    def __init__(self, args):
        super(Discriminator2, self).__init__()
        print('Using ENAS Discriminator')

        self.num_layers = args.child_num_layers
        self.num_branches = args.child_num_branches
        self.out_filters = 2*args.z_dim
        self.fixed_arc = args.fixed_arc

        self.stem_conv = nn.Sequential(
            nn.Conv2d(args.channels, self.out_filters, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(self.out_filters, affine=True, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.layers = nn.ModuleList([])
        for layer_id in range(self.num_layers):
            if self.fixed_arc == False:
                layer = DENASLayer(layer_id, self.out_filters, self.out_filters*2)
            else:
                layer = DFixedLayer(layer_id, self.out_filters, self.out_filters*2, self.fixed_arc[str(layer_id)])
            self.layers.append(layer)
            self.out_filters *= 2

        self.leaf_conv = nn.Conv2d(int(self.out_filters), 1, kernel_size=4, stride=1, padding=0)

        #for m in self.modules():
        #    if isinstance(m, nn.ConvTranspose2d):
        #        nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x, sample_arc):
        x = self.stem_conv(x)

        prev_layers = []
        for layer_id in range(self.num_layers):
            x = self.layers[layer_id](x, prev_layers, sample_arc[str(layer_id)])
            prev_layers.append(x)

        out = self.leaf_conv(x)
        
        return out

class GENASLayer(nn.Module):
    def __init__(self, layer_id, in_planes, out_planes):
        super(GENASLayer, self).__init__()

        self.layer_id = layer_id
        self.in_planes = int(in_planes)
        self.out_planes = int(out_planes)

        self.branch_0 = ConvTransposeBranch(self.in_planes, self.out_planes, 2, 2, 0)
        self.branch_1 = ConvTransposeBranch(self.in_planes, self.out_planes, 4, 2, 1)
        self.branch_2 = ConvTransposeBranch(self.in_planes, self.out_planes, 6, 2, 2)
        self.branch_3 = ConvTransposeBranch(self.in_planes, self.out_planes, 8, 2, 3)

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
        elif layer_type == 2:
            out = self.branch_2(x)
        elif layer_type == 3:
            out = self.branch_3(x)
        else:
            raise ValueError("Unknown layer_type {}".format(layer_type))

        #for i, skip in enumerate(skip_indices):
        #    if skip == 1:
        #        out += prev_layers[i]

        return out


class GFixedLayer(nn.Module):
    def __init__(self, layer_id, in_planes, out_planes, sample_arc):
        super(GFixedLayer, self).__init__()

        self.layer_id = layer_id
        self.in_planes = int(in_planes)
        self.out_planes = int(out_planes)
        self.sample_arc = sample_arc

        self.layer_type = sample_arc[0]
        if self.layer_id > 0:
            self.skip_indices = sample_arc[1]
        else:
            self.skip_indices = torch.zeros(1)

        if self.layer_type == 0:
            self.branch = ConvTransposeBranch(self.in_planes, self.out_planes, 2, 2, 0)
        elif self.layer_type == 1:
            self.branch = ConvTransposeBranch(self.in_planes, self.out_planes, 4, 2, 1)
        elif self.layer_type == 2:
            self.branch = ConvTransposeBranch(self.in_planes, self.out_planes, 6, 2, 2)
        elif self.layer_type == 3:
            self.branch = ConvTransposeBranch(self.in_planes, self.out_planes, 8, 2, 3)
        else:
            raise ValueError("Unknown layer_type {}".format(self.layer_type))

        # Use concatentation instead of addition in the fixed layer for some reason
        in_planes = int((torch.sum(self.skip_indices).item() + 1) * in_planes)
        self.dim_reduc = nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(out_planes, track_running_stats=False)
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


class ConvTransposeBranch(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding):
        super(ConvTransposeBranch, self).__init__()
        self.in_planes = int(in_planes)
        self.out_planes = int(out_planes)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)

        self.out_conv = nn.Sequential(
            nn.ConvTranspose2d(self.in_planes, self.out_planes, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding),
            nn.BatchNorm2d(self.out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        out = self.out_conv(x)
        return out

class Generator2(nn.Module):
    def __init__(self, args):
        super(Generator2, self).__init__()
        print('Using ENAS Generator')

        self.num_layers = args.child_num_layers
        self.num_branches = args.child_num_branches
        self.out_filters = (2**(self.num_layers+1))*args.z_dim
        self.fixed_arc = args.fixed_arc

        self.stem_deconv = nn.Sequential(
            nn.ConvTranspose2d(args.latent_dim, self.out_filters, kernel_size=4, stride=1, padding=self.num_layers-2, bias=False),
            nn.BatchNorm2d(self.out_filters, track_running_stats=False),
            nn.ReLU()
        )

        self.layers = nn.ModuleList([])
        for layer_id in range(self.num_layers):
            if self.fixed_arc == False:
                layer = GENASLayer(layer_id, self.out_filters, self.out_filters/2)
            else:
                layer = GFixedLayer(layer_id, self.out_filters, self.out_filters/2, self.fixed_arc[str(layer_id)])
            self.layers.append(layer)
            self.out_filters /= 2

        self.leaf_deconv = nn.ConvTranspose2d(int(self.out_filters), args.channels, kernel_size=4, stride=2, padding=1)
        self.generate = nn.Tanh() 

        #for m in self.modules():
        #    if isinstance(m, nn.ConvTranspose2d):
        #        nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x, sample_arc):
        x = self.stem_deconv(x)

        prev_layers = []
        for layer_id in range(self.num_layers):
            x = self.layers[layer_id](x, prev_layers, sample_arc[str(layer_id)])
            prev_layers.append(x)

        x = self.leaf_deconv(x)
        out = self.generate(x)
        
        return out
