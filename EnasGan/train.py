import os
import sys
import glob
import logging
import random
import time
import argparse
import numpy as np
import pickle
from datetime import timedelta

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import save_image

import utils
from controller import Controller
from shared_cnn import Discriminator, Generator, Discriminator2, Generator2
from shared_gan import Discriminator3, Generator3

parser = argparse.ArgumentParser(description='ENAS')

parser.add_argument('--search_for', default='macro', choices=['macro'])
parser.add_argument('--data_path', default='/research/dept2/xbzheng/code/enas_gan/data/', type=str)
parser.add_argument('--output_filename', default='ENAS', type=str)
parser.add_argument('--resume', default='', type=str)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_epochs', type=int, default=310)
parser.add_argument('--log_every', type=int, default=50)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--fixed_arc', action='store_true', default=False)
parser.add_argument('--save', default='EXP', help='name of experiment')
parser.add_argument('--gpu', type=int, default=0)

parser.add_argument('--child_num_layers', type=int, default=2)
parser.add_argument('--child_num_branches', type=int, default=2)
parser.add_argument('--child_grad_bound', type=float, default=5.0)

#gan parameters
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--b1', type=float, default=0.5)
parser.add_argument('--b2', type=float, default=0.999)
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, default=32)
parser.add_argument('--channels', type=int, default=3)
parser.add_argument('--z_dim', type=int, default=128)
parser.add_argument('--grad_lambda', type=int, default=10)
parser.add_argument('--trainD', action='store_true', default=False)
parser.add_argument('--trainG', action='store_true', default=False)
parser.add_argument('--train3', action='store_true', default=False)
parser.add_argument('--useclass', default='')
parser.add_argument('--repeat_train', type=int,default=1)

parser.add_argument('--controller_lstm_size', type=int, default=64)
parser.add_argument('--controller_lstm_num_layers', type=int, default=1)
parser.add_argument('--controller_entropy_weight', type=float, default=0.0001)
parser.add_argument('--controller_train_every', type=int, default=1)
parser.add_argument('--controller_num_aggregate', type=int, default=10)
parser.add_argument('--controller_train_steps', type=int, default=30)
parser.add_argument('--controller_lr', type=float, default=0.001)
parser.add_argument('--controller_tanh_constant', type=float, default=1.5)
parser.add_argument('--controller_op_tanh_reduce', type=float, default=2.5)
parser.add_argument('--controller_skip_target', type=float, default=0.4)
parser.add_argument('--controller_skip_weight', type=float, default=0.8)
parser.add_argument('--controller_bl_dec', type=float, default=0.99)

class ElapsedFormatter():
    def __init__(self):
        self.start_time = time.time()
    def format(self, record):
        elapsed_seconds = record.created - self.start_time
        elapsed = timedelta(seconds=elapsed_seconds)
        return "{} {}".format(elapsed, record.getMessage())

class CIFAR10Filtered(datasets.CIFAR10):
    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False, classes=[]):

        super(CIFAR10Filtered, self).__init__(root)
        self.transform = transform
        self.target_transform = target_transform

        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()
        
        if len(classes) != 0:
            self.data = np.array(self.data)
            self.targets = np.array(self.targets)
            self.data = self.data[np.isin(self.targets, classes)]
            self.targets = self.targets[np.isin(self.targets, classes)]
            print(classes)
            print(np.isin(self.targets, classes))
    


args = parser.parse_args()
print(args)

# seed and gpu
random.seed(args.seed)
np.random.seed(args.seed)
torch.cuda.set_device(args.gpu)
torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# save and log
args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(ElapsedFormatter())
logging.getLogger().addHandler(fh)

# dataset
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

if args.useclass == '':
    print('Use all dataset')
    train_dataset = datasets.CIFAR10(root=args.data_path, train=True, transform=train_transform, download=True)
else:
    print('Use part of dataset')
    classes = list(map(int, args.useclass.split(',')))
    train_dataset = CIFAR10Filtered(root=args.data_path, train=True, transform=train_transform, download=True, classes=classes)
data_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)
print(len(data_loader))


# models
controller = Controller(search_for=args.search_for,
                        search_whole_channels=True,
                        num_layers=args.child_num_layers,
                        num_branches=args.child_num_branches,
                        lstm_size=args.controller_lstm_size,
                        lstm_num_layers=args.controller_lstm_num_layers,
                        tanh_constant=args.controller_tanh_constant,
                        temperature=None,
                        skip_target=args.controller_skip_target,
                        skip_weight=args.controller_skip_weight)
controller = controller.cuda()
if args.trainD and args.train3:
    netD = Discriminator3(args)
elif args.trainD:
    netD = Discriminator2(args)
else:
    netD = Discriminator(args)
netD = netD.cuda()
if args.trainG and args.train3:
    netG = Generator3(args)
elif args.trainG:
    netG = Generator2(args)
else:
    netG = Generator(args)
netG = netG.cuda()

# https://github.com/melodyguan/enas/blob/master/src/utils.py#L218
controller_optimizer = torch.optim.Adam(params=controller.parameters(),
                                        lr=args.controller_lr,
                                        betas=(0.0, 0.999),
                                        eps=1e-3)

optimizerD = torch.optim.Adam(netD.parameters(), lr=args.lr, betas=(args.b1, args.b2))
optimizerG = torch.optim.Adam(netG.parameters(), lr=args.lr, betas=(args.b1, args.b2))

def train_gan(epoch, fixed_arc=None):
    global args
    global netD
    global netG
    global controller
    global optimizerD
    global optimizerG

    controller.eval()

    one = torch.FloatTensor([1])
    mone = one * -1
    one = one.cuda()
    mone = mone.cuda()
    
    for j in range(args.repeat_train):
        for i, (images, labels) in enumerate(data_loader):
            if images.shape[0] != args.batch_size:
                continue
            start = time.time()
            images = images.cuda()

            if fixed_arc is None:
                with torch.no_grad():
                    controller()  # perform forward pass to generate a new architecture
                sample_arc = controller.sample_arc
            else:
                sample_arc = fixed_arc
            
            ######################################
            # (1) Update D network
            ######################################
            for p in netD.parameters():
                p.requires_grad=True
        
            optimizerD.zero_grad()
        
            d_loss_real = netD(images, sample_arc) if args.trainD else netD(images)
            d_loss_real = d_loss_real.mean()
            d_loss_real.backward(mone)
        
            z = torch.randn(args.batch_size, args.latent_dim, 1, 1).cuda()
            fake_images = netG(z, sample_arc) if args.trainG else netG(z)
            d_loss_fake = netD(fake_images, sample_arc) if args.trainD else netD(fake_images)
            d_loss_fake = d_loss_fake.mean()
            d_loss_fake.backward(one)
        
            gradient_penalty = utils.calc_gradient_peanlty(netD, images.data, fake_images.data, args, sample_arc)
            gradient_penalty.backward(retain_graph=True)
        
            d_loss = d_loss_fake - d_loss_real + gradient_penalty
            WassersteinD = d_loss_real - d_loss_fake
        
            optimizerD.step()
        
            ######################################
            # (2) Update G network
            ######################################
            for p in netD.parameters():
                p.requires_grad = False
            optimizerG.zero_grad()
        
            z = torch.randn(args.batch_size, args.latent_dim, 1, 1).cuda()
            fake_images = netG(z, sample_arc) if args.trainG else netG(z)
            g_loss = netD(fake_images, sample_arc) if args.trainD else netD(fake_images)
            g_loss = g_loss.mean()
            g_loss.backward(mone)
            g_cost = -g_loss
        
            optimizerG.step()
            if i % args.log_every == 0:
                logging.info('[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]' % (epoch, args.num_epochs, i, len(data_loader), d_loss.item(), g_loss.item()))
            if i + 2 == len(data_loader):
                #inception_score = utils.get_inception_score(netG, sample_arc, args, 50)
                #logging.info('inception score {}'.format(str(inception_score)))
                save_image(fake_images.data[:25], os.path.join(args.save,'samples/samples_%d.png' % epoch), nrow=5, normalize=True)
            
    controller.train()


def train_controller(epoch, baseline=None):
    global args
    global netG
    global controller
    global controller_optimizer

    logging.info('Epoch ' + str(epoch) + ': Training controller')

    netG.eval()

    controller.zero_grad()
    for i in range(args.controller_train_steps * args.controller_num_aggregate):
        start = time.time()

        controller()  # perform forward pass to generate a new architecture
        sample_arc = controller.sample_arc
        
        with torch.no_grad():
            score, _ = utils.get_inception_score(netG, sample_arc, args, 50)
        
        # detach to make sure that gradients aren't backpropped through the reward
        reward = score/12
        reward += args.controller_entropy_weight * controller.sample_entropy

        if baseline is None:
            baseline = score
        else:
            baseline -= (1 - args.controller_bl_dec) * (baseline - reward)
            # detach to make sure that gradients are not backpropped through the baseline
            baseline = baseline.detach()

        loss = -1 * controller.sample_log_prob * (reward - baseline)

        if args.controller_skip_weight is not None:
            loss += args.controller_skip_weight * controller.skip_penaltys

        # Average gradient over controller_num_aggregate samples
        loss = loss / args.controller_num_aggregate

        loss.backward(retain_graph=True)

        end = time.time()

        # Aggregate gradients for controller_num_aggregate iterationa, then update weights
        if (i + 1) % args.controller_num_aggregate == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(controller.parameters(), args.child_grad_bound)
            controller_optimizer.step()
            controller.zero_grad()

            if (i + 1) % (2 * args.controller_num_aggregate) == 0:
                learning_rate = controller_optimizer.param_groups[0]['lr']
                display = 'ctrl_step=' + str(i // args.controller_num_aggregate) + \
                          '\tloss=%.3f' % (loss.item()) + \
                          '\tent=%.2f' % (controller.sample_entropy.item()) + \
                          '\tlr=%.4f' % (learning_rate) + \
                          '\t|g|=%.4f' % (grad_norm) + \
                          '\tacc=%.4f' % (score) + \
                          '\tbl=%.2f' % (baseline) + \
                          '\ttime=%.2fit/s' % (1. / (end - start))
                logging.info(display)

    netG.train()
    return baseline


def get_best_arc(n_samples=10, verbose=False):
    global args
    global controller
    global netG
    
    controller.eval()
    netG.eval()

    arcs = []
    inception_scores = []
    for i in range(n_samples):
        with torch.no_grad():
            controller()  # perform forward pass to generate a new architecture
        sample_arc = controller.sample_arc
        arcs.append(sample_arc)

        with torch.no_grad():
            score, _ = utils.get_inception_score(netG, sample_arc, args, 50)
        inception_scores.append(score)

        if verbose:
            print_arc(sample_arc)
            logging.info('score=' + str(score))
            logging.info('-' * 80)

    best_iter = np.argmax(inception_scores)
    best_arc = arcs[best_iter]
    best_score = inception_scores[best_iter]

    controller.train()
    netG.train()
    return best_arc, best_score

def print_arc(sample_arc):
    for key, value in sample_arc.items():
        if len(value) == 1:
            branch_type = value[0].cpu().numpy().tolist()
            logging.info('[' + ' '.join(str(n) for n in branch_type) + ']')
        else:
            branch_type = value[0].cpu().numpy().tolist()
            skips = value[1].cpu().numpy().tolist()
            logging.info('[' + ' '.join(str(n) for n in (branch_type + skips)) + ']')


def train_enas(start_epoch):
    global args
    global netD
    global netG
    global controller
    global optimizerD
    global optimizerG
    global controller_optimizer
    
    baseline = None
    for epoch in range(start_epoch, args.num_epochs):
        train_gan(epoch)
        baseline = train_controller(epoch, baseline)

        n_samples = 10
        logging.info('Here are ' + str(n_samples) + ' architectures:')
        _ = get_best_arc(n_samples, verbose=True)

        state = {'epoch': epoch + 1,
                 'args': args,
                 'netD_state_dict': netD.state_dict(),
                 'netG_state_dict': netG.state_dict(),
                 'controller_state_dict': controller.state_dict(),
                 'optimizerD': optimizerD.state_dict(),
                 'optimizerG': optimizerG.state_dict(),
                 'controller_optimizer': controller_optimizer.state_dict()}
        filename = args.save + '/checkpoints/' + args.output_filename + '.pth.tar'
        torch.save(state, filename)


def train_fixed(start_epoch):
    global args
    global netD
    global netG
    global controller
    global optimizerD
    global optimizerG
    global controller_optimizer

    best_arc, best_score = get_best_arc(n_samples=100, verbose=False)
    logging.info('Best architecture:')
    print_arc(best_arc)
    logging.info('Inception score: ' + str(best_score))
    
    if args.trainD:
        netD = Discriminator3(args) if args.train3 else Discriminator2(args)
        netD = netD.cuda()
    if args.trainG:
        netG = Generator3(args) if args.train3 else Generator2(args)
        netG = netG.cuda()
    optimizerD = torch.optim.Adam(netD.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    for epoch in range(args.num_epochs):
        train_gan(epoch, best_arc)

        state = {'epoch': epoch + 1,
                 'args': args,
                 'best_arc': best_arc,
                 'netD_state_dict': netD.state_dict(),
                 'netG_state_dict': netG.state_dict(),
                 'optimizerD': optimizerD.state_dict(),
                 'optimizerG': optimizerG.state_dict()}
        filename = 'checkpoints/' + args.output_filename + '_fixed.pth.tar'
        torch.save(state, filename)


def main():
    global args
    global netD
    global netG
    global controller
    global optimizerD
    global optimizerG
    global controller_optimizer

    if args.resume:
        if os.path.isfile(args.resume):
            print("Loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            args = checkpoint['args']
            netD.load_state_dict(checkpoint['netD_state_dict'])
            netG.load_state_dict(checkpoint['netG_state_dict'])
            controller.load_state_dict(checkpoint['controller_state_dict'])
            optimizerD.load_state_dict(checkpoint['optimizerD'])
            optimizerG.load_state_dict(checkpoint['optimizerG'])
            controller_optimizer.load_state_dict(checkpoint['controller_optimizer'])
            print("Loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            raise ValueError("No checkpoint found at '{}'".format(args.resume))
    else:
        start_epoch = 0

    if not args.fixed_arc:
        train_enas(start_epoch)
    else:
        assert args.resume != '', 'A pretrained model should be used when training a fixed architecture.'
        train_fixed(start_epoch)


if __name__ == "__main__":
    main()
