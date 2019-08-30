import os
import numpy as np
import torch
import scipy
from scipy.misc import imsave
import shutil
from torch.autograd import Variable

import inception_score

def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        os.mkdir(os.path.join(path, 'samples'))
        os.mkdir(os.path.join(path, 'checkpoints'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)

def calc_gradient_peanlty(netD, real_data, fake_data, args, arc):
    alpha = torch.FloatTensor(args.batch_size, 1, 1, 1).uniform_(0, 1)
    alpha = alpha.expand(args.batch_size, real_data.size(1), real_data.size(2), real_data.size(3))
    alpha = alpha.cuda()
    interpolates = alpha * real_data + (1 - alpha) * fake_data
    interpolates = interpolates.cuda()
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)
    prob_interpolates = netD(interpolates, arc) if args.trainD else netD(interpolates)
    gradients = torch.autograd.grad(outputs=prob_interpolates, inputs=interpolates,
                                    grad_outputs=torch.ones(prob_interpolates.size()).cuda(),
                                    create_graph=True, retain_graph=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * args.grad_lambda
    return gradient_penalty

def get_inception_score(netG, arc, args, n_iter=500):
    all_samples = []
    for i in range(n_iter):
        with torch.no_grad():
            samples_100 = Variable(torch.randn(100, args.latent_dim, 1, 1)).cuda()
        if args.trainG:
            fake_images = netG(samples_100, arc).mul_(0.5).add_(0.5).mul_(255).clamp_(0, 255).to('cpu', torch.uint8).numpy()
        else:
            fake_images = netG(samples_100).mul_(0.5).add_(0.5).mul_(255).clamp_(0, 255).to('cpu', torch.uint8).numpy()
        all_samples.append(fake_images)
    all_samples = np.concatenate(all_samples, axis=0)
    all_samples = all_samples.reshape((-1, 3, 32, 32))
    return inception_score.get_inception_score(list(all_samples), 100)
