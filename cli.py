import pickle
from collections import defaultdict
import pandas as pd
import os
from clize import run
import time
import numpy as np
from skimage.io import imsave

from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from torch.nn.init import xavier_uniform

from machinedesign.transformers import ColorDiscretizer
from machinedesign.viz import grid_of_images_default
from machinedesign.viz import horiz_merge

cudnn.benchmark = True

DATA_PATH = '/home/mcherti/work/data'
def data_path(folder):
    return os.path.join(DATA_PATH, folder)

def acc(pred, true_classes):
    _, pred_classes = pred.max(1)
    acc = (pred_classes == true_classes).float().mean()
    return acc


class AE(nn.Module):

    def __init__(self, nc=1, no=1, ndf=64, w=64, nb_blocks=None):
        super().__init__()
        self.ndf = ndf
        self.w = w
        
        if nb_blocks is None:
            nb_blocks = int(np.log(w)/np.log(2)) - 3
        nf = ndf 
        layers = [
            nn.Conv2d(nc, nf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        for _ in range(nb_blocks):
            layers.extend([
                nn.Conv2d(nf, nf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(nf * 2),
                nn.LeakyReLU(0.2, inplace=True),
            ])
            nf = nf * 2
        self.encode = nn.Sequential(*layers)
        layers = []
        for _ in range(nb_blocks):
            layers.extend([
                nn.ConvTranspose2d(nf, nf // 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(nf // 2),
                nn.ReLU(True),
            ])
            nf = nf // 2
        layers.append(
            nn.ConvTranspose2d(nf,  no, 4, 2, 1, bias=False)
        )
        self.decode = nn.Sequential(*layers)
        self.apply(weights_init)

    def forward(self, input):
        x = self.encode(input)
        xrec = self.decode(x)
        return xrec


def train(*, folder='out', dataset='shoes', resume=False, discretize=False):
    lr = 1e-4
    batch_size = 64
    train = dset.ImageFolder(root=data_path(dataset),
        transform=transforms.Compose([
        transforms.Scale(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
     ]))
    trainl = torch.utils.data.DataLoader(
        train, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=1
    )
    if discretize:
        if resume:
            with open('{}/discretizer.pkl'.format(folder), 'rb') as fd:
                color_discretizer = pickle.load(fd)
        else:
            nb_centers = 100
            color_discretizer = ColorDiscretizer(nb_centers=nb_centers)
            for i, (X, _) in enumerate(trainl):
                X = X.numpy()
                color_discretizer.partial_fit(X)
                if i == 10:
                    break
            with open('{}/discretizer.pkl'.format(folder), 'wb') as fd:
                pickle.dump(color_discretizer, fd)
    x0, _ = train[0]
    nc = x0.size(0)
    width = x0.size(2)
    no = 256
    nc = 3
    crit = torch.nn.CrossEntropyLoss()
    if resume:
        ae = torch.load('{}/ae.th'.format(folder))
    else:
        ae = AE(nc=1, no=nb_centers if discretize else no * nc, w=width)
        ae.apply(weights_init)
    ae = ae.cuda()
    print(ae)
    opt = optim.Adam(ae.parameters(), lr=lr, betas=(0.5, 0.999))
    nb_epochs = 20000
    avg_loss = 0.
    avg_precision = 0.
    nb_updates = 0
    stats = defaultdict(list)
    t0 = time.time()
    for epoch in range(nb_epochs):
        for X, _ in trainl:
            X_gray = X.mean(1, keepdim=True)

            if discretize:
                X_col = color_discretizer.transform(X.numpy())
                X_col_orig = color_discretizer.inverse_transform(X_col)
                X_col = X_col.transpose((0, 3, 2, 1))
                X_col = X_col.argmax(axis=3)
                X_col = torch.from_numpy(X_col).long()
                X_col = X_col.view(-1)
            else:
                X_col = (X * 255).long()
                X_col_orig = X_col
                X_col = X_col.transpose(1, 3)
                X_col = X_col.contiguous()
                X_col = X_col.view(-1)

            X_gray = Variable(X_gray).cuda()
            X_col = Variable(X_col).cuda()

            ae.zero_grad()
            X_rec = ae(X_gray)
            if discretize:
                X_rec_orig = X_rec
            else:
                X_rec_orig = X_rec.view(X_rec.size(0), nc, no, X_rec.size(2), X_rec.size(3))
            X_rec = X_rec.transpose(1, 3)
            X_rec = X_rec.contiguous()
            if discretize:
                X_rec = X_rec.view(-1, nb_centers)
            else:
                X_rec = X_rec.view(X_rec.size(0), X_rec.size(1), X_rec.size(2), nc, no)
                X_rec = X_rec.view(-1, no)
            loss = crit(X_rec, X_col)
            loss.backward()
            opt.step()
            precision = acc(X_rec, X_col)
            avg_loss = avg_loss * 0.9 + loss.data[0] * 0.1
            avg_precision = avg_precision * 0.9 + precision.data[0] * 0.1
            stats['precision'].append(precision.data[0])
            stats['loss'].append(loss.data[0])
            if nb_updates % 100 == 0:
                pd.DataFrame(stats).to_csv('{}/stats.csv'.format(folder))
                dt = time.time() - t0
                t0 = time.time()
                print('Epoch {:03d}/{:03d}, Avg loss : {:.6f}, Avg precision : {:.6f}, dt : {:.6f}(s)'.format(epoch + 1, nb_epochs, avg_loss, avg_precision, dt))
                im = X_rec_orig.data.cpu().numpy()
                if discretize:
                    im = color_discretizer.inverse_transform(im)
                    im = im.astype('float32')
                else:
                    im = im.argmax(axis=2)
                    im = im.astype('float32')
                im_left = grid_of_images_default(im, normalize=True)
                im = X_col_orig
                im = im.astype('float32')
                im_right = grid_of_images_default(im, normalize=True)
                im = horiz_merge(im_left, im_right)
                imsave('{}/ae.png'.format(folder), im)
            nb_updates += 1
        torch.save(ae, '{}/ae.th'.format(folder))


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname == 'Linear':
        xavier_uniform(m.weight.data)
        m.bias.data.fill_(0)


if __name__ == '__main__':
    run([train])
