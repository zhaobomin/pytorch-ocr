# -*- coding:utf-8 -*-
import os
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import optim

from model_ctpn.text_model import RPN_CLS_Loss, RPN_REGR_Loss
from model_ctpn import darknet_model
from model_ctpn.helper.dataset_ctpn import VOCDataset
from model_ctpn.helper.image import reshape_tensor

random_seed = 2019
torch.random.manual_seed(random_seed)
np.random.seed(random_seed)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def init_dataloader(img_dir, label_dir, num_workers=1):
    dataset = VOCDataset(img_dir, label_dir)
    dataloader = DataLoader(dataset, batch_size=1,
                            shuffle=True, num_workers=num_workers)
    return dataloader


def init_model(cfg, pretrained=''):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = darknet_model.Darknet('weights/text/text.cfg').to(device)
    if pretrained == "" or pretrained == 'init':
        model.apply(weights_init)
    else:
        print('weights pretrained:%s' % pretrained)
        model.load_state_dict(torch.load(pretrained, map_location=device))

    critetion_cls = RPN_CLS_Loss(device)
    critetion_regr = RPN_REGR_Loss(device)

    return model, critetion_cls, critetion_regr, device


def init_optimizer(model, lr=0.001):
    #optimizer = optim.Adam(model.parameters(), lr=lr)
    #optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
    #optimizer = optim.Adadelta(model.parameters())
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    return optimizer, scheduler


def train(img_dir, label_dir, model_cfg='weights/text/text.cfg', pretrained='weights/text/text.pth', batch_size=4, epochs=1000,  lr=0.001):

    train_dataloader = init_dataloader(img_dir, label_dir)
    model, critetion_cls, critetion_regr, device = init_model(
        model_cfg, pretrained)
    optimizer, scheduler = init_optimizer(model, lr=lr)

    batch_total_loss = 0
    n_iter = 0
    epoch_size = len(train_dataloader) // 1

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs))

        model.train()
        epoch_loss_cls = 0
        epoch_loss_regr = 0
        epoch_loss = 0
        # scheduler.step(epoch)

        for batch_i, (imgs, clss, regrs) in enumerate(train_dataloader):
            # continue
            since = time.time()
            imgs = imgs.to(device)
            clss = clss.to(device)
            regrs = regrs.to(device)

            out = model(imgs)
            out_cls, out_regr = reshape_tensor(
                out[:, :20, ...]), reshape_tensor(out[:, 20:, ...])

            # regrs = [N, anchor_nums, 3], clss = [N, 1, anchor_nums]
            # out_regr = [N, anchor_nums, 2], out_cls = [N, anchor_nums, 2]

            loss_regr = critetion_regr(out_regr, regrs)
            loss_cls = critetion_cls(out_cls, clss)

            loss = loss_cls + loss_regr

            epoch_loss_cls += loss_cls.item()
            epoch_loss_regr += loss_regr.item()
            epoch_loss += loss.item()

            mmp = batch_i + 1

            n_iter += 1
            batch_total_loss += loss

            if n_iter % batch_size == 0:
                batch_total_loss = batch_total_loss/batch_size

                print('time:{}'.format(time.time() - since))
                print('EPOCH:{}/{}--BATCH:{}/{}\n'.format(epoch, epochs-1, batch_i, epoch_size),
                      'batch: loss_cls:{:.4f}--loss_regr:{:.4f}--loss:{:.4f}\n'.format(
                    loss_cls.item(), loss_regr.item(), loss.item()),
                    'epoch: loss_cls:{:.4f}--loss_regr:{:.4f}--loss:{:.4f}\n'.format(
                    epoch_loss_cls/mmp, epoch_loss_regr/mmp, epoch_loss/mmp),
                    'batch_total_loss:%.6f' % batch_total_loss
                )
                optimizer.zero_grad()
                batch_total_loss.backward()
                optimizer.step()
                batch_total_loss = 0

        epoch_loss_cls /= epoch_size
        epoch_loss_regr /= epoch_size
        epoch_loss /= epoch_size
        print('Epoch:{}--{:.4f}--{:.4f}--{:.4f}'.format(epoch,
                                                        epoch_loss_cls, epoch_loss_regr, epoch_loss))
        if epoch % 20 == 0:
            print('====save weights: weights/text/text_training.pth=====')
            torch.save(model.state_dict(), 'weights/text/text_training.pth')

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':

    pretrained = sys.argv[1]
    img_dir = './ctpn_dataset/image/'
    label_dir = './ctpn_dataset/voc-label/'
    train(img_dir, label_dir, model_cfg='weights/text/text.cfg',
          pretrained=pretrained, batch_size=1, epochs=1000000,  lr=0.001)
