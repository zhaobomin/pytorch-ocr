from __future__ import print_function
from __future__ import division

import argparse
import random
import torch
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
from torch.nn import CTCLoss
import os
from helper import dataset_crnn

from models import darknet_model
import config

# 1.  创建数据集：1. cd crnn_dataset 2. python3 create_dataset.py train_val.txt test_val.txt
# 2. 训练模型：
# python3 train_crnn.py   --trainRoot crnn_dataset/train/ --valRoot crnn_dataset/test/

parser = argparse.ArgumentParser()
parser.add_argument('--trainRoot', default='crnn_dataset/train/',
                    help='path to dataset')
parser.add_argument(
    '--valRoot', default='crnn_dataset/test/',  help='path to dataset')
parser.add_argument('--workers', type=int,
                    help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int,
                    default=4, help='input batch size')
parser.add_argument('--imgH', type=int, default=32,
                    help='the height of the input image to network')

parser.add_argument('--nepoch', type=int, default=5000000000,
                    help='number of epochs to train for')

parser.add_argument('--pretrained', default='weights/ocr/chinese/ocr_best.pth',
                    help="path to pretrained model (to continue training)")

parser.add_argument('--lr', type=float, default=0.01,
                    help='learning rate for Critic, not used by adadealta')
parser.add_argument('--beta1', type=float, default=0.5,
                    help='beta1 for adam. default=0.5')
parser.add_argument('--adam', action='store_true',
                    help='Whether to use adam (default is rmsprop)')
parser.add_argument('--adadelta', action='store_true',
                    help='Whether to use adadelta (default is rmsprop)')


def init_dataloader(opt):
    train_dataset = dataset_crnn.lmdbDataset(root=opt.trainRoot)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1,
        shuffle=True,
        num_workers=int(opt.workers),
        collate_fn=dataset_crnn.alignCollate(imgH=opt.imgH, imgW=None, keep_ratio=True))

    test_dataset = dataset_crnn.lmdbDataset(
        root=opt.valRoot, transform=dataset_crnn.resizeNormalize(32))

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, num_workers=int(opt.workers))
    return train_loader, test_loader


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def init_model(ocrCfgPath, alphabet, pretrained=''):
    crnn = darknet_model.Darknet(ocrCfgPath)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    crnn.to(device)
    crnn.apply(weights_init)
    if opt.pretrained != '':
        print('loading pretrained model from %s' % opt.pretrained)
        crnn.load_state_dict(torch.load(pretrained))
    criterion = CTCLoss(blank=len(alphabet)-1)
    criterion.to(device)
    converter = dataset_crnn.strLabelConverter(opt.alphabet)
    return crnn, criterion, converter, device


def init_optimizer(net, opt):
    # setup optimizer
    params = net.parameters()
    opt.adadelta = 1

    if opt.adam:
        print('use optim.Adam')
        optimizer = optim.Adam(params, lr=opt.lr,
                               betas=(opt.beta1, 0.999))
    elif opt.adadelta:
        print('use optim.Adadelta')
        optimizer = optim.Adadelta(params)
    else:
        print('use optim.RMSprop')
        optimizer = optim.RMSprop(params, lr=opt.lr)
    return optimizer


def net_forward(net, criterion, converter, images, texts, device, mode='train'):

    batch_size = images.size(0)
    t, l = converter.encode(texts)
    images = images.to(device)
    t = t.to(device)
    l = l.to(device)

    if mode != 'train':
        net.eval()
    else:
        net.train()
    preds = net(images)
    preds = preds.squeeze(2)  # remove dim(2), h -> [N, 512, 26]
    preds = preds.permute(2, 0, 1)  # [T, b, c] => [26, N, 512]
    preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))

    log_preds = torch.nn.functional.log_softmax(preds, dim=2)

    cost = criterion(log_preds, t, preds_size, l) / batch_size
    return preds, preds_size, cost


def val(net, test_loader, criterion, converter, device, max_iter=100):
    print('Start val')
    n_total = 0
    n_correct = 0
    loss_avg = dataset_crnn.averager()

    for images, texts in test_loader:
        preds, preds_size, cost = net_forward(
            net, criterion, converter, images, texts, device, mode='eval')
        loss_avg.add(cost)

        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)

        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        raw_preds = converter.decode(preds.data, preds_size.data, raw=True)
        n_total += 1
        if sim_preds == texts[0]:
            n_correct += 1
        print('%-20s => %-20s, gt: %-20s' %
              (raw_preds, sim_preds, texts[0]))

        for pred, target in zip(sim_preds, texts):
            if pred == target.lower():
                n_correct += 1

    accuracy = n_correct / float(n_total)
    print('Test loss: %f, accuray: %.2f%%' % (loss_avg.val(), accuracy*100))


def train(opt):

    train_loader, test_loader = init_dataloader(opt)
    net, criterion, converter, device = init_model(
        config.ocrCfgPath, opt.alphabet, opt.pretrained)
    optimizer = init_optimizer(net, opt)

    # loss averager
    loss_avg = dataset_crnn.averager()
    iter_num = 0
    total_loss = 0
    for epoch in range(opt.nepoch):
        for images, texts in train_loader:

            preds, preds_size, cost = net_forward(
                net, criterion, converter, images, texts, device=device)
            total_loss += cost
            iter_num += 1
            if iter_num % opt.batchSize == 0:
                total_loss = total_loss/opt.batchSize
                print('total loss : %.9f' % total_loss)
                print('[%d/%d][%d/%d] Loss: %f' %
                      (epoch, opt.nepoch, iter_num, len(train_loader), loss_avg.val()))

                val(net, test_loader, criterion, converter, device)
                net.zero_grad()
                total_loss.backward()
                optimizer.step()

                total_loss = 0
                loss_avg.reset()

            loss_avg.add(cost)

        if epoch % 1 == 0:
            torch.save(net.state_dict(), opt.save_weights)
            print('[%d/%d][%d/%d] Loss: %.9f' %
                  (epoch, opt.nepoch, iter_num, len(train_loader), loss_avg.val()))
            val(net, train_loader, criterion, converter, device)
            loss_avg.reset()


if __name__ == '__main__':
    opt = parser.parse_args()
    opt.keep_ratio = True
    opt.alphabet, _ = config.get_characters()
    opt.save_weights = 'weights/ocr/chinese/ocr_training.pth'

    train(opt)
