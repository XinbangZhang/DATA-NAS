import os
import numpy as np
import torch
import torch.nn.init as init
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable
import math


class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10(args):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return train_transform, valid_transform


def count_parameters_in_MB(model):
    # for name, v in model.named_parameters():
    # if 'cells.0._ops.0' in name:
    # print name, np.sum(np.prod(v.size()))
    # print name, v.size()
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name) / 1e6


def count_flops_in_MB(model):
    # for name, v in model.named_parameters():
    #     if 'cells.0._ops.0' in name:
    #         print name, np.sum(np.prod(v.size()))
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name) / 1e6


def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)


def save(model, model_path):
    torch.save(model.state_dict(), model_path)


def load(model, model_path):
    model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
        x.div_(keep_prob)
        x.mul_(mask)
    return x


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        print(path)
        os.mkdir(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        # init.kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='relu')
        init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='relu')
        # init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        # init.kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='relu')
        init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='relu')
        # init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)


class MyCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, T_max, eta_min=0, warmup_factor=1.0 / 3, warmup_epochs=5, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        self.warmup_factor = warmup_factor
        self.warmup_epochs = warmup_epochs
        super(MyCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            alpha = float(self.last_epoch) / (self.warmup_epochs)
            warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            return [self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(
                math.pi * (self.last_epoch - self.warmup_epochs) / (self.T_max - self.warmup_epochs))) / 2
                    for base_lr in self.base_lrs]

    # def get_lr(self):
    #     warmup_factor = 1
    #     if self.last_epoch < self.warmup_epochs:
    #         if self.warmup_method == "constant":
    #             warmup_factor = self.warmup_factor
    #         elif self.warmup_method == "linear":
    #             alpha = float(self.last_epoch) / self.warmup_epochs
    #             warmup_factor = self.warmup_factor * (1 - alpha) + alpha
    #     return [
    #         base_lr
    #         * warmup_factor
    #         * self.gamma ** bisect_right(self.milestones, self.last_epoch)
    #         for base_lr in self.base_lrs
    #     ]
