import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

np.set_printoptions(precision=3, suppress=True)
np.set_printoptions(formatter={'all': lambda x: '%02.3f' % x + ','})
from architect import Architect
from model_infer import model_infer_gentype, model_infer_gentype_mod, model_infer_gentype_layer

# os.environ['CUDA_VISIBLE_DEVICES'] = '7'
parser = argparse.ArgumentParser("cifar")
parser.add_argument('--gpu', type=str, default='1', help='gpu device id')
parser.add_argument('--save_epoch', type=int, default='1', help='gpu device id')
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
# parser.add_argument('--learning_rate', type=float, default=0.005, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--epochs', type=int, default=100, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=0, help='weight decay for arch encoding')
parser.add_argument('--act_funciton', type=str, default='relu', choices=['softmax', 'relu'],
                    help='activate function for prob')
parser.add_argument('--sample_times', type=int, default='2', help='sampling times for DATA')
parser.add_argument('--arch_reg', type=bool, default=True, help='whether to apply architecture regulation')

args = parser.parse_args()
if args.act_funciton == 'relu':
    from model_search_relu import Network
else:
    from model_search_softmax import Network
# temp force_hard structure
# args.save = 'log/{}-large-search-softmax-gumbel-1-softmax-sample4i-{}-{}'.format(args.epochs, args.save, time.strftime("%Y%m%d-%H%M%S"))
multi_node_flag = '-multinode' if args.multi_nodes else ''
arch_reg_flag = 'entro' if args.arch_reg else ''
args.save = 'log/{}{}{}-l1-05-{}-gumbel-sample{}-{}-{}-{}'.format(arch_reg_flag, args.act_funciton,
                                                                       multi_node_flag, args.epochs, args.sample_times,
                                                                       args.save, args.weight_decay,
                                                                       time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CIFAR_CLASSES = 10


def main():

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)
    np.random.seed(args.seed)
    # gpus = [int(i) for i in args.gpu.split(',')]
    # if len(gpus) == 1:
    #   torch.cuda.set_device(int(args.gpu))
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %s' % args.gpu)
    logging.info("args = %s", args)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion, sample_times=args.sample_times)
    model = nn.DataParallel(model)
    # weight = torch.load('log/pretrain-nocutout-datapor0.5-epoch600-lr0.025-20190909-230307-M7/weights_550.pt')
    # print {k: v for k, v in weight.items() if k in model.module.state_dict() and 'alphas' not in k}
    # model.module.load_state_dict({k: v for k, v in weight.items() if k in model.module.state_dict() and 'alphas' not in k}, strict=False)
    # model.module.load_state_dict(weight, strict=False)
    model = model.cuda()

    arch_params = list(map(id, model.module.arch_parameters()))
    weight_params = filter(lambda p: id(p) not in arch_params, model.module.parameters())

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(
        weight_params,  # model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    train_transform, valid_transform = utils._data_transforms_cifar10(args)
    train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=False, num_workers=4)

    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=False, num_workers=4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

    architect = Architect(model, criterion, args)

    # warming
    # with torch.no_grad():
    #     valid_acc, valid_obj = infer(valid_queue, model, criterion)

    for epoch in range(args.epochs):
        scheduler.step()
        lr = scheduler.get_lr()[0]
        logging.info('epoch %d lr %e', epoch, lr)

        train_acc, train_obj = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, epochs=epoch)
        logging.info('train_acc %f', train_acc)

        if not args.multi_nodes:
            if args.act_funciton == 'relu':
                alphas_normal = F.relu(model.module.alphas_normal)
                alphas_normal = alphas_normal / (torch.sum(alphas_normal, dim=-1).view(-1, 1) + 1e-7)
                alphas_reduce = F.relu(model.module.alphas_reduce)
                alphas_reduce = alphas_reduce / (torch.sum(alphas_reduce, dim=-1).view(-1, 1) + 1e-7)
            else:
                alphas_normal = F.softmax(model.module.alphas_normal, dim=-1)
                alphas_reduce = F.softmax(model.module.alphas_reduce, dim=-1)
        else:
            alphas_normal = torch.zeros(model.module.alphas_normal.size())
            alphas_reduce = torch.zeros(model.module.alphas_reduce.size())
            n = 2
            start = 0
            for i in range(model.module._steps):
                end = start + n
                weights_norm = model.module.alphas_normal[start:end, :].view(1, -1)
                weights_reduce = model.module.alphas_reduce[start:end, :].view(1, -1)
                weights_size = model.module.alphas_normal[start:end, :].size()
                # multi nodes relu
                if args.act_funciton == 'relu':
                    weights_norm = F.relu(weights_norm)
                    weights_norm = weights_norm / (torch.sum(weights_norm, dim=-1).view(-1, 1) + 1e-7)
                    weights_reduce = F.relu(weights_reduce)
                    weights_reduce = weights_reduce / (torch.sum(weights_reduce, dim=-1).view(-1, 1) + 1e-7)
                # multi nodes softmax
                else:
                    weights_norm = F.softmax(weights_norm, dim=-1)
                    weights_reduce = F.softmax(weights_reduce, dim=-1)

                alphas_normal[start:end, :] = weights_norm.view(weights_size)
                alphas_reduce[start:end, :] = weights_reduce.view(weights_size)

                start = end
                n = n + 1
        # multi nodes softmax
        genotype_infer_mod = model_infer_gentype_mod(alphas_normal, alphas_reduce,
                                                     model.module._steps, args.sample_times)
        # logging.info('genotype = %s', genotype)
        # logging.info('infer genotype = %s', genotype_infer)
        logging.info('infer genotype = %s', genotype_infer_mod)

        # validation
        with torch.no_grad():
            valid_acc, valid_obj = infer(valid_queue, model, criterion)
            valid_acc_infer, valid_obj_infer = infer(valid_queue, model, criterion, arch=genotype_infer)
            valid_acc_infer_mod, valid_obj_infer_mod = infer(valid_queue, model, criterion, arch=genotype_infer_mod)
        logging.info('valid_acc:%f', valid_acc)
        logging.info('valid_acc_infer_origin:%f  gap%f', valid_acc_infer, np.abs(valid_acc_infer - valid_acc))
        logging.info('valid_acc_infer_mod:%f  gap:%f', valid_acc_infer_mod, np.abs(valid_acc_infer_mod - valid_acc))

        if (epoch + 1) % args.save_epoch == 0:
            utils.save(model, os.path.join(args.save, 'weights_{}.pt'.format(epoch + 1)))


def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, epochs=1):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    objs_search = utils.AvgrageMeter()
    top1_search = utils.AvgrageMeter()
    top5_search = utils.AvgrageMeter()
    for step, (input, target) in enumerate(train_queue):

        model.train()
        n = input.size(0)

        input = input.cuda()
        target = target.cuda()
        # get a random minibatch from the search queue with replacement
        input_search, target_search = next(iter(valid_queue))
        input_search = input_search.cuda()
        target_search = target_search.cuda()

        logits_search, loss_search = architect.step(input, target, input_search, target_search, lr, optimizer,
                                                    unrolled=args.unrolled)

        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)

        loss.backward()
        nn.utils.clip_grad_norm_(model.module.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        prec1_search, prec5_search = utils.accuracy(logits_search, target_search, topk=(1, 5))

        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        objs_search.update(loss_search.item(), n)
        top1_search.update(prec1_search.item(), n)
        top5_search.update(prec5_search.item(), n)

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
            logging.info('trval %03d %e %f %f', step, objs_search.avg, top1_search.avg, top5_search.avg)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion, arch=None):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        input = input.cuda()
        target = target.cuda()

        logits = model(input, arch=arch)
        loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


if __name__ == '__main__':
    main()
