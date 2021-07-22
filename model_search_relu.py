import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from genotypes import PRIMITIVES
from genotypes import Genotype
from gumbelmodule import GumbleSoftmax
import numpy as np


class MixedOp(nn.Module):

    def __init__(self, C, stride):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))


class Cell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        self.reduction = reduction

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        for i in range(self._steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride)
                self._ops.append(op)

    def forward(self, s0, s1, weights):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            # s = sum(self._ops[offset + j](h, weights[offset + j]) for j, h in enumerate(states))
            s = sum(self._ops[offset + j](h, weights[offset + j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):

    def __init__(self, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3, sample_times=4):
        super(Network, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier
        self._sample_times = sample_times

        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

        self._initialize_alphas()
        self.GumbleSoftmax = GumbleSoftmax()
        self.norm_w = None
        self.redu_w = None

    def new(self):
        model_new = Network(self._C, self._num_classes, self._layers, self._criterion).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def forward(self, input, epochs=150, arch=None):
        temp = 5 - (5. - 1.) / 150. * epochs

        s0 = s1 = self.stem(input)
        if arch is None:
            # alphas_normal_weight = (self.alphas_normal + 0.5) / (0.5 * self.alphas_normal.shape[1] + 1)
            # alphas_reduce_weight = (self.alphas_reduce + 0.5) / (0.5 * self.alphas_reduce.shape[1] + 1)
            weights_norm = self.gumbel_sample_weight(self.alphas_normal, self._sample_times, temp=temp,
                                                     flops_param=None)
            weights_reduce = self.gumbel_sample_weight(self.alphas_reduce, self._sample_times, temp=temp,
                                                       flops_param=None)
            # print weights_norm
            # geno = self.weight2genotype(weights_norm, weights_reduce)
            # print geno
            # weights_norm, weights_reduce = self.genotype2weight(geno)
        else:
            weights_norm, weights_reduce = self.genotype2weight(arch)
        # print weights_norm
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = weights_reduce
            else:
                weights = weights_norm

            s0, s1 = s1, cell(s0, s1, weights)

        self.weights_norm = torch.zeros(weights_norm.size())
        self.weights_norm[:] = weights_norm[:]
        self.weights_norm = nn.Parameter(self.weights_norm, requires_grad=False).cuda()

        self.weights_reduce = torch.zeros(weights_reduce.size())
        self.weights_reduce[:] = weights_reduce[:]
        self.weights_reduce = nn.Parameter(self.weights_reduce, requires_grad=False).cuda()

        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(PRIMITIVES)

        self.alphas_normal = nn.Parameter(1e-3 * torch.randn(k, num_ops) + torch.ones(k, num_ops))
        self.alphas_reduce = nn.Parameter(1e-3 * torch.randn(k, num_ops) + torch.ones(k, num_ops))
        self.weights_norm = None
        self.weights_reduce = None
        # self.weights_norm = nn.Parameter(1e-3 * torch.randn(k, num_ops), requires_grad=False)
        # self.weights_reduce = nn.Parameter(1e-3 * torch.randn(k, num_ops), requires_grad=False)
        # self.alphas_normal = nn.Parameter(1e-3 * torch.zeros(k, num_ops))
        # self.alphas_reduce = nn.Parameter(1e-3 * torch.zeros(k, num_ops))
        self.alphas_flops = np.zeros((k, num_ops), dtype=np.float32)
        self.alphas_flops[:, 4:] = 1
        self.alphas_flops = nn.Parameter(torch.from_numpy(self.alphas_flops), requires_grad=False)
        # self.alphas_flops = torch.from_numpy(self.alphas_flops)
        # self.alphas_normal = nn.Parameter(1e-1 * torch.ones(k, num_ops))
        # self.alphas_reduce = nn.Parameter(1e-1 * torch.ones(k, num_ops))
        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
        ]

    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self):

        def _parse(weights):
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                edges = sorted(range(i + 2),
                               key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[
                        :2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != PRIMITIVES.index('none'):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene

        gene_normal = _parse(self.GumbleSoftmax(self.alphas_normal).data.cpu().numpy())
        gene_reduce = _parse(self.GumbleSoftmax(self.alphas_reduce).data.cpu().numpy())

        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )
        return genotype

    def gentype_gumbel(self):

        def _parse_gumbel(weights):
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                gene_step = []
                end = start + n
                W = weights[start:end]
                W_size = W.size()
                sample = self.gumbel_sample(W.view(1, -1), sample_time=self._sample_times, temp=1,
                                            flops_param=None).view(W_size)
                sample = sample.data.cpu().numpy()
                sample_index = np.where(sample > 0)
                node_index = sample_index[0]
                op_index = sample_index[1]
                for j in range(len(op_index)):
                    if op_index[j] != PRIMITIVES.index('none'):
                        gene_step.append((PRIMITIVES[op_index[j]], node_index[j]))
                start = end
                n = n + 1
                gene.append(gene_step)
            return gene

        gene_normal = _parse_gumbel(self.alphas_normal)
        gene_reduce = _parse_gumbel(self.alphas_reduce)
        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )
        return genotype

    def gentype_gumbel_layer(self):

        def _parse_gumbel(weights):
            gene = []
            n = 2
            start = 0
            for i in range(weights.size()[0]):
                gene_step = []
                end = start + n
                W = weights[start:end]
                for layer_num in range(W.size()[0]):
                    prob_list = W[layer_num, :]
                    prob_list_size = prob_list.size()
                    sample = self.gumbel_sample(prob_list.view(1, -1), sample_time=self._sample_times, temp=1,
                                                flops_param=None).view(prob_list_size)
                    sample = sample.data.cpu().numpy()
                    sample_index = np.where(sample > 0)
                    node_index = layer_num
                    op_index = sample_index[0]
                    for j in range(len(op_index)):
                        if op_index[j] != PRIMITIVES.index('none'):
                            gene_step.append((PRIMITIVES[op_index[j]], node_index))
                start = end
                n = n + 1
                gene.append(gene_step)
            return gene

        gene_normal = _parse_gumbel(self.alphas_normal)
        gene_reduce = _parse_gumbel(self.alphas_reduce)
        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )
        return genotype

    def gumbel_sample_weight_batch(self, str_weights, sample_time=3, batch_size=1, temp=1., flops_param=None):
        weights_batch = self.gumbel_sample_weight(str_weights, sample_time, temp, flops_param).unsqueeze(0)
        for i in range(batch_size - 1):
            weights = self.gumbel_sample_weight(str_weights, sample_time, temp, flops_param).unsqueeze(0)
            weights_batch = torch.cat([weights_batch, weights], 0)
        weights_batch = weights_batch.transpose(0, 1).transpose(1, 2)
        return weights_batch

    def gumbel_sample_weight(self, str_weights, sample_time=3, temp=1., flops_param=None):

        weights = self.gumbel_sample(str_weights[0, :].view(1, -1), sample_time, temp=temp,
                                     flops_param=flops_param)
        # weights = self.gumbel_sample(self.alphas_reduce[:offset, :].view(1, -1), 4, temp = temp)
        for j in range(1, str_weights.size()[0]):
            weight_op = self.gumbel_sample(str_weights[j, :].view(1, -1), sample_time,
                                           temp=temp, flops_param=flops_param)
            # weight_op = self.gumbel_sample(self.alphas_reduce[offset:offset + input_num, :].view(1, -1),
            #                                4 * i + 8, temp=temp)
            weights = torch.cat([weights, weight_op], 1)
        weights = weights.view(self.alphas_reduce.size())
        return weights

    # sample across all former nodes
    def gumbel_sample_weight_multi_node(self, str_weights, sample_time=3, temp=1., flops_param=None):

        n = 2
        start = 0
        weights = None
        for i in range(self._steps):
            end = start + n
            weight_op = self.gumbel_sample(str_weights[start:end, :].view(1, -1), sample_time, temp=temp,
                                           flops_param=flops_param)
            weights = weight_op if weights is None else torch.cat([weights, weight_op], 1)
            start = end
            n = n + 1
        weights = weights.view(self.alphas_reduce.size())
        return weights

    def gumbel_sample(self, str_weights, sample_time=3, temp=1., flops_param=None):
        weight_size = str_weights.size()
        str_weights = str_weights.view(1, -1)
        # str_weights = F.softmax(str_weights, dim=-1)
        str_weights = F.relu(str_weights)
        str_weights = str_weights
        if flops_param is not None:
            # flops_weights = F.softmax(flops_param.view(1, -1), dim=-1)
            flops_param = flops_param.view(1, -1)
            flops_weights = flops_param / (torch.sum(flops_param, dim=-1).view(1, -1) + 1e-7)
            str_weights = 0.5 * str_weights + 0.5 * flops_weights
        str_weights = str_weights / (torch.sum(str_weights, dim=-1).view(-1, 1) + 1e-7)
        # str_weights = self.GumbleSoftmax(str_weights, temp=0.7, force_hard=False)
        # print str_weights
        weight_output = self.GumbleSoftmax(str_weights, temp=temp, force_hard=True)
        for i in range(sample_time - 1):
            weights_t0 = self.GumbleSoftmax(str_weights, temp=temp, force_hard=True)
            weight_output = torch.cat([weight_output, weights_t0], 0)
        weight_output = torch.max(weight_output, 0)[0]
        # weight_output = weight_output / (torch.sum(weight_output[1:], dim=-1).view(1, -1) + 1e-9)
        weight_output = weight_output / (torch.sum(weight_output, dim=-1).view(1, -1) + 1e-9)
        # weight_output = F.softmax(weight_output, dim=-1)
        weight_output = weight_output.view(weight_size)
        return weight_output

    def genotype2weight(self, arch):
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(PRIMITIVES)
        weights_normal = np.zeros((k, num_ops), dtype=np.float32)
        weights_reduce = np.zeros((k, num_ops), dtype=np.float32)
        normal_arch = arch[0]
        reduce_arch = arch[2]
        index_start = 0
        for i in range(self._steps):
            for count in range(len(normal_arch[i])):
                op, ind = normal_arch[i][count]
                weights_normal[index_start + ind, PRIMITIVES.index(op)] = 1

            for count in range(len(reduce_arch[i])):
                op, ind = reduce_arch[i][count]
                weights_reduce[index_start + ind, PRIMITIVES.index(op)] = 1

            index_start = index_start + i + 2

        # weights_normal /= np.sum(weights_normal[:, 1:], axis = 1).reshape(-1, 1) + 1e-7
        # weights_reduce /= np.sum(weights_reduce[:, 1:], axis = 1).reshape(-1, 1) + 1e-7

        weights_normal /= np.sum(weights_normal, axis=1).reshape(-1, 1) + 1e-7
        weights_reduce /= np.sum(weights_reduce, axis=1).reshape(-1, 1) + 1e-7

        weights_normal = torch.from_numpy(weights_normal).cuda()
        weights_reduce = torch.from_numpy(weights_reduce).cuda()
        return weights_normal, weights_reduce

    def weight2genotype(self, norm_weights, reduce_weights):

        def _parse_gumbel(weights):
            gene = []
            n = 2
            start = 0
            for i in range(weights.size()[0]):
                gene_step = []
                end = start + n
                W = weights[start:end]
                for layer_num in range(W.size()[0]):
                    prob_list = W[layer_num, :]
                    sample = prob_list.data.cpu().numpy()
                    sample_index = np.where(sample > 0)
                    node_index = layer_num
                    op_index = sample_index[0]
                    for j in range(len(op_index)):
                        # if op_index[j] != PRIMITIVES.index('none'):
                        gene_step.append((PRIMITIVES[op_index[j]], node_index))
                start = end
                n = n + 1
                gene.append(gene_step)
            return gene
        gene_normal = _parse_gumbel(norm_weights)
        gene_reduce = _parse_gumbel(reduce_weights)
        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )
        return genotype

