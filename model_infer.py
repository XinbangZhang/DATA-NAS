import numpy as np
import os
import torch
import torch.nn as nn
from genotypes import *
from model_search_relu import Network

# obtain select time for every op
def get_binary_ratio(num, times):
    if num == 1:
        return [str(times)]
    result_all = []
    rich_time = times - num
    for i in range(rich_time + 1):
        result_next = get_binary_ratio(num - 1, times - 1 - i)
        for result in result_next:
            result_all.append(str(i + 1) + result)
    return result_all

# the prob of select target input
def get_binary_prob(input_prob, m):
    prob_num = len(input_prob)
    if m < prob_num:
        return 0
    binary_ratio = get_binary_ratio(prob_num, m)
    prob_sum = 0.
    num = np.math.factorial(m)
    for ratio in binary_ratio:
        fum = 1.
        prob = 1.
        for index in range(prob_num):
            fum *= np.math.factorial(int(ratio[index]))
            prob *= float(input_prob[index]) ** int(ratio[index])
        prob_sum += num / fum * prob
    return prob_sum

# infer how many op to select
def infer_prob(prob, m):
    prob = prob.reshape(-1)
    prob_index = np.argsort(-prob)
    prob_new = prob[prob_index]
    high_prob = 0.
    high_num = 0
    for i in range(1, m + 1):
        prob_i = get_binary_prob(prob_new[:i], m)
        if prob_i > high_prob:
            high_prob = prob_i
            high_num = i
    return prob_index[:high_num]


def model_infer_gentype(alpha_normal, alpha_reduce, steps, M):
    # print alpha_reduce

    def _parse_gumbel(weights, steps, M):
        gene = []
        n = 2
        start = 0
        for i in range(steps):
            gene_step = []
            end = start + n
            W = weights[start:end, :].detach().cpu().numpy().reshape(-1)
            W = np.clip(W, 0, np.inf)
            W = W / np.sum(W)
            result = infer_prob(W, M)
            for j in range(len(result)):
                node_index = result[j] // len(PRIMITIVES)
                op_index = result[j] % len(PRIMITIVES)
                gene_step.append((PRIMITIVES[op_index], node_index))
            start = end
            n = n + 1
            gene.append(gene_step)
        return gene
    gene_normal = _parse_gumbel(alpha_normal, steps, M)
    gene_reduce = _parse_gumbel(alpha_reduce, steps, M)
    concat = range(2, steps + 2)
    genotype = Genotype(
        normal=gene_normal, normal_concat=concat,
        reduce=gene_reduce, reduce_concat=concat
    )
    return genotype

def model_infer_gentype_mod(alpha_normal, alpha_reduce, steps, M):

    def _parse_gumbel(weights, steps, M):
        gene = []
        n = 2
        start = 0
        for i in range(steps):
            gene_step = []
            end = start + n
            W = weights[start:end, :].detach().cpu().numpy()
            W = np.clip(W, 0, np.inf)
            W = W / np.sum(W)
            none_prob = np.array([np.sum(W[:, 0])])
            W = np.concatenate([none_prob, W[:, 1:].reshape(-1)], axis=0)
            result = infer_prob(W, M)
            for j in range(len(result)):
                if result[j] == 0:
                    continue
                select_op = result[j] - 1
                node_index = select_op // (len(PRIMITIVES) - 1)
                op_index = select_op % (len(PRIMITIVES) - 1)
                gene_step.append((PRIMITIVES[op_index + 1], node_index))
            start = end
            n = n + 1
            gene.append(gene_step)
        return gene

    gene_normal = _parse_gumbel(alpha_normal, steps, M)
    gene_reduce = _parse_gumbel(alpha_reduce, steps, M)
    concat = range(2, steps + 2)
    genotype = Genotype(
        normal=gene_normal, normal_concat=concat,
        reduce=gene_reduce, reduce_concat=concat
    )
    return genotype

def model_infer_gentype_mod_mod(alpha_normal, alpha_reduce, steps, M):

    def _parse_gumbel(weights, steps, M):
        gene = []
        n = 2
        start = 0
        for i in range(steps):
            gene_step = []
            end = start + n
            W = weights[start:end, :].detach().cpu().numpy()
            W = np.clip(W, 0, np.inf)
            W = W[:, 1:].reshape(-1)
            result = infer_prob(W, M)
            # W = W / np.sum(W)
            # none_prob = np.array([np.sum(W[:, 0])])
            # W = np.concatenate([none_prob, W[:, 1:].reshape(-1)], axis=0)
            # result = infer_prob(W, M)
            for j in range(len(result)):
                select_op = result[j]
                node_index = select_op // (len(PRIMITIVES) - 1)
                op_index = select_op % (len(PRIMITIVES) - 1)
                gene_step.append((PRIMITIVES[op_index + 1], node_index))
            start = end
            n = n + 1
            gene.append(gene_step)
        return gene

    gene_normal = _parse_gumbel(alpha_normal, steps, M)
    gene_reduce = _parse_gumbel(alpha_reduce, steps, M)
    concat = range(2, steps + 2)
    genotype = Genotype(
        normal=gene_normal, normal_concat=concat,
        reduce=gene_reduce, reduce_concat=concat
    )
    return genotype


def model_infer_gentype_layer(alpha_normal, alpha_reduce, steps, M, relu = True):

    def _parse_gumbel(weights, steps, M, relu = True):
        gene = []
        n = 2
        start = 0
        for i in range(steps):
            gene_step = []
            end = start + n
            W = weights[start:end, :].detach().cpu().numpy()
            if relu:
                W = np.clip(W, 0, np.inf)
                W = W / np.sum(W, axis=1).reshape(-1, 1)
            else:
                W = np.exp(W) / np.sum(np.exp(W), axis=1).reshape(-1, 1)
            for layer_num in range(W.shape[0]):
                prob_list = W[layer_num, :]
                result = infer_prob(prob_list, M)
                for j in range(len(result)):
                    if result[j] == 0:
                        continue
                    op_index = result[j] % len(PRIMITIVES)
                    node_index = layer_num
                    gene_step.append((PRIMITIVES[op_index], node_index))
            start = end
            n = n + 1
            gene.append(gene_step)
        return gene

    gene_normal = _parse_gumbel(alpha_normal, steps, M, relu = relu)
    gene_reduce = _parse_gumbel(alpha_reduce, steps, M, relu = relu)
    concat = range(2, steps + 2)
    genotype = Genotype(
        normal=gene_normal, normal_concat=concat,
        reduce=gene_reduce, reduce_concat=concat
    )
    return genotype

