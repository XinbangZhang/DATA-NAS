import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

zeros_index = np.array([1, 3, 4, 6, 7, 8, 10, 11, 12, 13])

def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


class Architect(object):

    def __init__(self, model, criterion, args):
        self.network_momentum = args.momentum
        self.network_weight_decay = args.weight_decay
        self.model = model
        self.criterion = criterion
        self.args = args
        # arch_params = list(map(id, self.model.module.arch_parameters()))
        # weight_params = filter(lambda p: id(p) not in arch_params,
        #                        self.model.parameters())
        val_params = self.model.module.arch_parameters()
        # weight_params.append(val_params[0])

        # self.optimizer = torch.optim.Adam(self.model.module.arch_parameters(),
        #                                   lr=args.arch_learning_rate, betas=(0.5, 0.999),
        #                                   weight_decay=args.arch_weight_decay)
        self.optimizer = torch.optim.Adam(val_params,
                                          lr=args.arch_learning_rate, betas=(0.5, 0.999),
                                          weight_decay=args.arch_weight_decay)

    def _compute_unrolled_model(self, input, target, eta, network_optimizer):
        loss = self.model.module._loss(input, target)
        theta = _concat(self.model.parameters()).data
        try:
            moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(
                self.network_momentum)
        except:
            moment = torch.zeros_like(theta)
        dtheta = _concat(torch.autograd.grad(loss, self.model.parameters())).data + self.network_weight_decay * theta
        unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment + dtheta))
        return unrolled_model

    def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, unrolled, epochs=150):
        self.optimizer.zero_grad()
        if unrolled:
            self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, eta, network_optimizer)
        else:
            logits, loss = self._backward_step(input_valid, target_valid, epochs=epochs)
        # self.optimizer.step()
        if self.args.multi_nodes:
            self.model.module.alphas_normal._grad[zeros_index, 0] = 0.
            self.model.module.alphas_reduce._grad[zeros_index, 0] = 0.
        self.optimizer.step()
        return logits, loss

    def _backward_step(self, input_valid, target_valid, epochs=1):

        logits = self.model(input_valid, epochs=epochs)

        def mult_nodes_prob_relu(prob):
            n = 2
            start = 0
            weight_norm = F.relu(prob)
            weigth_prob = None
            for i in range(self.model.module._steps):
                end = start + n
                weight_sample = weight_norm[start:end, :]
                weight_sample = weight_sample / (torch.sum(weight_sample).view(-1, 1) + 1e-7)
                weigth_prob = weight_sample if weigth_prob is None else torch.cat([weigth_prob, weight_sample], 0)
                n = n + 1
                start = end
            return weigth_prob

        def mult_nodes_prob_softmax(prob):
            n = 2
            start = 0
            weight_norm = F.relu(prob)
            weigth_prob = None
            for i in range(self.model.module._steps):
                end = start + n
                weight_sample = weight_norm[start:end, :]
                size = weight_sample.size()
                weight_sample = F.softmax(weight_sample.view(1, -1), dim=-1).view(size)
                weigth_prob = weight_sample if weigth_prob is None else torch.cat([weigth_prob, weight_sample], 0)
                n = n + 1
                start = end
            return weigth_prob

        if not self.args.multi_nodes:
            if self.args.act_funciton == 'relu':
                weigth_norm_normal = F.relu(self.model.module.alphas_normal)
                weigth_norm_reduce = F.relu(self.model.module.alphas_reduce)
                weigth_norm_normal = weigth_norm_normal / (torch.sum(weigth_norm_normal, dim=-1).view(-1, 1) + 1e-7)
                weigth_norm_reduce = weigth_norm_reduce / (torch.sum(weigth_norm_reduce, dim=-1).view(-1, 1) + 1e-7)
            else:
                weigth_norm_normal = F.softmax(self.model.module.alphas_normal, dim=-1)
                weigth_norm_reduce = F.softmax(self.model.module.alphas_reduce, dim=-1)
        else:
            if self.args.act_funciton == 'relu':
                weigth_norm_normal = mult_nodes_prob_relu(self.model.module.alphas_normal)
                weigth_norm_reduce = mult_nodes_prob_relu(self.model.module.alphas_reduce)
            else:
                weigth_norm_normal = mult_nodes_prob_softmax(self.model.module.alphas_normal)
                weigth_norm_reduce = mult_nodes_prob_softmax(self.model.module.alphas_reduce)

        loss = self.criterion(logits, target_valid)
        if self.args.arch_reg:
            loss_l1 = F.l1_loss(weigth_norm_normal, self.model.module.weights_norm) + \
                   F.l1_loss(weigth_norm_reduce, self.model.module.weights_reduce)
            # loss_l2 = F.mse_loss(weigth_norm_normal, self.model.module.weights_norm) + \
            #           F.mse_loss(weigth_norm_reduce, self.model.module.weights_reduce)
            # loss_entro = -torch.sum(weigth_norm_normal * torch.log(weigth_norm_normal + 1e-5) + \
            #                        weigth_norm_reduce * torch.log(weigth_norm_reduce + 1e-5))

            # loss_crossentro = -torch.sum(self.model.module.weights_norm * torch.log(weigth_norm_normal + 1e-5) + \
            #                         self.model.module.weights_reduce * torch.log(weigth_norm_reduce + 1e-5))
            # loss_kl = F.kl_div(weigth_norm_normal, self.model.module.weights_norm) + \
            #           F.kl_div(weigth_norm_reduce, self.model.module.weights_reduce)
            loss = loss + 0.1 * loss_l1
        # loss = self.criterion(logits, target_valid)

        loss.backward()
        return logits, loss

    def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer):
        unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)
        unrolled_loss = unrolled_model._loss(input_valid, target_valid)

        unrolled_loss.backward()
        dalpha = [v.grad for v in unrolled_model.arch_parameters()]
        vector = [v.grad.data for v in unrolled_model.parameters()]
        implicit_grads = self._hessian_vector_product(vector, input_train, target_train)

        for g, ig in zip(dalpha, implicit_grads):
            g.data.sub_(eta, ig.data)

        for v, g in zip(self.model.module.arch_parameters(), dalpha):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)

    def _construct_model_from_theta(self, theta):
        model_new = self.model.module.new()
        model_dict = self.model.state_dict()

        params, offset = {}, 0
        for k, v in self.model.named_parameters():
            v_length = np.prod(v.size())
            params[k] = theta[offset: offset + v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        return model_new.cuda()

    def _hessian_vector_product(self, vector, input, target, r=1e-2):
        R = r / _concat(vector).norm()
        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)
        loss = self.model.module._loss(input, target)
        grads_p = torch.autograd.grad(loss, self.model.module.arch_parameters())

        for p, v in zip(self.model.parameters(), vector):
            p.data.sub_(2 * R, v)
        loss = self.model.module._loss(input, target)
        grads_n = torch.autograd.grad(loss, self.model.module.arch_parameters())

        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)

        return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]
