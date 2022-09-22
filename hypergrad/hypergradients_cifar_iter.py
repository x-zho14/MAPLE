import copy

import torch
from torch.autograd import grad as torch_grad
from torch import Tensor
from typing import List, Callable
from torch.utils.data import DataLoader
import logging
import numpy as np
# noinspection PyUnusedLocal

val_iter, x_v,y_v, ind = None,None,None, None
def one_step_old_grad(model, scores, outer_loss, val_loader, set_grad=True, lr = 0.1, wgrads=None, fix = False):
    global val_iter, x_v,y_v, ind
    if val_iter is None:
        val_iter = iter(val_loader)
    if not fix or x_v is None:
        try:
            x_v,y_v,ind = next(val_iter)
        except:
            val_iter = iter(val_loader)
            x_v,y_v,ind = next(val_iter)
    params = [p for p in model.parameters()]
    hparams = [scores]
    val_loss_batch = forward_for_loss_stochastic(model, (x_v,y_v), outer_loss)
    grad_outer_w, grad_outer_hparams = get_outer_gradients(val_loss_batch, params, hparams, retain_graph=False)
    hgrads = torch_grad(wgrads, hparams, grad_outputs= grad_outer_w, retain_graph=False)
    hgrads = [g - lr*v for g, v in zip(grad_outer_hparams, hgrads)]
    if set_grad:
        update_tensor_grads(hparams, hgrads)
    return hgrads, grad_outer_w, params

def one_step_old_grad_batch(model, weights, outer_loss, val_loader, set_grad=True, lr = 0.1, wgrads=None, fix = False):
    global val_iter, x_v,y_v, ind
    if val_iter is None:
        val_iter = iter(val_loader)
    if not fix or x_v is None:
        try:
            x_v,y_v,ind = next(val_iter)
        except:
            val_iter = iter(val_loader)
            x_v,y_v,ind = next(val_iter)

    params = [p for p in model.parameters() if p.requires_grad is True]
    hparams = weights
    if not isinstance(weights, list):
        hparams = [weights]

    loss_batch_total, grad_outer_w, grad_outer_hparams = forward_for_loss_batch(model, val_loader, outer_loss, hparams = hparams, params = params)

    hgrads = torch_grad(wgrads, hparams, grad_outputs= grad_outer_w, retain_graph=False)
    hgrads = [g - lr*v for g, v in zip(grad_outer_hparams, hgrads)]

    if set_grad:
        update_tensor_grads(hparams, hgrads)
    return hgrads, grad_outer_w, params

def grd(a, b):
    return torch.autograd.grad(a, b, create_graph=True, retain_graph=True)


def list_dot(l1, l2):  # extended dot product for lists
    return torch.stack([(a*b).sum() for a, b in zip(l1, l2)]).sum()


def jvp(fp_map, params, vs):
    dummy = [torch.ones_like(phw).requires_grad_(True) for phw in fp_map(params)]
    g1 = grd(list_dot(fp_map(params), dummy), params)
    return grd(list_dot(vs, g1), dummy)


def get_outer_gradients(outer_loss, params, hparams, retain_graph=True):
    grad_outer_w = grad_unused_zero(outer_loss, params, retain_graph=retain_graph)
    grad_outer_hparams = grad_unused_zero(outer_loss, hparams, retain_graph=retain_graph)
    return grad_outer_w, grad_outer_hparams


def cat_list_to_tensor(list_tx):
    return torch.cat([xx.reshape([-1]) for xx in list_tx])


def update_tensor_grads(hparams, grads):
    with torch.no_grad():
        for l, g in zip(hparams, grads):
            if l.grad is None:
                l.grad = torch.zeros_like(l)
            if g is not None:
                l.grad.data = g.data

def grad_unused_zero(output, inputs, grad_outputs=None, retain_graph=False, create_graph=False):
    grads = torch.autograd.grad(output, inputs, grad_outputs=grad_outputs, allow_unused=True,
                                retain_graph=retain_graph, create_graph=create_graph)

    def grad_or_zeros(grad, var):
        return torch.zeros_like(var) if grad is None else grad

    return tuple(grad_or_zeros(g, v) for g, v in zip(grads, inputs))

def forward_for_loss(model, loader, loss):
    model.eval()
    model.zero_grad()
    loss_total = 0
    for batch_idx, (x, y) in enumerate(loader):
        # Load x.
        x, y = x.cuda(), y.cuda()
        loss_batch, pred_y = loss(x, y, model)
        loss_total += loss_batch
    loss_total /= len(loader)  # batch_idx
    return loss_total

def forward_for_loss_stochastic(model, xy_tuple, loss):
    model.eval()
    # model.zero_grad()
    x, y = xy_tuple
    x, y = x.cuda(), y.cuda()
    loss_batch, _ = loss(model, x, y)
    return loss_batch


def forward_for_loss_batch(model, val_loader, loss, hparams = None, params = None):
    model.eval()
    loss_batch_total = 0
    grad_outer_w, grad_outer_hparams, wgrads = [], [], []
    for i , (x,y,_) in enumerate(val_loader):
        x, y = x.cuda(), y.cuda()
        loss_batch, _ = loss(model, x, y)
        grad_outer_w_batch, grad_outer_hparams_batch = get_outer_gradients(loss_batch, params, hparams, retain_graph=True)
        if i > 0:
            grad_outer_w = [w+wb for w, wb in zip(grad_outer_w, grad_outer_w_batch)]
            grad_outer_hparams = [h+hb for h, hb in zip(grad_outer_hparams, grad_outer_hparams_batch)]
        else:
            grad_outer_w = grad_outer_w_batch
            grad_outer_hparams = grad_outer_hparams_batch
        loss_batch_total += loss_batch.item()
    return loss_batch_total, grad_outer_w, grad_outer_hparams

# def update_params(params, grads, meta_lr):
#     return [param - meta_lr*grad for param, grad in zip(params, grads)]
def update_params(model, grads, inner_opt, meta_lr):
    params = _concat(model.parameters()).data
    grads = _concat(grads).data
    try:
        #TODO: fixed for now, change later
        moments = _concat(inner_opt.state[v]['momentum_buffer'] for v in model.parameters()).mul_(0.9)
    except:
        moments = torch.zeros_like(params)
    with torch.no_grad():
        new_params = torch.sub(params, moments+0.1*grads, alpha=meta_lr)
        offset = 0
        for i, param in enumerate(model.parameters()):
            v_length = np.prod(param.size())
            param.copy_(new_params[offset:offset+v_length].view(param.size()))
            offset += v_length

def copyParams(module_src, module_dest):
    module_dest.load_state_dict(module_src.state_dict())
    # params_src = module_src.named_parameters()
    # params_dest = module_dest.named_parameters()
    #
    # dict_dest = dict(params_dest)
    #
    # for name, param in params_src:
    #     if name in dict_dest:
    #         dict_dest[name].data.copy_(param.data)

def copyRegs(module_src, module_dest):
    regs_src = module_src.get_reg_params()
    regs_dest = module_dest.get_reg_params()

    for i, p in enumerate(regs_src):
        regs_dest[i].data.copy_(regs_src[i].data)

def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])
def _sum(xs):
    return sum([torch.sum(x) for x in xs])
solvers = {
    "one_step_old_grad":one_step_old_grad
}
