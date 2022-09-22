from torchvision.transforms import transforms
import loss_utils
import models
import torch.nn.functional as F
from logging_utils.dir_manage import get_directories
from torch.utils.tensorboard import SummaryWriter
import copy
from hypergrad.meta import MetaSGD
import random
import argparse
import numpy as np
import torch
import wandb
from torch import nn, optim, autograd
from model import EBD, MLP
from utils import concat_envs, eval_acc_class, mean_nll_class, mean_accuracy_class, make_mnist_envs, pretty_print, CMNIST_LYDP, CIFAR_LYPD

def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr

def train_to_converge(model, coreset_loader, coreset_theta):
    model_copy = copy.deepcopy(model)
    model_copy.train()
    optimizer = torch.optim.SGD(model_copy.parameters(), lr=args.inner_lr, momentum=0.9)
    data, target, _, _ = next(iter(coreset_loader))
    data, target = data.cuda(), target.cuda()
    model_weights_cache = []
    opt_checkpoints_cache = []
    for i in range(args.epoch_converge):
        optimizer.zero_grad()
        output = model_copy(data)
        if i == args.epoch_converge - 1:
            opt_checkpoints_cache.append(optimizer.state_dict())
            model_weights_cache.append([w.detach().cpu() for w in model_copy.parameters()])
        acc1 = mean_accuracy(output, target)
        loss = torch.sum(F.binary_cross_entropy_with_logits(output, target, reduction='none').flatten()*coreset_theta/coreset_theta.sum())
        loss.backward()
        optimizer.step()
    data_full, target_full, _, _ = next(iter(full_train_loader))
    data_full, target_full = data_full.cuda(), target_full.cuda()

    model_copy_2 = copy.deepcopy(model)
    model_copy_2.train()
    optimizer = torch.optim.SGD(model_copy_2.parameters(), lr=1, momentum=0.9)
    max_acc, corresponding_train_full = 0, 0

    for i in range(200):
        optimizer.zero_grad()
        output = model_copy_2(data)
        acc1 = mean_accuracy(output, target)
        loss = torch.sum(F.binary_cross_entropy_with_logits(output, target, reduction='none').flatten()*coreset_theta/coreset_theta.sum())
        loss.backward()
        optimizer.step()
        acc_test, loss_test = test(model_copy_2, data_t, target_t)
        acc_full, loss_full = test(model_copy_2, data_full, target_full)
        if acc_full.item() > 0.68:
            corresponding_train_full = acc_full if acc_test > max_acc else corresponding_train_full
            max_acc = max(acc_test, max_acc)
    print("Final Result: Train: ", corresponding_train_full, "Test: ", max_acc)
    return model_copy, loss.item(), acc1, model_weights_cache, opt_checkpoints_cache, corresponding_train_full, max_acc

def get_grad_weights_on_full_train(model, full_train_loader):
    model.train()
    grad_weights_on_full_train = []
    for batch_idx, (train_x, train_y, train_g, train_c) in enumerate(full_train_loader):
        train_x, train_y, train_g, train_c = train_x.cuda(), train_y.cuda().float(), train_g.cuda(), train_c.cuda()
        if args.irm_type == "rex":
            loss_list = []
            train_logits = model(train_x)
            train_nll = 0
            for i in range(int(train_g.max()) + 1):
                ei = (train_g == i).view(-1)
                ey = train_y[ei]
                el = train_logits[ei]
                enll = torch.nn.functional.binary_cross_entropy_with_logits(el, ey)
                train_nll += enll / (train_g.max() + 1)
                loss_list.append(enll)
            loss_t = torch.stack(loss_list)
            loss = ((loss_t - loss_t.mean()) ** 2).mean()
        elif args.irm_type == "irmv1":
            train_logits = ebd(train_g).view(-1, 1) * model(train_x)
            train_nll = torch.nn.functional.binary_cross_entropy_with_logits(train_logits, train_y)
            grad = torch.autograd.grad(
                train_nll * args.envs_num, ebd.parameters(),
                create_graph=True)[0]
            loss = torch.mean(grad ** 2)
        elif args.irm_type == "irmv1b":
            e1 = (train_g == 0).view(-1).nonzero().view(-1)
            e2 = (train_g == 1).view(-1).nonzero().view(-1)
            e1 = e1[torch.randperm(len(e1))]
            e2 = e2[torch.randperm(len(e2))]
            s1 = torch.cat([e1[::2], e2[::2]])
            s2 = torch.cat([e1[1::2], e2[1::2]])
            train_logits = ebd(train_g).view(-1, 1) * model(train_x)

            train_nll1 = torch.nn.functional.binary_cross_entropy_with_logits(train_logits[s1], train_y[s1])
            train_nll2 = torch.nn.functional.binary_cross_entropy_with_logits(train_logits[s2], train_y[s2])
            grad1 = torch.autograd.grad(train_nll1 * args.envs_num, ebd.parameters(), create_graph=True)[0]
            grad2 = torch.autograd.grad(train_nll2 * args.envs_num, ebd.parameters(), create_graph=True)[0]
            loss = torch.mean(torch.abs(grad1 * grad2))
        else:
            raise Exception
        grad_weights_on_full_train_batch = torch.autograd.grad(loss, model.parameters())
        if batch_idx > 0:
            grad_weights_on_full_train = [wb+w for wb, w in zip(grad_weights_on_full_train_batch, grad_weights_on_full_train)]
        else:
            grad_weights_on_full_train = grad_weights_on_full_train_batch
    if args.mean_grad:
        grad_weights_on_full_train = [g/len(full_train_loader) for g in grad_weights_on_full_train]
    return grad_weights_on_full_train

def repass_backward(model, subnet, model_checkpoints, opt_checkpoints, outer_grads_w, loader, theta):
    subnet_grads = 0
    theta_grads = 0
    old_params = model_checkpoints[0]
    old_opt = opt_checkpoints[0]
    for batch_idx, (train_x, train_y, _, _) in enumerate(loader):
        train_x, train_y = train_x.cuda(), train_y.cuda().float()
        old_params_, w_mapped = pseudo_updated_params(model, old_params, old_opt, train_x, train_y, subnet, theta)
        if args.score_update:
            subnet_grads += torch.autograd.grad(w_mapped, subnet, grad_outputs=outer_grads_w, retain_graph=True)[0]
        if args.theta_update:
            theta_grads += torch.autograd.grad(w_mapped, theta, grad_outputs=outer_grads_w)[0]
    return subnet_grads, theta_grads

def pseudo_updated_params(model, model_checkpoint, opt_checkpoint, data, target, subnet_idx, theta_idx):
    pseudo_net = copy.deepcopy(model)
    pseudo_net.train()
    for p, p_old in zip(pseudo_net.parameters(), model_checkpoint):
        p.data.copy_(p_old.cuda())
    pseudo_optimizer = MetaSGD(pseudo_net, pseudo_net.parameters(), lr=0.1)
    pseudo_optimizer.load_state_dict(opt_checkpoint)
    w_old = [p for p in pseudo_net.parameters()]
    pseudo_outputs = pseudo_net(data)
    pseudo_loss_vector = F.binary_cross_entropy_with_logits(pseudo_outputs, target, reduction='none').flatten()
    pseudo_loss_vector *= subnet_idx * theta_idx / torch.sum(subnet_idx * theta_idx).detach()
    pseudo_loss = torch.sum(pseudo_loss_vector)
    pseudo_grads = torch.autograd.grad(pseudo_loss, pseudo_net.parameters(), create_graph=True)
    w_mapped = pseudo_optimizer.meta_step(pseudo_grads, if_update=False)
    return w_old, w_mapped

def solve(model, full_train_loader, writer):
    pr_target = args.coreset_size / args.limit
    prune_rate = pr_target
    ts = int(args.ts * args.max_outer_iter)
    te = int(args.te * args.max_outer_iter)
    pr_start = args.start_coreset_size / args.limit

    scores = torch.full([args.limit], pr_start, dtype=torch.float, requires_grad=True, device="cuda")
    theta = torch.full([args.limit], 1, dtype=torch.float, requires_grad=True, device="cuda")
    scores_opt = torch.optim.Adam([scores], lr=args.outer_lr)
    theta_opt = torch.optim.Adam([theta], lr=args.theta_lr)
    scores.grad = torch.zeros_like(scores)
    theta.grad = torch.zeros_like(theta)
    corresponding_train, best_test = 0, 0
    for outer_iter in range(args.max_outer_iter):
        if args.iterative:
            if outer_iter < ts:
                prune_rate = pr_start
            elif outer_iter < te:
                prune_rate = pr_target + (pr_start - pr_target) * (1 - (outer_iter - ts) / (te - ts)) ** 3
            else:
                prune_rate = pr_target
        args.coreset_size = prune_rate * args.limit
        print("now coreset_size", args.coreset_size)
        print(f"outer_iter {outer_iter}")
        writer.add_histogram("Scores Distribution", scores, outer_iter)
        temp = 1 / ((1 - 0.03) * (1 - outer_iter / args.max_outer_iter) + 0.03)
        scores_opt.zero_grad()
        theta_opt.zero_grad()
        for i in range(args.K):
            subnet = obtain_mask(scores, temp)
            grad_subnet_to_scores = torch.autograd.grad(subnet, scores, torch.ones_like(scores))[0]
            subnet = subnet.detach()
            subnet.requires_grad = True
            indices = torch.nonzero(subnet.squeeze())
            indices = indices.reshape(len(indices))
            coreset_train_loader = dp.get_coreset_train_loader(indices)
            coreset_theta = theta[indices].detach()
            model_copy_converged, loss, top1, model_weights_cache, opt_checkpoints_cache, train_acc, test_acc = train_to_converge(model, coreset_train_loader, coreset_theta)
            if test_acc > best_test:
                best_test = test_acc
                corresponding_train = train_acc
            grad_weights_on_full_train = get_grad_weights_on_full_train(model_copy_converged, full_train_loader)
            del model_copy_converged
            grad_subnet, grad_theta = repass_backward(model, subnet, model_weights_cache, opt_checkpoints_cache, grad_weights_on_full_train, full_train_loader, theta)
            with torch.no_grad():
                if args.score_update:
                    scores.grad += grad_subnet.data*grad_subnet_to_scores.data
                if args.theta_update:
                    theta.grad += grad_theta.data
            # print("grad ", i, grad_subnet.data*grad_subnet_to_scores.data)
            del grad_subnet, grad_subnet_to_scores, grad_theta
            torch.cuda.empty_cache()
        with torch.no_grad():
            if args.score_update:
                scores.grad /= args.K
            if args.theta_update:
                theta.grad /= args.K
        # print("grad ", scores.grad)
        torch.nn.utils.clip_grad_norm_(scores, 3)
        torch.nn.utils.clip_grad_norm_(theta, 3)
        if args.score_update:
            scores_opt.step()
        if args.theta_update:
            theta_opt.step()
        constrainScoreByWhole(scores)
        with torch.no_grad():
            theta.data = F.relu(theta.data)
    subnet = (torch.rand_like(scores) < scores).float()
    indices = torch.nonzero(subnet.squeeze())
    indices = indices.reshape(len(indices))
    return indices, coreset_theta, corresponding_train, best_test

def solve_v_total(weight, subset):
    k = subset
    a, b = 0, 0
    b = max(b, weight.max())
    def f(v):
        s = (weight - v).clamp(0, 1).sum()
        return s - k
    if f(0) < 0:
        return 0
    itr = 0
    while (1):
        itr += 1
        v = (a + b) / 2
        obj = f(v)
        if abs(obj) < 1e-3 or itr > 20:
            break
        if obj < 0:
            b = v
        else:
            a = v
    v = max(0, v)
    return v

def constrainScoreByWhole(scores):
    with torch.no_grad():
        v = solve_v_total(scores, args.coreset_size)
        scores.sub_(v).clamp_(0, 1)

class GetMaskDiscrete(torch.autograd.Function):
    @staticmethod
    def forward(ctx, m_cont):
        m_dis = (m_cont >= 0.5).float()
        return m_dis
    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs

def train(model, data, target, optimizer, coreset_theta):
    model.train()
    optimizer.zero_grad()
    output = model(data)
    acc1 = mean_accuracy(output, target)
    loss = torch.sum(F.binary_cross_entropy_with_logits(output, target, reduction='none') * coreset_theta/coreset_theta.sum())
    loss.backward()
    optimizer.step()
    return acc1, loss

def test(model, data_t, target_t):
    model.eval()
    output = model(data_t)
    acc1 = mean_accuracy(output, target_t)
    loss = F.binary_cross_entropy_with_logits(output, target_t)
    return acc1, loss


def obtain_mask(scores, temp, eps=1e-20):
    uniform0 = torch.rand_like(scores)
    uniform1 = torch.rand_like(scores)
    noise = -torch.log(torch.log(uniform0 + eps) / torch.log(uniform1 + eps) + eps)
    return GetMaskDiscrete.apply(torch.sigmoid((torch.log(scores + eps) - torch.log(1.0 - scores + eps) + noise) * temp))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='MNIST summary generator')
    parser.add_argument('--seed', type=int, default=None, metavar='seed', help='random seed (default: 0)')
    parser.add_argument('--method', type=str, default="probability_1step")
    parser.add_argument('--coreset_size', default=100, type=int)
    parser.add_argument('--start_coreset_size', default=100, type=int)
    parser.add_argument('--K', default=1, type=int)
    parser.add_argument('--max_prob_update', default=2000, type=int)
    parser.add_argument('--max_weight_update', default=100, type=int)
    parser.add_argument('--limit', default=1000, type=int)
    parser.add_argument('--train_epoch', default=150, type=int)
    parser.add_argument('--batch_size', default=1000, type=int)
    parser.add_argument('--outer_lr', default=5e-2, type=float)
    parser.add_argument('--theta_lr', default=5e-2, type=float)
    parser.add_argument('--inner_lr', default=0.1, type=float)
    parser.add_argument('--max_outer_iter', default=100, type=int)
    parser.add_argument('--runs_name', default="ours", type=str)
    parser.add_argument('--epoch_converge', default=100, type=int)
    parser.add_argument("--theta_update", default=False, action="store_true")
    parser.add_argument("--score_update", default=False, action="store_true")
    parser.add_argument("--mean_grad", default=False, action="store_true")
    parser.add_argument("--iterative", default=False, action="store_true")
    parser.add_argument('--ts', default=0.16, type=float)
    parser.add_argument('--te', default=0.6, type=float)

    parser.add_argument('--envs_num', type=int, default=2)
    parser.add_argument('--dataset', type=str, default="mnist", choices=["cifar", "mnist"])
    parser.add_argument('--irm_type', default="irmv1", type=str)
    parser.add_argument('--classes_num', type=int, default=2)
    parser.add_argument('--data_num', type=int, default=50000)
    parser.add_argument('--env_type', default="linear", type=str, choices=["2_group", "cos", "linear"])
    parser.add_argument('--hidden_dim', type=int, default=390)
    parser.add_argument('--penalty_weight', type=float, default=10000.0)
    parser.add_argument('--grayscale_model', type=int, default=0)
    parser.add_argument('--noise', type=float, default=0.25)
    parser.add_argument('--corr', nargs="*", type=float, default=[0.1, 0.2, 0.9])
    parser.add_argument('--drop', type=float, default=0)

    args = parser.parse_args()
    wandb.init(project="bilevel", name=args.runs_name, config=args)
    run_base_dir, ckpt_base_dir, log_base_dir = get_directories(args)
    print(run_base_dir, ckpt_base_dir, log_base_dir)
    writer = SummaryWriter(log_dir=log_base_dir)
    if args.seed:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    ebd = EBD(args).cuda()

    if args.dataset == "mnist":
        dp = CMNIST_LYDP(args)
        mean_nll = mean_nll_class
        mean_accuracy = mean_accuracy_class
        eval_acc = eval_acc_class

    full_train_loader = dp.get_train_loader(args.limit)
    test_loader = dp.get_test_loader()

    data_t, target_t, _, _ = next(iter(test_loader))
    data_t, target_t = data_t.cuda(), target_t.cuda()

    model_select = MLP(args).cuda()
    print(model_select)
    indices, coreset_theta, corresponding_train, best_test = solve(model_select, full_train_loader, writer)
    print("Training done: Best result: Train", corresponding_train, "Test: ", best_test)