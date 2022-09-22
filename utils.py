import argparse
import pdb
from cm_spurious_dataset import get_data_loader_cifarminst
import math
import numpy as np
import torch
from torchvision import datasets
import os
import sys
from torch import nn, optim, autograd


def torch_bernoulli(p, size):
    return (torch.rand(size) < p).float()

def torch_xor(a, b):
    return (a-b).abs()

def concat_envs(con_envs):
    con_x = torch.cat([env["images"] for env in con_envs])
    con_y = torch.cat([env["labels"] for env in con_envs])
    con_g = torch.cat([
        ig * torch.ones_like(env["labels"])
        for ig,env in enumerate(con_envs)])
    # con_2g = torch.cat([
    #     (ig < (len(con_envs) // 2)) * torch.ones_like(env["labels"])
    #     for ig,env in enumerate(con_envs)]).long()
    con_c = torch.cat([env["color"] for env in con_envs])
    # con_yn = torch.cat([env["noise"] for env in con_envs])
    # return con_x, con_y, con_g, con_c
    return con_x.cuda(), con_y.cuda(), con_g.cuda(), con_c.cuda()


def merge_env(original_env, merged_num):
    merged_envs = merged_num
    a = original_env
    interval = (a.max() - a.min()) // merged_envs + 1
    b = (a - a.min()) // interval
    return b

def eval_acc_class(logits, labels, colors):
    acc  = mean_accuracy_class(logits, labels)
    minacc = mean_accuracy_class(
      logits[colors!=1],
      labels[colors!=1])
    majacc = mean_accuracy_class(
      logits[colors==1],
      labels[colors==1])
    return acc, minacc, majacc

def mean_accuracy_class(logits, y):
    preds = (logits > 0.).float()
    return ((preds - y).abs() < 1e-2).float().mean()

def eval_acc_reg(logits, labels, colors):
    acc  = mean_nll_reg(logits, labels)
    minacc = torch.tensor(0.0)
    majacc = torch.tensor(0.0)
    return acc, minacc, majacc


def get_strctured_penalty(strctnet, ebd, envs_num, xis):
    x0, x1, x2 = xis
    assert envs_num > 2
    x2_ebd = ebd(x2).view(-1, 1) - 1
    x1_ebd = ebd(x1).view(-1, 1) - 1
    x0_ebd = ebd(x0).view(-1, 1) - 1
    x01_ebd = (x0_ebd-x1_ebd)[:, None]
    x12_ebd = (x1_ebd-x2_ebd)[:, None]
    x12_ebd_logit = strctnet(x01_ebd)
    return 10**13 * (x12_ebd_logit - x12_ebd).pow(2).mean()


def make_environment(images, labels, e, noise):
    # 2x subsample for computational convenience
    images = images.reshape((-1, 28, 28))[:, ::2, ::2]
    # Assign a binary label based on the digit; flip label with probability 0.25
    labels = (labels < 5).float()
    labels = torch_xor(labels, torch_bernoulli(noise, len(labels)))
    # Assign a color based on the label; flip the color with probability e
    color_mask = torch_bernoulli(e, len(labels))
    colors = torch_xor(labels, color_mask)
    # colors = torch_xor(labels, torch_bernoulli(e, len(labels)))
    # Apply the color to the image by zeroing out the other color channel
    images = torch.stack([images, images], dim=1)
    images[torch.tensor(range(len(images))), (1-colors).long(), :, :] *= 0
    return {
      'images': (images.float() / 255.),
      'labels': labels[:, None],
      'color': (1- color_mask[:, None])
    }


def make_mnist_envs(flags):
    mnist = datasets.MNIST('~/datasets/mnist', train=True, download=True)
    mnist_train = (mnist.data[:flags.data_num], mnist.targets[:flags.data_num])
    mnist_val = (mnist.data[50000:], mnist.targets[50000:])
    rng_state = np.random.get_state()
    np.random.shuffle(mnist_train[0].numpy())
    np.random.set_state(rng_state)
    np.random.shuffle(mnist_train[1].numpy())
    # Build environments
    envs_num = flags.envs_num
    envs = []
    if flags.env_type == "linear":
        for i in range(envs_num):
            envs.append(make_environment(mnist_train[0][i::envs_num], mnist_train[1][i::envs_num], flags.corr[i], flags.noise))
    elif flags.env_type == "sin":
        for i in range(envs_num):
            envs.append(
                make_environment(mnist_train[0][i::envs_num], mnist_train[1][i::envs_num], (0.2 - 0.1) * math.sin(i * 2.0 * math.pi / (envs_num-1)) * i + 0.1, flags.noise))
    elif flags.env_type == "step":
        lower_coef = 0.1
        upper_coef = 0.2
        env_per_group = flags.envs_num // 2
        for i in range(envs_num):
            env_coef = lower_coef if i < env_per_group else upper_coef
            envs.append(
                make_environment(
                    mnist_train[0][i::envs_num],
                    mnist_train[1][i::envs_num],
                    env_coef, flags.noise))
    else:
        raise Exception
    envs.append(make_environment(mnist_val[0], mnist_val[1], flags.corr[2], flags.noise))
    return envs

def make_one_logit(num, sp_ratio, dim_inv, dim_spu):
    cc = CowCamels(
        dim_inv=dim_inv, dim_spu=dim_spu, n_envs=1,
        p=[sp_ratio], s= [0.5])
    inputs, outputs, colors, inv_noise= cc.sample(
        n=num, env="E0")
    return {
        'images': inputs,
        'labels': outputs,
        'color': colors[:, None]
    }

def make_one_reg(num, sp_cond, inv_cond, dim_inv, dim_spu):
    ar = AntiReg(
        dim_inv=dim_inv, dim_spu=dim_spu, n_envs=1,
        s=[sp_cond], inv= [inv_cond])
    inputs, outputs, colors, inv_noise= ar.sample(
        n=num, env="E0")
    return {
        'images': inputs,
        'labels': outputs,
        'color': colors,
        'noise': None,
    }

def make_logit_envs(total_num, flags):
    envs_num = flags.envs_num
    envs = []
    if flags.env_type == "linear":
        lower_coef = 0.8
        upper_coef = 0.9
        for i in range(envs_num):
            envs.append(
                make_one_logit(
                    total_num // envs_num,
                    (upper_coef - lower_coef)/(envs_num-1) * i + lower_coef,
                    flags.dim_inv,
                    flags.dim_spu))
    elif flags.env_type == "cos":
        lower_coef = 0.8
        upper_coef = 0.9
        for i in range(envs_num):
            envs.append(
                make_one_logit(
                    total_num // envs_num,
                    (upper_coef - lower_coef) * math.cos(i * 2.0 * math.pi / envs_num) + lower_coef,
                    flags.dim_inv,
                    flags.dim_spu))
    elif flags.env_type == "sin":
        lower_coef = 0.8
        upper_coef = 0.9
        for i in range(envs_num):
            envs.append(
                make_one_logit(
                    total_num // envs_num,
                    (upper_coef - lower_coef) * math.sin(i * 2.0 * math.pi / envs_num) + lower_coef,
                    flags.dim_inv,
                    flags.dim_spu))
    elif flags.env_type == "2cos":
        lower_coef = 0.8
        upper_coef = 0.9
        for i in range(envs_num):
            envs.append(
                make_one_logit(
                    total_num // envs_num,
                    (upper_coef - lower_coef) * math.cos(i * 4.0 * math.pi / envs_num) + lower_coef,
                    flags.dim_inv,
                    flags.dim_spu))
    elif flags.env_type == "2sin":
        lower_coef = 0.8
        upper_coef = 0.9
        for i in range(envs_num):
            envs.append(
                make_one_logit(
                    total_num // envs_num,
                    (upper_coef - lower_coef) * math.sin(i * 4.0 * math.pi / envs_num) + lower_coef,
                    flags.dim_inv,
                    flags.dim_spu))
    else:
        raise Exception
    envs.append(make_one_logit(total_num, 0.1, flags.dim_inv, flags.dim_spu))
    return envs

def make_reg_envs(total_num, flags):
    envs_num = flags.envs_num
    envs = []
    if flags.env_type == "linear":
        lower_coef = 0.5
        upper_coef = 1.0
        inv_cond = 1.0
        for i in range(envs_num):
            envs.append(
                make_one_reg(
                    total_num // envs_num,
                    (upper_coef - lower_coef)/(envs_num-1) * i + lower_coef,
                    inv_cond,
                    flags.dim_inv,
                    flags.dim_spu))
    else:
        raise Exception
    envs.append(make_one_reg(total_num, 9.9, inv_cond, flags.dim_inv, flags.dim_spu))
    return envs


def mean_nll_class(logits, y):
    return nn.functional.binary_cross_entropy_with_logits(logits, y)

def mean_nll_reg(logits, y):
    l2loss = nn.MSELoss()
    return l2loss(logits, y)

def mean_accuracy_reg(logits, y, colors=None):
    return mean_nll_reg(logits, y)


def pretty_print(*values):
    col_width = 13
    def format_val(v):
        if not isinstance(v, str):
            v = np.array2string(v, precision=5, floatmode='fixed')
        return v.ljust(col_width)
    str_values = [format_val(v) for v in values]
    print("   ".join(str_values))


class LYDataProvider(object):
    def __init__(self):
        pass

    def preprocess_data(self):
        pass

    def fetch_train(self):
        pass

    def fetch_test(self):
        pass

class LYDataProviderMK(LYDataProvider):
    def __init__(self, flags):
        super(LYDataProviderMK, self).__init__()

    def preprocess_data(self):
        self.train_x, self.train_y, self.train_g, self.train_c = concat_envs(self.envs[:-1])
        self.test_x, self.test_y, self.test_g, self.test_c = concat_envs(self.envs[-1:])

    def fetch_train(self):
        return self.train_x, self.train_y, self.train_g, self.train_c


    def fetch_test(self):
        return self.test_x, self.test_y, self.test_g, self.test_c


class CMNIST_LYDP(LYDataProviderMK):
    def __init__(self, flags):
        super(CMNIST_LYDP, self).__init__(flags)
        self.flags = flags
        self.envs = make_mnist_envs(flags)
        self.preprocess_data()

    def get_train_loader(self, limit):
        return [(self.train_x[:limit], self.train_y[:limit], self.train_g[:limit], self.train_c[:limit])]

    def get_test_loader(self):
        return [(self.test_x, self.test_y, self.test_g, self.test_c)]

    def get_coreset_train_loader(self, indices):
        return [(self.train_x[indices], self.train_y[indices], self.train_g[indices], self.train_c[indices])]

class LOGIT_LYDP(LYDataProviderMK):
    def __init__(self, flags):
        super(LOGIT_LYDP, self).__init__(flags)
        self.flags = flags
        self.envs = make_logit_envs(flags.data_num, flags)
        self.preprocess_data()

class REG_LYDP(LYDataProviderMK):
    def __init__(self, flags):
        super(REG_LYDP, self).__init__(flags)
        self.flags = flags
        self.envs = make_logit_envs(flags.data_num, flags)
        self.preprocess_data()

class CIFAR_LYPD(LYDataProvider):
    def __init__(self, flags):
        super(CIFAR_LYPD, self).__init__()
        self.flags = flags
        self.preprocess_data()

    def preprocess_data(self):
        train_num=10000
        test_num=1000 #1800
        cons_list = [0.999,0.7,0.1]
        train_envs = len(cons_list) - 1
        ratio_list = [1. / train_envs] * (train_envs)
        spd, self.train_loader, self.val_loader, self.test_loader, self.train_data, self.val_data, self.test_data = get_data_loader_cifarminst(
            batch_size=self.flags.batch_size,
            train_num=train_num,
            test_num=test_num,
            cons_ratios=cons_list,
            train_envs_ratio=ratio_list,
            label_noise_ratio=0.1,
            color_spurious=False,
            transform_data_to_standard=0,
            oracle=0)
        self.train_loader_iter = iter(self.train_loader)

    def fetch_train(self):
        try:
            batch_data = self.train_loader_iter.__next__()
        except:
            self.train_loader_iter = iter(self.train_loader)
            batch_data = self.train_loader_iter.__next__()
        batch_data = tuple(t.cuda() for t in batch_data)
        x, y, g, sp = batch_data
        return x, y.float().cuda(), g, sp

    def fetch_test(self):
        ds = self.test_data.val_dataset
        batch = ds.x_array, ds.y_array, ds.env_array, ds.sp_array
        batch = tuple(
            torch.Tensor(t).cuda()
            for t in batch)
        x, y, g, sp = batch
        return x, y.float(), g, sp


import sys
import os
import torch
import numpy as np
import csv

import torch
import torch.nn as nn
import torchvision
from models import model_attributes


class Logger(object):
    def __init__(self, fpath=None, mode="w"):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            self.file = open(fpath, mode)

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


class CSVBatchLogger:
    def __init__(self, csv_path, n_groups, mode="w"):
        columns = ["epoch", "batch"]
        for idx in range(n_groups):
            columns.append(f"avg_loss_group:{idx}")
            columns.append(f"exp_avg_loss_group:{idx}")
            columns.append(f"avg_acc_group:{idx}")
            columns.append(f"processed_data_count_group:{idx}")
            columns.append(f"update_data_count_group:{idx}")
            columns.append(f"update_batch_count_group:{idx}")
        columns.append("avg_actual_loss")
        columns.append("avg_per_sample_loss")
        columns.append("avg_acc")
        columns.append("model_norm_sq")
        columns.append("reg_loss")
        columns.append("worst_group_loss")
        columns.append("worst_group_acc")

        self.path = csv_path
        self.file = open(csv_path, mode)
        self.columns = columns
        self.writer = csv.DictWriter(self.file, fieldnames=columns)
        if mode == "w":
            self.writer.writeheader()

    def log(self, epoch, batch, stats_dict):
        stats_dict["epoch"] = epoch
        stats_dict["batch"] = batch
        self.writer.writerow(stats_dict)

    def flush(self):
        self.file.flush()

    def close(self):
        self.file.close()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    temp = target.view(1, -1).expand_as(pred)
    temp = temp.cuda()
    correct = pred.eq(temp)

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def set_seed(seed):
    """Sets seed"""
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def log_args(args, logger):
    for argname, argval in vars(args).items():
        logger.write(f'{argname.replace("_", " ").capitalize()}: {argval}\n')
    logger.write("\n")


def hinge_loss(yhat, y):
    # The torch loss takes in three arguments so we need to split yhat
    # It also expects classes in {+1.0, -1.0} whereas by default we give them in {0, 1}
    # Furthermore, if y = 1 it expects the first input to be higher instead of the second,
    # so we need to swap yhat[:, 0] and yhat[:, 1]...
    torch_loss = torch.nn.MarginRankingLoss(margin=1.0, reduction="none")
    y = (y.float() * 2.0) - 1.0
    return torch_loss(yhat[:, 1], yhat[:, 0], y)


def get_model(model, pretrained, resume, n_classes, dataset, log_dir):
    if resume:
        model = torch.load(os.path.join(log_dir, "last_model.pth"))
        d = train_data.input_size()[0]
    elif model_attributes[model]["feature_type"] in (
            "precomputed",
            "raw_flattened",
    ):
        assert pretrained
        # Load precomputed features
        d = train_data.input_size()[0]
        model = nn.Linear(d, n_classes)
        model.has_aux_logits = False
    elif model == "resnet50":
        model = torchvision.models.resnet50(pretrained=pretrained)
        d = model.fc.in_features
        model.fc = nn.Linear(d, n_classes)
    elif model == "resnet18":
        model = torchvision.models.resnet18(pretrained=pretrained)
        d = model.fc.in_features
        model.fc = nn.Linear(d, n_classes)
    elif model == "resnet34":
        model = torchvision.models.resnet34(pretrained=pretrained)
        d = model.fc.in_features
        model.fc = nn.Linear(d, n_classes)
    elif model == "wideresnet50":
        model = torchvision.models.wide_resnet50_2(pretrained=pretrained)
        d = model.fc.in_features
        model.fc = nn.Linear(d, n_classes)
    elif model.startswith('bert'):
        if dataset == "MultiNLI":

            assert dataset == "MultiNLI"

            from pytorch_transformers import BertConfig, BertForSequenceClassification

            config_class = BertConfig
            model_class = BertForSequenceClassification

            config = config_class.from_pretrained("bert-base-uncased",
                                                  num_labels=3,
                                                  finetuning_task="mnli")
            model = model_class.from_pretrained("bert-base-uncased",
                                                from_tf=False,
                                                config=config)
        elif dataset == "jigsaw":
            from transformers import BertForSequenceClassification
            model = BertForSequenceClassification.from_pretrained(
                model,
                num_labels=n_classes)
            print(f'n_classes = {n_classes}')
        else:
            raise NotImplementedError
    else:
        raise ValueError(f"{model} Model not recognized.")

    return model
