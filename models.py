import torch
import torch.nn as nn
import torch.nn.functional as F

model_attributes = {
    "bert": {
        "feature_type": "text"
    },
    "inception_v3": {
        "feature_type": "image",
        "target_resolution": (299, 299),
        "flatten": False,
    },
    "wideresnet50": {
        "feature_type": "image",
        "target_resolution": (224, 224),
        "flatten": False,
    },
    "resnet50": {
        "feature_type": "image",
        "target_resolution": (224, 224),
        "flatten": False,
    },
    "resnet18": {
        "feature_type": "image",
        "target_resolution": (224, 224),
        "flatten": False,
    },
    "resnet34": {
        "feature_type": "image",
        "target_resolution": None,
        "flatten": False
    },
    "raw_logistic_regression": {
        "feature_type": "image",
        "target_resolution": None,
        "flatten": True,
    },
    "bert-base-uncased": {
        'feature_type': 'text'
    },
}

class LogisticRegression(nn.Module):

    def __init__(self, input_dim, nr_classes):
        super(LogisticRegression, self).__init__()
        self.fc = nn.Linear(input_dim, nr_classes)

    def forward(self, x):
        return self.fc(x)


class FNNet(nn.Module):

    def __init__(self, input_dim, interm_dim, output_dim):
        super(FNNet, self).__init__()

        self.input_dim = input_dim
        self.dp1 = torch.nn.Dropout(0.2)
        self.dp2 = torch.nn.Dropout(0.2)
        self.fc1 = nn.Linear(input_dim, interm_dim)
        self.fc2 = nn.Linear(interm_dim, interm_dim)
        self.fc3 = nn.Linear(interm_dim, output_dim)

    def forward(self, x):
        x = self.embed(x)
        x = self.fc3(x)
        return x

    def embed(self, x):
        x = self.dp1(F.relu(self.fc1(x.view(-1, self.input_dim))))
        x = self.dp2(F.relu(self.fc2(x)))
        return x


class ConvNet(nn.Module):
    def __init__(self, output_dim, maxpool=True, base_hid=32):
        super(ConvNet, self).__init__()
        self.base_hid = base_hid
        self.conv1 = nn.Conv2d(1, base_hid, 5, 1)
        self.dp1 = torch.nn.Dropout(0.5)
        self.conv2 = nn.Conv2d(base_hid, base_hid*2, 5, 1)
        self.dp2 = torch.nn.Dropout(0.5)
        self.fc1 = nn.Linear(4 * 4 * base_hid*2, base_hid*4)
        self.dp3 = torch.nn.Dropout(0.5)
        self.fc2 = nn.Linear(base_hid*4, output_dim)
        self.maxpool = maxpool

    def forward(self, x, return_feat=False):
        x = self.embed(x)
        out = self.fc2(x)
        if return_feat:
            return out, x.detach()
        return out

    def embed(self, x):
        x = F.relu(self.dp1(self.conv1(x)))
        if self.maxpool:
            x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.dp2(self.conv2(x)))
        if self.maxpool:
            x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 2*self.base_hid)
        x = F.relu(self.dp3(self.fc1(x)))
        return x

class ConvNetNoDropout(nn.Module):
    def __init__(self, output_dim, maxpool=True, base_hid=32):
        super(ConvNetNoDropout, self).__init__()
        self.base_hid = base_hid
        self.conv1 = nn.Conv2d(1, base_hid, 5, 1)
        self.conv2 = nn.Conv2d(base_hid, base_hid*2, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * base_hid*2, base_hid*4)
        self.fc2 = nn.Linear(base_hid*4, output_dim)
        self.maxpool = maxpool

    def forward(self, x, return_feat=False):
        x = self.embed(x)
        out = self.fc2(x)
        if return_feat:
            return out, x.detach()
        return out

    def embed(self, x):
        x = F.relu(self.conv1(x))
        if self.maxpool:
            x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        if self.maxpool:
            x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 2*self.base_hid)
        x = F.relu(self.fc1(x))
        return x

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
            )

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
            )

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = self.conv3(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.embed(x)
        out = self.linear(out)
        return out

    def embed(self, x):
        out = F.relu(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

# class ScoreNet(torch.nn.Module):
#     def __init__(self, input=10, hidden=100, output=1):
#         super(ScoreNet, self).__init__()
#         self.linear1 = nn.Linear(input, hidden)
#         # self.relu = nn.ReLU(inplace=True)
#         self.relu = nn.ReLU()
#         self.linear2 = nn.Linear(hidden, output)
#         # torch.nn.init.xavier_uniform(self.linear1.weight)
#         # torch.nn.init.xavier_uniform(self.linear2.weight)
#
#     def forward(self, x):
#         x = self.linear1(x)
#         x = self.relu(x)
#         out = self.linear2(x)
#         # return out.flatten()
#         return torch.sigmoid(out)

class HiddenLayer(nn.Module):
    def __init__(self, input_size, output_size, ifbn=False):
        super(HiddenLayer, self).__init__()
        self.ifbn = ifbn
        self.fc = nn.Linear(input_size, output_size)
        if ifbn:
            self.bn = nn.BatchNorm1d(output_size)
        self.relu = nn.ReLU()
        torch.nn.init.xavier_uniform(self.fc.weight)

    def forward(self, x):
        out = self.fc(x)
        if self.ifbn:
            out = self.bn(out)
        return self.relu(out)


class ScoreNet(nn.Module):
    def __init__(self,input=10, hidden=100, num_layers=1, ifbn=False):
        super(ScoreNet, self).__init__()
        # self.normalize = Normalize()
        self.first_hidden_layer = HiddenLayer(input, hidden, ifbn)
        self.rest_hidden_layers = nn.Sequential(*[HiddenLayer(hidden, hidden, ifbn) for _ in range(num_layers - 1)])
        self.output_layer = nn.Linear(hidden, 1)

    def forward(self, x):
        # x = self.normalize(x)
        x = self.first_hidden_layer(x)
        x = self.rest_hidden_layers(x)
        x = self.output_layer(x)
        return torch.sigmoid(x)

class LBSign(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clamp_(-1, 1)
    