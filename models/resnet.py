import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ["downsample"]

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
        last_activation="relu",
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ["downsample"]

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
        last_activation="relu",
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        if last_activation == "relu":
            self.last_activation = nn.ReLU(inplace=True)
        elif last_activation == "none":
            self.last_activation = nn.Identity()
        elif last_activation == "sigmoid":
            self.last_activation = nn.Sigmoid()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.last_activation(out)

        return out

class ResNetBase(nn.Module):
    def __init__(self, block_type, num_blocks, c_k=7, widen=1, groups=1,
                 width_per_group=64, get_features=False, norm_layer=None):
        super(ResNetBase, self).__init__()
        if block_type == 'basic':
            block = BasicBlock
        elif block_type == 'bottleneck':
            block = Bottleneck
        self.groups = groups
        self.dilation = 1
        self.c_k = c_k
        self.get_features = get_features
        self.inplanes = width_per_group * widen
        self.base_width = width_per_group
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        num_out_filters = width_per_group * widen
        if c_k == 7:
            # ImageNet 224x224
            self.padding = nn.ConstantPad2d(1, 0.0)
            self.conv1 = nn.Conv2d(3, num_out_filters, kernel_size=c_k, stride=2, padding=2, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        elif c_k == 3:
            # CIFAR-10 32x32
            self.padding = nn.Identity()
            self.conv1 = nn.Conv2d(3, num_out_filters, kernel_size=c_k, stride=1, padding=1, bias=False)
            self.maxpool = nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(num_out_filters)

        self.layer1 = self._make_layer(block, 64 * widen, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128 * widen, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256 * widen, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512 * widen, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        for m in self.modules():
            if isinstance(m, Bottleneck):
                nn.init.constant_(m.bn3.weight, 0)
            elif isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)
    
    def _make_layer(
        self, block, planes, blocks, stride=1, dilate=False, last_activation="relu"
    ):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
                last_activation=(last_activation if blocks == 1 else "relu"),
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    last_activation=(last_activation if i == blocks - 1 else "relu"),
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.padding(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if self.get_features:
            return x, x
        else:
            return x
    
class ResNet(ResNetBase):
    def __init__(self, block_type, num_blocks, c_k=7, num_classes=1000, 
                 widen=1, width_per_group=64, get_features=False, bn_linear_eval=True,):
        super(ResNet, self).__init__(block_type=block_type, 
                                     num_blocks=num_blocks, 
                                     c_k=c_k, 
                                     get_features=get_features,
                                     widen=widen,
                                     width_per_group=width_per_group,
                                     )
        if block_type == 'basic':
            block = BasicBlock
        elif block_type == 'bottleneck':
            block = Bottleneck
        self.linear = nn.Linear(512 * widen * block.expansion, num_classes)
        self.num_classes = num_classes
        self.block = block
        self.reset_parameters() 
        self.bn_linear_eval = bn_linear_eval
        
    def adjust_train_mode(self):
        self.eval()
        self.linear.train()
        for param in self.parameters():
            param.requires_grad = False
        for param in self.linear.parameters():
            param.requires_grad = True

    def add_linear_prob(self):
        if self.bn_linear_eval:
            self.linear = nn.Sequential(
                nn.BatchNorm1d(512 * self.block.expansion, affine=False), # Helps to stablize training
                nn.Linear(512 * self.block.expansion, self.num_classes)
            )
            self.linear[-1].weight.data.normal_(mean=0.0, std=0.01)
            self.linear[-1].bias.data.zero_()
        else:
            self.linear = nn.Linear(512 * self.block.expansion, self.num_classes)
            self.linear.weight.data.normal_(mean=0.0, std=0.01)
            self.linear.bias.data.zero_()
        for _, p in self.named_parameters():
            p.requires_grad = False
        for _, p in self.linear.named_parameters():
            p.requires_grad = True
        return self.linear

    def finetune(self):
        self.linear = nn.Sequential(
                nn.BatchNorm1d(512 * self.block.expansion, affine=False), # Helps to stablize training
                nn.Linear(512 * self.block.expansion, self.num_classes)
        )
        self.linear[-1].weight.data.normal_(mean=0.0, std=0.01)
        self.linear[-1].bias.data.zero_()

    def forward(self, x):
        out = self.padding(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        fe = out
        out = self.linear(out)
        if self.get_features:
            return fe, out
        return out

class BatchNorm1dNoBias(nn.BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bias.requires_grad = False

class ResNetSimCLR(ResNetBase):
    def __init__(self, block_type, num_blocks, c_k=7, projection_dim=128, get_features=False, num_classes=1000):
        super(ResNetSimCLR, self).__init__(block_type=block_type, num_blocks=num_blocks, c_k=c_k, get_features=get_features)
        if block_type == 'basic':
            block = BasicBlock
        elif block_type == 'bottleneck':
            block = Bottleneck
        self.projector = nn.Sequential(
            nn.Linear(block.expansion * 512, block.expansion * 512, bias=False),
            nn.BatchNorm1d(block.expansion * 512),
            nn.ReLU(inplace=True),
            nn.Linear(block.expansion * 512, projection_dim, bias=False),
            BatchNorm1dNoBias(projection_dim)
        )
        self.online_cls = nn.Sequential(
            nn.BatchNorm1d(512 * block.expansion, affine=False),
            nn.Linear(512 * block.expansion, num_classes)
        )
        self.reset_parameters()
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.01)
            
    def forward(self, x):
        out = self.padding(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        fe = out
        cls = self.online_cls(fe.detach().clone())
        out = self.projector(out)
        if self.get_features:
            return fe, out, cls
        return out

class ResNetSimCLRTuned(ResNetBase):
    def __init__(self, block_type, num_blocks, c_k=7, projection_dim=128, get_features=False, num_classes=1000, use_bias=False):
        super(ResNetSimCLRTuned, self).__init__(block_type=block_type, num_blocks=num_blocks, c_k=c_k, get_features=get_features)
        if block_type == 'basic':
            block = BasicBlock
        elif block_type == 'bottleneck':
            block = Bottleneck
        self.projector = nn.Sequential(
            nn.Linear(block.expansion * 512, 8192, bias=use_bias),
            nn.BatchNorm1d(8192),
            nn.ReLU(inplace=True),
            nn.Linear(8192, 8192, bias=use_bias),
            nn.BatchNorm1d(8192),
            nn.ReLU(inplace=True),
            nn.Linear(8192, projection_dim, bias=False),
            BatchNorm1dNoBias(projection_dim)
        )
        self.online_cls = nn.Sequential(
            nn.BatchNorm1d(512 * block.expansion, affine=False),
            nn.Linear(512 * block.expansion, num_classes)
        )
        self.reset_parameters()
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.01)

    def forward(self, x):
        out = self.padding(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        fe = out
        cls = self.online_cls(fe.detach().clone())
        out = self.projector(out)
        if self.get_features:
            return fe, out, cls
        return out
    
class ResNetBYOL(ResNetBase):
    def __init__(self, block_type, num_blocks, c_k=7, projection_dim=256, get_features=False, momentum_branch=False, num_classes=1000,
                 use_bias=False):
        super(ResNetBYOL, self).__init__(block_type=block_type, num_blocks=num_blocks, c_k=c_k, get_features=get_features)
        if block_type == 'basic':
            block = BasicBlock
        elif block_type == 'bottleneck':
            block = Bottleneck
        self.momentum_branch = momentum_branch
        # https://github.com/deepmind/deepmind-research/blob/f5de0ede8430809180254ee957abf36ed62579ef/byol/utils/networks.py#L41
        self.projector = nn.Sequential(
            nn.Linear(block.expansion * 512, 4096, bias=use_bias),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, projection_dim, bias=False),
        )
        self.predictor = nn.Sequential(
            nn.Linear(projection_dim, 4096, bias=use_bias),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, projection_dim, bias=False),
        )
        self.online_cls = nn.Sequential(
            nn.BatchNorm1d(512 * block.expansion, affine=False),
            nn.Linear(512 * block.expansion, num_classes)
        )
        
    def forward(self, x):
        out = self.padding(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        fe = out
        cls = self.online_cls(fe.detach().clone())
        z = self.projector(out)
        if not self.momentum_branch:
            p = self.predictor(z)
        else:
            p = None
            z = z.detach()
        if self.get_features:
            return fe, z, p, cls
        return z, p
    