import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
import numpy as np
from region_loss import RegionLoss
from cfg import *
#from layers.batchnorm.bn import BN2d

class MaxPoolStride1(nn.Module):
    def __init__(self):
        super(MaxPoolStride1, self).__init__()

    def forward(self, x):
        x = F.max_pool2d(F.pad(x, (0,1,0,1), mode='replicate'), 2, stride=1)
        return x

class Reorg(nn.Module):
    def __init__(self, stride=2):
        super(Reorg, self).__init__()
        self.stride = stride
    def forward(self, x):
        stride = self.stride
        assert(x.data.dim() == 4)
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        assert(H % stride == 0)
        assert(W % stride == 0)
        ws = stride
        hs = stride
        x = x.view(B, C, H//hs, hs, W//ws, ws).transpose(3,4).contiguous()
        x = x.view(B, C, H//hs*W//ws, hs*ws).transpose(2,3).contiguous()
        x = x.view(B, C, hs*ws, H//hs, W//ws).transpose(1,2).contiguous()
        x = x.view(B, hs*ws*C, H//hs, W//ws)
        return x

class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        N = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        x = F.avg_pool2d(x, (H, W))
        x = x.view(N, C)
        return x

# for route and shortcut
class EmptyModule(nn.Module):
    def __init__(self):
        super(EmptyModule, self).__init__()

    def forward(self, x):
        return x

def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class LinkNet(nn.Module):

    def __init__(self, cfgfile,
                 block=BasicBlock,
                 layers=[2, 2, 2, 2],
                 shortcut_type='A'):
        self.inplanes = 64
        super(LinkNet, self).__init__()

        # create Resnet-18
        self.conv1 = nn.Conv3d(
            3,
            64,
            kernel_size=7,
            stride=(1, 2, 2),
            padding=(3, 3, 3),
            bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(
            block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            block, 256, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], shortcut_type, stride=2)

        # self.resnet_models = self.create_network(self.blocks) # merge conv, bn,leaky

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # create Darknet-19
        self.blocks = parse_cfg(cfgfile)
        self.model_2d = self.create_network(self.blocks) # merge conv, bn,leaky
        self.loss = self.model_2d[len(self.model_2d)-1]

        self.width = int(self.blocks[0]['width'])
        self.height = int(self.blocks[0]['height'])

        if self.blocks[(len(self.blocks)-1)]['type'] == 'region':
            self.anchors = self.loss.anchors
            self.num_anchors = self.loss.num_anchors
            self.anchor_step = self.loss.anchor_step
            self.num_classes = self.loss.num_classes

        self.header = torch.IntTensor([0,0,0,0])
        self.seen = 0

        # link conv layers
        self.link_cov_1 = nn.Sequential(
            nn.Conv3d(64, 32,
            kernel_size=3,
            stride=(1, 1, 1),
            padding=(1, 1, 1),
            bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            nn.Conv3d(32, 64,
            kernel_size=3,
            stride=(1, 1, 1),
            padding=(1, 1, 1),
            bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

        )

        self.link_cov_2 = nn.Sequential(

            nn.Conv3d(128, 64,
            kernel_size=3,
            stride=(1, 1, 1),
            padding=(1, 1, 1),
            bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            nn.Conv3d(64, 128,
            kernel_size=3,
            stride=(1, 1, 1),
            padding=(1, 1, 1),
            bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
        )

        self.link_cov_3 = nn.Sequential(
            nn.Conv3d(256, 128,
            kernel_size=3,
            stride=(1, 1, 1),
            padding=(1, 1, 1),
            bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),

            nn.Conv3d(128, 256,
            kernel_size=3,
            stride=(1, 1, 1),
            padding=(1, 1, 1),
            bias=False),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
        )

    def forward(self, input2D, input3D):
        outputs = dict()
        # input 2D has a shape like 3X224X224, input 3D has a shape like 3X16X224X224
        # Darknet19-conv1
        x2 = self.model_2d[0](input2D)
        outputs[0] = x2 
        x2 = self.model_2d[1](x2)# output shape 32X112X112
        outputs[1] = x2 
        #print("Darknet19-conv1:",x2.shape)

        # Resnet18-conv1
        x3 = self.conv1(input3D)
        x3 = self.bn1(x3)
        x3 = self.relu(x3)
        x3 = self.maxpool(x3) # output shape 64X8X56X56
        #print("Resnet18-conv1:",x3.shape)

        # Darknet19-conv2
        x2 = self.model_2d[2](x2) 
        outputs[2] = x2 
        x2 = self.model_2d[3](x2) # output shape 64X56X56
        outputs[3] = x2 
        #print("Darknet19-conv2:",x2.shape)

        # Resnet18-conv2
        x3 = self.layer1(x3) # output shape 64X8X56X56
        #print("Resnet18-conv2:",x3.shape)
        
        
        # expand x2 to 64X8X56X56 
        expand_x2 = torch.unsqueeze(x2, 2)
        # print("expand_x2:", expand_x2.shape)
        expand_x2 = torch.cat((expand_x2, expand_x2), 2)
        # print("expand_x2:", expand_x2.shape)
        expand_x2 = torch.cat((expand_x2, expand_x2), 2)
        # print("expand_x2:", expand_x2.shape)
        expand_x2 = torch.cat((expand_x2, expand_x2), 2) #expand_x2 is 64X8X56X56
        # print("expand_x2:", expand_x2.shape)
        
        # We can add convolutional layer here. For Resnet18 it is the same size so I use only addition here
        expand_x2 = self.link_cov_1(expand_x2)
        # print("expand_x2:", expand_x2.shape)
        # add expand_x2 to x3
        x3 = x3 + expand_x2

        # Darknet19-conv3
        x2 = self.model_2d[4](x2) 
        outputs[4] = x2 
        x2 = self.model_2d[5](x2)
        outputs[5] = x2 
        x2 = self.model_2d[6](x2)
        outputs[6] = x2  
        x2 = self.model_2d[7](x2) # output shape 128X28X28
        outputs[7] = x2 
        #print("Darknet19-conv3:",x2.shape)

        # Resnet18-conv3
        x3 = self.layer2(x3) # output shape 128X4X28X28
        #print("Resnet18-conv3:",x3.shape)

        # expand x2 to 128X4X28X28 
        expand_x2 = torch.unsqueeze(x2, 2)
        expand_x2 = torch.cat((expand_x2, expand_x2), 2)
        # print("expand_x2:", expand_x2.shape)
        expand_x2 = torch.cat((expand_x2, expand_x2), 2) 
        #print("expand_x2:", expand_x2.shape) # expand_x2 is 128X4X28X28

        # We can add convolutional layer here. For Resnet18 it is the same size so I use only addition here
        expand_x2 = self.link_cov_2(expand_x2)
       # print("expand_x2:", expand_x2.shape)
        
        # add expand_x2 to x3
        x3 = x3 + expand_x2
        
        # Darknet19-conv4
        x2 = self.model_2d[8](x2)
        outputs[8] = x2  
        x2 = self.model_2d[9](x2)
        outputs[9] = x2 
        x2 = self.model_2d[10](x2)
        outputs[10] = x2  
        x2 = self.model_2d[11](x2) # output shape 256X14X14
        outputs[11] = x2 
        #print("Darknet19-conv4:",x2.shape)

        # Resnet18-conv4
        x3 = self.layer3(x3) # output shape 256X2X14X14
        #print("Resnet18-conv4:",x3.shape)

        # expand x2 to 128X4X28X28 
        expand_x2 = torch.unsqueeze(x2, 2)
        expand_x2 = torch.cat((expand_x2, expand_x2), 2) 
        #print("expand_x2:", expand_x2.shape) # expand_x2 is 256X2X14X14

        # We can add convolutional layer here. For Resnet18 it is the same size so I use only addition here
        expand_x2 = self.link_cov_3(expand_x2)
        #print("expand_x2:", expand_x2.shape)

        # add expand_x2 to x3
        x3 = x3 + expand_x2

        
        # Darknet19-conv5
        x2 = self.model_2d[12](x2)
        outputs[12] = x2  
        x2 = self.model_2d[13](x2)
        outputs[13] = x2 
        x2 = self.model_2d[14](x2) 
        outputs[14] = x2 
        x2 = self.model_2d[15](x2)
        outputs[15] = x2 
        x2 = self.model_2d[16](x2) 
        outputs[16] = x2 
        x2 = self.model_2d[17](x2) # output shape 512X7X7
        outputs[17] = x2 
        #print("Darknet19-conv5:",x2.shape)

        # Resnet18-conv5
        x3 = self.layer4(x3) # output shape 512X1X7X7
        #print("Resnet18-conv5:",x3.shape)

        # Darknet19-conv6
        x2 = self.model_2d[18](x2) 
        outputs[18] = x2 
        x2 = self.model_2d[19](x2)
        outputs[19] = x2 
        x2 = self.model_2d[20](x2) 
        outputs[20] = x2 
        x2 = self.model_2d[21](x2)
        outputs[21] = x2 
        x2 = self.model_2d[22](x2)  # output shape 1024X7X7
        outputs[22] = x2 
        #print("Darknet19-conv6:",x2.shape)

        
        ind = -2
        self.loss = None
        
        for block in self.blocks:
            ind = ind + 1
            if ind<=22:
              continue
            #if ind > 0:
            #    return x

            if block['type'] == 'net':
                continue
            elif block['type'] == 'convolutional' or block['type'] == 'maxpool' or block['type'] == 'reorg' or block['type'] == 'avgpool' or block['type'] == 'softmax' or block['type'] == 'connected':
                x2 = self.model_2d[ind](x2)
                outputs[ind] = x2
            elif block['type'] == 'route':
                layers = block['layers'].split(',')
                layers = [int(i) if int(i) > 0 else int(i)+ind for i in layers]
                if len(layers) == 1:
                    x2 = outputs[layers[0]]
                    outputs[ind] = x2
                elif len(layers) == 2:
                    x_1 = outputs[layers[0]]
                    x_2 = outputs[layers[1]]
                    x2 = torch.cat((x_1,x_2),1)
                    outputs[ind] = x2
            elif block['type'] == 'shortcut':
                from_layer = int(block['from'])
                activation = block['activation']
                from_layer = from_layer if from_layer > 0 else from_layer + ind
                x_1 = outputs[from_layer]
                x_2 = outputs[ind-1]
                x2  = x_1 + x_2
                if activation == 'leaky':
                    x2 = F.leaky_relu(x, 0.1, inplace=True)
                elif activation == 'relu':
                    x2 = F.relu(x, inplace=True)
                outputs[ind] = x2
            elif block['type'] == 'region':
                continue
                print("LOSSS")
            elif block['type'] == 'cost':
                continue
            else:
                print('unknown type %s' % (block['type']))
        # print(x.shape)
        return x2, x3

    def print_network(self):
        print_cfg(self.blocks)

    def create_network(self, blocks):
        models = nn.ModuleList()
    
        prev_filters = 3
        out_filters =[]
        conv_id = 0
        for block in blocks:
            if block['type'] == 'net':
                prev_filters = int(block['channels'])
                continue
            elif block['type'] == 'convolutional':
                conv_id = conv_id + 1
                batch_normalize = int(block['batch_normalize'])
                filters = int(block['filters'])
                kernel_size = int(block['size'])
                stride = int(block['stride'])
                is_pad = int(block['pad'])
                pad = (kernel_size-1)//2 if is_pad else 0
                activation = block['activation']
                model = nn.Sequential()
                if batch_normalize:
                    model.add_module('conv{0}'.format(conv_id), nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=False))
                    model.add_module('bn{0}'.format(conv_id), nn.BatchNorm2d(filters))
                    #model.add_module('bn{0}'.format(conv_id), BN2d(filters))
                else:
                    model.add_module('conv{0}'.format(conv_id), nn.Conv2d(prev_filters, filters, kernel_size, stride, pad))
                if activation == 'leaky':
                    model.add_module('leaky{0}'.format(conv_id), nn.LeakyReLU(0.1, inplace=True))
                elif activation == 'relu':
                    model.add_module('relu{0}'.format(conv_id), nn.ReLU(inplace=True))
                prev_filters = filters
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'maxpool':
                pool_size = int(block['size'])
                stride = int(block['stride'])
                if stride > 1:
                    model = nn.MaxPool2d(pool_size, stride)
                else:
                    model = MaxPoolStride1()
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'avgpool':
                model = GlobalAvgPool2d()
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'softmax':
                model = nn.Softmax()
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'cost':
                if block['_type'] == 'sse':
                    model = nn.MSELoss(size_average=True)
                elif block['_type'] == 'L1':
                    model = nn.L1Loss(size_average=True)
                elif block['_type'] == 'smooth':
                    model = nn.SmoothL1Loss(size_average=True)
                out_filters.append(1)
                models.append(model)
            elif block['type'] == 'reorg':
                stride = int(block['stride'])
                prev_filters = stride * stride * prev_filters
                out_filters.append(prev_filters)
                models.append(Reorg(stride))
            elif block['type'] == 'route':
                layers = block['layers'].split(',')
                ind = len(models)
                layers = [int(i) if int(i) > 0 else int(i)+ind for i in layers]
                if len(layers) == 1:
                    prev_filters = out_filters[layers[0]]
                elif len(layers) == 2:
                    assert(layers[0] == ind - 1)
                    prev_filters = out_filters[layers[0]] + out_filters[layers[1]]
                out_filters.append(prev_filters)
                models.append(EmptyModule())
            elif block['type'] == 'shortcut':
                ind = len(models)
                prev_filters = out_filters[ind-1]
                out_filters.append(prev_filters)
                models.append(EmptyModule())
            elif block['type'] == 'connected':
                filters = int(block['output'])
                if block['activation'] == 'linear':
                    model = nn.Linear(prev_filters, filters)
                elif block['activation'] == 'leaky':
                    model = nn.Sequential(
                               nn.Linear(prev_filters, filters),
                               nn.LeakyReLU(0.1, inplace=True))
                elif block['activation'] == 'relu':
                    model = nn.Sequential(
                               nn.Linear(prev_filters, filters),
                               nn.ReLU(inplace=True))
                prev_filters = filters
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'region':
                loss = RegionLoss()
                anchors = block['anchors'].split(',')
                loss.anchors = [float(i) for i in anchors]
                loss.num_classes = int(block['classes'])
                loss.num_anchors = int(block['num'])
                loss.anchor_step = len(loss.anchors)//loss.num_anchors
                loss.object_scale = float(block['object_scale'])
                loss.noobject_scale = float(block['noobject_scale'])
                loss.class_scale = float(block['class_scale'])
                loss.coord_scale = float(block['coord_scale'])
                out_filters.append(prev_filters)
                models.append(loss)
            else:
                print('unknown type %s' % (block['type']))
    
        return models

    def load_weights(self, weightfile):
        fp = open(weightfile, 'rb')
        header = np.fromfile(fp, count=4, dtype=np.int32)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]
        buf = np.fromfile(fp, dtype = np.float32)
        fp.close()

        start = 0
        ind = -2
        for block in self.blocks:
            if start >= buf.size:
                break
            ind = ind + 1
            if block['type'] == 'net':
                continue
            elif block['type'] == 'convolutional':
                model = self.model_2d[ind]
                batch_normalize = int(block['batch_normalize'])
                if batch_normalize:
                    start = load_conv_bn(buf, start, model[0], model[1])
                else:
                    start = load_conv(buf, start, model[0])
            elif block['type'] == 'connected':
                model = self.model_2d[ind]
                if block['activation'] != 'linear':
                    start = load_fc(buf, start, model[0])
                else:
                    start = load_fc(buf, start, model)
            elif block['type'] == 'maxpool':
                pass
            elif block['type'] == 'reorg':
                pass
            elif block['type'] == 'route':
                pass
            elif block['type'] == 'shortcut':
                pass
            elif block['type'] == 'region':
                pass
            elif block['type'] == 'avgpool':
                pass
            elif block['type'] == 'softmax':
                pass
            elif block['type'] == 'cost':
                pass
            else:
                print('unknown type %s' % (block['type']))


    def save_weights(self, outfile, cutoff=0):
        if cutoff <= 0:
            cutoff = len(self.blocks)-1

        fp = open(outfile, 'wb')
        self.header[3] = self.seen
        header = self.header
        header.numpy().tofile(fp)

        ind = -1
        for blockId in range(1, cutoff+1):
            ind = ind + 1
            block = self.blocks[blockId]
            if block['type'] == 'convolutional':
                model = self.model_2d[ind]
                batch_normalize = int(block['batch_normalize'])
                if batch_normalize:
                    save_conv_bn(fp, model[0], model[1])
                else:
                    save_conv(fp, model[0])
            elif block['type'] == 'connected':
                model = self.model_2d[ind]
                if block['activation'] != 'linear':
                    save_fc(fc, model)
                else:
                    save_fc(fc, model[0])
            elif block['type'] == 'maxpool':
                pass
            elif block['type'] == 'reorg':
                pass
            elif block['type'] == 'route':
                pass
            elif block['type'] == 'shortcut':
                pass
            elif block['type'] == 'region':
                pass
            elif block['type'] == 'avgpool':
                pass
            elif block['type'] == 'softmax':
                pass
            elif block['type'] == 'cost':
                pass
            else:
                print('unknown type %s' % (block['type']))
        fp.close()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

def get_fine_tuning_parameters(model, ft_portion):
    if ft_portion == "complete":
        return model.parameters()

    elif ft_portion == "last_layer":
        ft_module_names = []
        ft_module_names.append('classifier')

        parameters = []
        for k, v in model.named_parameters():
            for ft_module in ft_module_names:
                if ft_module in k:
                    parameters.append({'params': v})
                    break
            else:
                parameters.append({'params': v, 'lr': 0.0})
        return parameters

    else:
        raise ValueError("Unsupported ft_portion: 'complete' or 'last_layer' expected")





def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


if __name__ == "__main__":
    
    model = LinkNet("cfg/yolo.cfg")
    print(model)

    x_2d = torch.randn(1, 3, 224, 224)
    x_3d = torch.randn(1, 3, 16, 224, 224)

    y2, y3 = model(x_2d, x_3d)

    print(y2.shape, y3.shape)

    # backbone_3d_weights = "weights/resnet-18-kinetics.pth"
    # model = nn.DataParallel(model, device_ids=None) # Because the pretrained backbone models are saved in Dataparalled mode
    # pretrained_3d_backbone = torch.load(backbone_3d_weights, map_location="cpu")
    # # print(pretrained_3d_backbone)

    # backbone_3d_dict = model.state_dict()

    # pretrained_3d_backbone_dict = {k: v for k, v in pretrained_3d_backbone['state_dict'].items() if k in backbone_3d_dict} # 1. filter out unnecessary keys
    
    # model.load_state_dict(backbone_3d_dict) # 3. load the new state dict
    
    # model = model.module # remove the dataparallel wrapper

    # model.load_weights("weights/yolo.weights")








