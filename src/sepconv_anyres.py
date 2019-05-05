import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
import numpy as np

def conv3x3(in_channels, out_channels, stride=1, 
            padding=1, bias=True, groups=1):    
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)

def conv7x7(in_channels, out_channels, stride=1, 
            padding=3, bias=True, groups=1):    
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=7,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)

def conv5x5(in_channels, out_channels, stride=1, 
            padding=2, bias=True, groups=1):    
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=5,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)

def upconv2x2(in_channels, out_channels, mode='transpose'):
    if mode == 'transpose':
        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2)
    else:
        # out_channels is always going to be the same
        # as in_channels
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            conv1x1(in_channels, out_channels))

def conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1)

def Subnet():
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
        torch.nn.ReLU(inplace=False),
        torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
        torch.nn.ReLU(inplace=False),
        torch.nn.Conv2d(in_channels=64, out_channels=
                        , kernel_size=3, stride=1, padding=1),
        torch.nn.ReLU(inplace=False),
        torch.nn.Upsample(scale_factor=2, mode='bilinear'),
        torch.nn.Conv2d(in_channels=51, out_channels=51, kernel_size=3, stride=1, padding=1)
    )
    
class DownConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels, pooling=True):
        super(DownConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling

        self.conv1 = conv3x3(self.in_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)
        self.conv3 = conv3x3(self.out_channels, self.out_channels)

        if self.pooling:
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        before_pool = x
        if self.pooling:
            x = self.pool(x)
        return x, before_pool


class UpConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels, 
                 merge_mode='concat', up_mode='transpose'):
        super(UpConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode

        self.upconv = upconv2x2(self.in_channels, self.out_channels, 
            mode=self.up_mode)

        if self.merge_mode == 'concat':
            self.conv1 = conv3x3(
                2*self.out_channels, self.out_channels)
        else:
            # num of input channels to conv2 is same
            self.conv1 = conv3x3(self.out_channels, self.out_channels)
        # self.conv2 = conv3x3(self.out_channels, self.out_channels)


    def forward(self, from_down, from_up):
        """ Forward pass
        Arguments:
            from_down: tensor from the encoder pathway
            from_up: upconv'd tensor from the decoder pathway
        """
        from_up = self.upconv(from_up)
        if self.merge_mode == 'concat':
            x = torch.cat((from_up, from_down), 1)
        else:
            x = from_up + from_down
        x = F.relu(self.conv1(x))
        #x = F.relu(self.conv2(x))
        return x

#class SepConv(nn.Module):
#    def __init__(self):
#        super(SepConv, self).__init__()
#       
#    def forward(self, imgs, vers, hors):
#        b, c, H, W = imgs.size()
#        res = []
#        for i in range(b):
#            a = vers[i].permute(1, 2, 0).contiguous().view(H, 1, W, H) @ imgs[i] * hors[i].permute(1, 2, 0).contiguous().view(H, 1, W, W)
#            res.append(a.sum(dim=-1).permute(1,0 , 2).contiguous().view(1, c, H, W))
#        return torch.cat(res)
        
class NewSepConv(nn.Module):
    def __init__(self):
        super(NewSepConv, self).__init__()
       
    def forward(self, imgs, vers, hors):
        b, c, H, W = imgs.size()
        _, s, _, _ = vers.size()
        
        all_kernels = vers.permute(0,2,3,1).contiguous().view(b, H, W, s, 1) @ hors.permute(0,2,3,1).contiguous().view(b, H, W, 1, s)
        all_kernels = all_kernels.view(b, 1, H, W, s, s)
        imgs = torch.nn.ReplicationPad2d([8,8,8,8]) (imgs)
        all_patches = []
        for i in range(H):
            for j in range(W):
                all_patches.append(imgs[:,:,i:i+17, j:j+17].contiguous().view(b, c, 17, 17, 1))
        all_patches = torch.cat(all_patches, dim=-1).view(b, c, 17, 17, H, W).permute(0,1,4,5,2,3).contiguous()
        #print (124,all_patches.size(), all_kernels.size())
        return (all_patches * all_kernels).sum(dim=-1).sum(dim=-1)
    
class SepConvNet(nn.Module):


    def __init__(self, num_classes=32, in_channels=6, depth=6, 
                 start_filts=32, up_mode='upsample', 
                 merge_mode='concat'):
        super(SepConvNet, self).__init__()

        if up_mode in ('transpose', 'upsample'):
            self.up_mode = up_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for "
                             "upsampling. Only \"transpose\" and "
                             "\"upsample\" are allowed.".format(up_mode))
    
        if merge_mode in ('concat', 'add'):
            self.merge_mode = merge_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for"
                             "merging up and down paths. "
                             "Only \"concat\" and "
                             "\"add\" are allowed.".format(up_mode))

        # NOTE: up_mode 'upsample' is incompatible with merge_mode 'add'
        if self.up_mode == 'upsample' and self.merge_mode == 'add':
            raise ValueError("up_mode \"upsample\" is incompatible "
                             "with merge_mode \"add\" at the moment "
                             "because it doesn't make sense to use "
                             "nearest neighbour to reduce "
                             "depth channels (by half).")

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.start_filts = start_filts
        self.depth = depth

        self.down_convs = []
        self.up_convs = []

        # create the encoder pathway and add to a list
        for i in range(depth):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts*(2**i)
            pooling = True if i < depth-1 else False

            down_conv = DownConv(ins, outs, pooling=pooling)
            self.down_convs.append(down_conv)

        # create the decoder pathway and add to a list
        # - careful! decoding only requires depth-1 blocks
        for i in range(depth-1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(ins, outs, up_mode=up_mode,
                merge_mode=merge_mode)
            self.up_convs.append(up_conv)

        self.conv_h1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=outs, out_channels=outs, kernel_size=3, stride=1, padding=1),
            torch.nn.Conv2d(in_channels=outs, out_channels=32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            #torch.nn.Upsample(scale_factor=2, mode='bilinear'),
            torch.nn.Conv2d(in_channels=32, out_channels=17, kernel_size=3, stride=1, padding=1)
        )
        self.conv_h2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=outs, out_channels=outs, kernel_size=3, stride=1, padding=1),
            torch.nn.Conv2d(in_channels=outs, out_channels=32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            #torch.nn.Upsample(scale_factor=2, mode='bilinear'),
            torch.nn.Conv2d(in_channels=32, out_channels=17, kernel_size=3, stride=1, padding=1)
        )
        self.conv_w1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=outs, out_channels=outs, kernel_size=3, stride=1, padding=1),
            torch.nn.Conv2d(in_channels=outs, out_channels=32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            #torch.nn.Upsample(scale_factor=2, mode='bilinear'),
            torch.nn.Conv2d(in_channels=32, out_channels=17, kernel_size=3, stride=1, padding=1)
        )
        self.conv_w2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=outs, out_channels=outs, kernel_size=3, stride=1, padding=1),
            torch.nn.Conv2d(in_channels=outs, out_channels=32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            #torch.nn.Upsample(scale_factor=2, mode='bilinear'),
            torch.nn.Conv2d(in_channels=32, out_channels=17, kernel_size=3, stride=1, padding=1)
        )
        #conv1x1(outs, self.num_classes)

        # add the list of modules to current module
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal(m.weight)
            init.constant(m.bias, 0)


    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)


    def forward(self, imgs, ts):
        encoder_outs = []
        x = imgs
        # encoder pathway, save outputs for merging
        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)
        #print (x.size(), ts.size())
        #p = Variable(torch.ones(ts.size(0), 1, x.size(2), x.size(3)).cuda(), requires_grad=False) * ts.view(ts.size(0), 1, 1, 1)
        #p = Variable(torch.ones(ts.size(0), 1, x.size(2), x.size(3)).cuda() * ( 1-ts.view(ts.size(0),1, 1, 1)
        x = torch.cat([Variable(torch.ones(ts.size(0), 1, x.size(2), x.size(3)).cuda(), requires_grad=False) * ts.view(ts.size(0), 1, 1, 1), 
                       x[:,1:x.size(1)-1,...], 
                       Variable(torch.ones(ts.size(0), 1, x.size(2), x.size(3)).cuda(), requires_grad=False) * ( 1-ts.view(ts.size(0),1, 1, 1))], dim=1)
        
        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i+2)]
            x = module(before_pool, x)
        
        # No softmax is used. This means you need to use
        # nn.CrossEntropyLoss is your training script,
        # as this module includes a softmax already.
        h1 = self.conv_h1(x)
        h2 = self.conv_h2(x)
        w1 = self.conv_w1(x)
        w2 = self.conv_w2(x)
        
        return NewSepConv()(imgs[:,:3,...], h1, w1) + NewSepConv()(imgs[:,3:,...], h2, w2)
