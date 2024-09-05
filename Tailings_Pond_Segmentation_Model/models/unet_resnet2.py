import torch
import torch.nn as nn

try:
    from .resnet import resnet50_v1b
except:
    from resnet import resnet50_v1b

class SegBaseModel(nn.Module):
    r"""Base Model for Semantic Segmentation

    Parameters
    ----------
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    """

    def __init__(self, nclass, aux, backbone='resnet50', jpu=False, pretrained_base=True, **kwargs):
        super(SegBaseModel, self).__init__()
        dilated = False if jpu else True
        self.aux = aux
        self.nclass = nclass
        if backbone == 'resnet50':
            self.pretrained = resnet50_v1b(pretrained=pretrained_base, dilated=dilated, **kwargs)

        # self.jpu = JPU([512, 1024, 2048], width=512, **kwargs) if jpu else None

    def base_forward(self, x):
        """forwarding pre-trained network"""
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        c1 = self.pretrained.relu(x)
        c2 = self.pretrained.maxpool(c1)
        c2 = self.pretrained.layer1(c2)
        c3 = self.pretrained.layer2(c2)
        c4 = self.pretrained.layer3(c3)
        c5 = self.pretrained.layer4(c4)

        return c1, c2, c3, c4, c5



def resnet50(pretrained, nclasses):
    model = SegBaseModel(aux=False, pretrained_base=pretrained, nclass=nclasses)
    return model

class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1  = nn.Conv2d(in_size, out_size, kernel_size = 3, padding = 1)
        self.conv2  = nn.Conv2d(out_size, out_size, kernel_size = 3, padding = 1)
        self.up     = nn.UpsamplingBilinear2d(scale_factor = 2)
        self.relu   = nn.ReLU(inplace = True)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs
            
class Unet_resnet(nn.Module):
    def __init__(self, nclass = 21, backbone = 'resnet50', pretrained_base=False):
        super(Unet_resnet, self).__init__()
        self.resnet    = resnet50(pretrained_base, nclass)
        in_filters  = [192, 512, 1024, 3072]
        out_filters = [64, 128, 256, 512]
        
        # upsampling
        # 64,64,512
        self.up_concat4 = unetUp(in_filters[3], out_filters[3])
        # 128,128,256
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])
        # 256,256,128
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        # 512,512,64
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])
        
        self.up_conv = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor = 2), 
            nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
            nn.ReLU(),
        )
        
        self.final = nn.Conv2d(out_filters[0], nclass, 1)
        
    def forward(self, inputs):
        [feat1, feat2, feat3, feat4, feat5] = self.resnet.base_forward(inputs)
        up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)
        up1 = self.up_conv(up1)
        final = self.final(up1)
        
        return final
    
