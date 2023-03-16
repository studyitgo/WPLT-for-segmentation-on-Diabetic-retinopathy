import os
import numpy as np
import torch
import math
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms
from utils import GradCAM, show_cam_on_image, center_crop_img
import matplotlib as mpl
from skimage.io import imread
from skimage.transform import resize
import cv2

load_model = torch.load('./logging/best__ACC=0.5783.pth')

class Bottleneck(nn.Module):
    """
    RexNeXt bottleneck type C
    """
    expansion = 4

    def __init__(self, inplanes, planes, baseWidth, cardinality, stride=1, downsample=None):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            baseWidth: base width.
            cardinality: num of convolution groups.
            stride: conv stride. Replaces pooling layer.
        """
        super(Bottleneck, self).__init__()

        D = int(math.floor(planes * (baseWidth / 64)))
        C = cardinality

        self.conv1 = nn.Conv2d(inplanes, D*C, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(D*C)
        self.conv2 = nn.Conv2d(D*C, D*C, kernel_size=3, stride=stride, padding=1, groups=C, bias=False)
        self.bn2 = nn.BatchNorm2d(D*C)
        self.conv3 = nn.Conv2d(D*C, planes * 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)# 相当于把宽高都除以2

        out += residual
        out = self.relu(out)

        return out
class ResNeXt_2(nn.Module):
    """
    ResNext optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1611.05431.pdf
    """

    def __init__(self, baseWidth, cardinality, layers, num_classes):
        """ Constructor
        Args:
            baseWidth: baseWidth for ResNeXt.
            cardinality: number of convolution groups.
            layers: config of layers, e.g., [3, 4, 6, 3]
            num_classes: number of classes
        """
        super(ResNeXt_2, self).__init__()
        block = Bottleneck

        self.cardinality = cardinality
        self.baseWidth = baseWidth
        self.num_classes = num_classes
        self.inplanes = 64
        self.output_size = 64

        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], 2)
        self.layer3 = self._make_layer(block, 256, layers[2], 2)
        self.layer4 = self._make_layer(block, 512, layers[3], 2)

        self.avgpool = nn.AvgPool2d(7)

        # 512 * block.expansion original
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        """ Stack n bottleneck modules where n is inferred from the depth of the network.
        Args:
            block: block type used to construct ResNext
            planes: number of output channels (need to multiply by block.expansion)
            blocks: number of blocks to be built
            stride: factor to reduce the spatial dimensionality in the first bottleneck of the block.
        Returns: a Module consisting of n sequential bottlenecks.
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, self.baseWidth, self.cardinality, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.baseWidth, self.cardinality))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        #        g_x = self.g_conv1(x)
        #        e_x = self.e_conv1(x)
        #
        #        g_se = self.g_se(g_x)
        #        e_se = self.e_se(e_x)
        #
        #        g_x = self.g_conv2(torch.cat((g_x, e_se), 1))
        #        e_x = self.e_conv2(torch.cat((e_x, g_se), 1))
        #
        #        g_x = self.g_bn(g_x)
        #        e_x = self.e_bn(e_x)
        #
        #        g_x = g_x.view(g_x.size(0), -1)
        #        e_x = e_x.view(e_x.size(0), -1)
        #
        #        g_x = self.g_fc(g_x)
        #        e_x = self.e_fc(e_x)

        return x
def resnext101(baseWidth=4, cardinality=32):
    """
    Construct ResNeXt-101.
    """
    model = ResNeXt_2(baseWidth, cardinality, [3, 4, 23, 3], 6)
    return model
model = resnext101()
#model = nn.DataParallel(model,device_ids = [0])
model_path = './checkpoints/1000_0.9906.pt'
# './checkpoints/1000_0.9906.pt'
# './logging/resnext101/best__ACC=0.5783.pth'
# './logging/best__ACC=0.3253.pth'
# './logging/resnext50/best_ACC=0.5301.pth'
# './logging/resnext50/best__ACC=0.5301.pth'
state_dict = torch.load(model_path)
model.load_state_dict(state_dict, strict=False)


class GradCamModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gradients = None
        self.tensorhook = []
        self.layerhook = []
        self.selected_out = None

        # PRETRAINED MODEL
        self.pretrained = model
        self.layerhook.append(self.pretrained.layer4.register_forward_hook(self.forward_hook()))

        for p in self.pretrained.parameters():
            p.requires_grad = True

    def activations_hook(self, grad):
        self.gradients = grad

    def get_act_grads(self):
        return self.gradients

    def forward_hook(self):
        def hook(module, inp, out):
            self.selected_out = out
            self.tensorhook.append(out.register_hook(self.activations_hook))

        return hook

    def forward(self, x):
        out = self.pretrained(x)
        return out, self.selected_out
gcmodel = GradCamModel().to('cuda:0')
####img-read####
c_path = 'F:\hcc/tf-Segmentation-of-Lesions-in-Diabetic-Retinopathy-Fundus-Images.-main/DR_data/Test/images/IDRiD_61.jpg'
#'F:\hcc/tf-Segmentation-of-Lesions-in-Diabetic-Retinopathy-Fundus-Images.-main/DR_data/Test/images/IDRiD_66.jpg'
#'F:\hcc\B. Disease Grading\B. Disease Grading/1. Original Images/b. Testing Set/IDRiD_040.jpg'
img = imread(c_path) #'bulbul.jpg'
img = resize(img, (224,224), preserve_range = True)# 224
img = np.expand_dims(img.transpose((2,0,1)),0)
img /= 255.0
mean = np.array([0.485, 0.456, 0.406]).reshape((1,3,1,1))
std = np.array([0.229, 0.224, 0.225]).reshape((1,3,1,1))
img = (img-mean)/std
inpimg = torch.from_numpy(img).to('cuda:0', torch.float32)

####Calculation of output and activation maps#####
out, acts = gcmodel(inpimg)
acts = acts.detach().cpu()
print(acts.size())
loss = nn.CrossEntropyLoss()(out,(torch.from_numpy(np.array([4]))).long().cuda().to('cuda:0'))
loss.backward()
grads = gcmodel.get_act_grads().detach().cpu()
print(grads.size())
pooled_grads = torch.mean(grads, dim=[0,2,3]).detach().cpu()
print(pooled_grads.size())
for i in range(acts.shape[1]):
    acts[:,i,:,:] += pooled_grads[i]

heatmap_j = torch.mean(acts, dim = 1).squeeze()
heatmap_j_max = heatmap_j.max(axis = 0)[0]
heatmap_j /= heatmap_j_max
print(heatmap_j.size())

heatmap_j = resize(heatmap_j,(224,224))# 224

cmap = mpl.cm.get_cmap('jet',256)
heatmap_j2 = cmap(heatmap_j,alpha = 0.2)

fig, axs = plt.subplots(1,1,figsize = (5,5))
axs.imshow((img*std+mean)[0].transpose(1,2,0))
axs.imshow(heatmap_j2)
# plt.savefig("./grad-camImgs/gc62.png")   # Save the image Note that before show(), otherwise show will recreate the new image.
plt.show()

heatmap_j3 = (heatmap_j > 0.75)
fig, axs = plt.subplots(1,1,figsize = (5,5))
print(((img*std+mean)[0].transpose(1,2,0)).shape)
hot = ((img*std+mean)[0].transpose(1,2,0))
hot[:,:,0] = hot[:,:,0]*heatmap_j3
hot[:,:,1] = hot[:,:,1]*heatmap_j3
hot[:,:,2] = hot[:,:,2]*heatmap_j3
axs.imshow(hot)
plt.show()