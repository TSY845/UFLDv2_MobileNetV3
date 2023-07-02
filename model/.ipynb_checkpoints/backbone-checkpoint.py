import torch, pdb
import torchvision
import torch.nn.modules


class vgg16bn(torch.nn.Module):
    def __init__(self, pretrained=False):
        super(vgg16bn, self).__init__()
        model = list(
            torchvision.models.vgg16_bn(
                pretrained=pretrained).features.children())
        model = model[:33] + model[34:43]
        self.model = torch.nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class resnet(torch.nn.Module):
    def __init__(self, layers, pretrained=False):
        super(resnet, self).__init__()
        if layers == '18':
            model = torchvision.models.resnet18(pretrained=pretrained)
        elif layers == '34':
            model = torchvision.models.resnet34(pretrained=pretrained)
        elif layers == '50':
            model = torchvision.models.resnet50(pretrained=pretrained)
        elif layers == '101':
            model = torchvision.models.resnet101(pretrained=pretrained)
        elif layers == '152':
            model = torchvision.models.resnet152(pretrained=pretrained)
        elif layers == '50next':
            model = torchvision.models.resnext50_32x4d(pretrained=pretrained)
        elif layers == '101next':
            model = torchvision.models.resnext101_32x8d(pretrained=pretrained)
        elif layers == '50wide':
            model = torchvision.models.wide_resnet50_2(pretrained=pretrained)
        elif layers == '101wide':
            model = torchvision.models.wide_resnet101_2(pretrained=pretrained)
        elif layers == '34fca':
            model = torch.hub.load('cfzd/FcaNet', 'fca34', pretrained=True)
        else:
            raise NotImplementedError

        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x2, x3, x4


class mobilenet_v3(torch.nn.Module):
    def __init__(self, weights=None, ):
        print("loading the mobilenetv3 model ...")
        super(mobilenet_v3, self).__init__()
        model = torchvision.models.mobilenet_v3_large(weights=None)
        self.layer0 = model.features[:5]
        self.layer1 = model.features[5:8]
        self.layer2 = model.features[8:13]
        self.layer3 = model.features[13:16]

    def forward(self, x):
        x = self.layer0(x)
        x2 = self.layer1(x)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        return x2, x3, x4
