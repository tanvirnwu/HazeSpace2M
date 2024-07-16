from utils.config import *
import torch.nn as nn
from torchvision.models import (convnext_large,densenet201, resnet152,
                                ConvNeXt_Large_Weights, DenseNet201_Weights, ResNet152_Weights)


def ConvNextLarge():
    # ================= Loading ConvNextLarge Model's Pretrained Weights  =================
    convnextlarge = convnext_large(weights=ConvNeXt_Large_Weights.IMAGENET1K_V1)
    # print(convnextlarge)
    # print(len(convnextlarge.classifier))
    # print(convnextlarge.classifier)

    # ================ Freezing Layers of ConvNextLarge Model =================
    # Freeze all layers in the model
    for param in convnextlarge.parameters():
        param.requires_grad = False

    # Unfreeze the last classifier layer
    for param in convnextlarge.classifier[2].parameters():
        param.requires_grad = True

    # ================= Modifying ConvNextLarge Model's Classifier Layer =================
    in_features = convnextlarge.classifier[2].in_features
    convnextlarge.classifier[2] = nn.Linear(in_features, output_shape)

    return convnextlarge.to(device)




def DenseNet201():
    # ================= Loading ConvNextLarge Model's Pretrained Weights  =================
    densenet = densenet201(weights=DenseNet201_Weights.IMAGENET1K_V1)
    # print(densenet)
    # print(len(densenet.classifier))
    # print(densenet.classifier)

    # ================ Freezing Layers of ConvNextLarge Model =================
    # Freeze all layers in the model
    for param in densenet.parameters():
        param.requires_grad = False

    # Unfreeze the last classifier layer
    for param in densenet.classifier.parameters():
        param.requires_grad = True

    # ================= Modifying ConvNextLarge Model's Classifier Layer =================
    in_features = densenet.classifier.in_features
    densenet.classifier = nn.Linear(in_features, output_shape)

    return densenet.to(device)


def ResNet152():
    # ================= Loading ConvNextLarge Model's Pretrained Weights  =================
    resNet152 = resnet152(weights=ResNet152_Weights.IMAGENET1K_V1)
    # print(resNet101)
    # print(len(resNet101.classifier))
    # print(resNet101.classifier)

    # ================ Freezing Layers of ConvNextLarge Model =================
    # Freeze all layers in the model
    for param in resNet152.parameters():
        param.requires_grad = False

    # Unfreeze the last classifier layer
    for param in resNet152.fc.parameters():
        param.requires_grad = True

    # ================= Modifying ConvNextLarge Model's Classifier Layer =================
    in_features = resNet152.fc.in_features
    resNet152.fc = nn.Linear(in_features, output_shape)

    return resNet152.to(device)



