import torchvision as tv
from torch import nn
from src.models import backbone_registry


@backbone_registry.BackBoneRegistry.register("VGG16")
class VGG16(nn.Module):
    """
    VGG16 model class.

    This class initializes a VGG16 model, optionally with pre-trained
    weights, and can be used as a backbone.
    """

    def __init__(self, weights: bool = True):
        """
        Initializes the VGG16 model.

        @param bool weights: Whether to load pre-trained weights.
        Defaults to True.
        """
        super(VGG16, self).__init__()
        self.model = tv.models.vgg16(
            weights=tv.models.VGG16_Weights.DEFAULT if weights else None
        )

    def forward(self, x):
        """
        Forward pass of the VGG16 model.

        @param x: Input tensor.
        @return: Output tensor.
        """
        return self.model(x)

    def pool_to_feature_vector(self, rois):
        """
        Flatten the input tensor.

        @param rois: Input tensor from RoI pooling.
        @return: Flattened tensor.
        """
        return rois.view(rois.size(0), -1)


@backbone_registry.BackBoneRegistry.register("RESNET34")
class RESNET34(nn.Module):
    """
    RESNET34 model class.

    This class initializes a RESNET34 model, optionally with pre-trained
    weights, and can be used as a backbone.
    """

    def __init__(self, weights: bool = True):
        """
        Initializes the RESNET34 model.

        @param bool weights: Whether to load pre-trained weights.
        Defaults to True.
        """
        super(RESNET34, self).__init__()
        self.model = tv.models.resnet34(
            weights=tv.models.ResNet34_Weights.DEFAULT if weights else None
        )

    def forward(self, x):
        """
        Forward pass of the RESNET34 model.

        @param x: Input tensor.
        @return: Output tensor.
        """
        return self.model(x)


@backbone_registry.BackBoneRegistry.register("RESNET50")
class RESNET50(nn.Module):
    """
    RESNET50 model class.

    This class initializes a RESNET50 model, optionally with pre-trained
    weights, and can be used as a backbone.
    """

    def __init__(self, weights: bool = True):
        """
        Initializes the VGG16 model.

        @param bool weights: Whether to load pre-trained weights.
        Defaults to True.
        """
        super(RESNET50, self).__init__()
        self.model = tv.models.resnet50(
            weights=tv.models.ResNet50_Weights.DEFAULT if weights else None
        )

    def forward(self, x):
        """
        Forward pass of the VGG16 model.

        @param x: Input tensor.
        @return: Output tensor.
        """
        return self.model(x)


@backbone_registry.BackBoneRegistry.register("RESNET101")
class RESNET101(nn.Module):
    """
    RESNET101 model class.

    This class initializes a RESNET101 model, optionally with pre-trained
    weights, and can be used as a backbone.
    """

    def __init__(self, weights: bool = True):
        """
        Initializes the RESNET101 model.

        @param bool weights: Whether to load pre-trained weights.
        Defaults to True.
        """
        super(RESNET101, self).__init__()
        self.model = tv.models.resnet101(
            weights=tv.models.ResNet101_Weights.DEFAULT if weights else None
        )

    def forward(self, x):
        """
        Forward pass of the RESNET101 model.

        @param x: Input tensor.
        @return: Output tensor.
        """
        return self.model(x)
