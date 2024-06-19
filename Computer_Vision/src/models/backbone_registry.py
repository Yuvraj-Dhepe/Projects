import torch
import torch.nn as nn
from typing import Dict, Type, Tuple, Optional, Callable


class BackBoneRegistry:
    """
    Registry for CNN backbones.

    This class provides methods to register and create CNN backbones for
    use in Faster R-CNN.
    """

    registry: Dict[str, Type[nn.Module]] = {}

    @classmethod
    def register(cls, name: str):
        """
        Registers a new backbone in the registry.

        @param str name: The name of the backbone to register.
        @return: A decorator for registering the backbone class.
        """

        def inner_wrapper(wrapped_class: Type[nn.Module]):
            cls.registry[name] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def create_backbone(
        cls,
        name: str,
        input_shape: Tuple[int, int, int],
        pool_to_feature_vector: bool = False,
        **kwargs
    ) -> Tuple[torch.Size, nn.Module, Optional[Callable]]:
        """
        Creates a backbone and extracts its feature layers.

        @param str name: The name of the backbone to create.
        @param Tuple[int, int, int] input_shape: Shape of the input
        tensor (C, H, W).
        @param bool pool_to_feature_vector: Whether to return the
        pool_to_feature_vector function or not.
        @param kwargs: Additional arguments to pass to the backbone
        class constructor.
        @return Tuple[torch.Size, nn.Module, Optional[Callable]]: A tuple
        containing
        the shape of the last layer of the extracted backbone, the feature
        extractor
        network, and the pool_to_feature_vector function (if requested).
        """
        model_class = cls.registry[name]
        model = model_class(**kwargs)
        # Initialize the model, as the model_class returns
        # a self.model variable.
        if name == "VGG16":
            last_layer_shape, backbone = extract_backbone_layers(
                model.model, input_shape, until_layer=-1
            )

        elif name in ["RESNET50", "RESNET101", "RESNET34"]:
            last_layer_shape, backbone = extract_backbone_layers(
                model.model, input_shape, until_layer=-2
            )

        pool_to_feature_vector_func = None
        if pool_to_feature_vector:
            pool_to_feature_vector_func = model.pool_to_feature_vector

        return last_layer_shape, backbone, pool_to_feature_vector_func


def extract_backbone_layers(
    model: nn.Module, input_shape: Tuple[int, int, int], until_layer: int = -1
) -> Tuple[torch.Size, nn.Module]:
    """
    Extracts backbone layers from a CNN network for use in Faster R-CNN.

    This function extracts all the layers except the classification head and
    the ending pooling layers from the provided CNN model.

    @param nn.Module model: A pre-trained CNN model.
    @param Tuple[int, int, int] input_shape: Shape of the input
    tensor (C, H, W).
    @return Tuple[torch.Size, nn.Module]: A tuple containing the shape of
    the last layer of the extracted backbone, and the
    feature extractor network.
    """
    try:
        # VGG 16 children are accessible via the model.features
        features = model.features
    except Exception:
        # ResNet Family children are directly accessible via the model class
        features = model

    # Remove the classification head and the ending pooling layers
    feature_extractor = nn.Sequential(*list(features.children())[:until_layer])

    # Dummy input to get the shape of the last layer
    dummy_input = torch.randn(
        1, *input_shape
    )  # assuming input shape is (C, H, W)
    with torch.no_grad():
        output = feature_extractor(dummy_input)

    last_layer_shape = output.shape

    return last_layer_shape, feature_extractor


# Example usage
if __name__ == "__main__":
    # Define the input shape
    input_shape = (3, 224, 224)  # (C, H, W)

    # Get the VGG16 backbone
    last_layer_shape, backbone = BackBoneRegistry.create_backbone(
        "VGG16", input_shape
    )

    print("Last layer shape:", last_layer_shape)
    print("Feature extractor network:", backbone)
