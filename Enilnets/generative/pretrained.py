"""Pretrained-architecture skeletons -- randomly initialized layer stacks
matching well-known network shapes, for loading your own converted
pretrained weights via `.set_weights()`. Enilnets never downloads or embeds
any pretrained weights itself.
"""
from ..base import NeuralNet

_VGG16_BLOCKS = [
    [64, 64],
    [128, 128],
    [256, 256, 256],
    [512, 512, 512],
    [512, 512, 512],
]


def build_vgg16_feature_extractor(up_to_block=5, input_ch=3, init_method="he_normal"):
    """Standard VGG16 conv/pool skeleton (through `up_to_block`, 1-5),
    randomly initialized -- call `.set_weights()` on the returned model with
    your own converted pretrained weights before using it for real features.

    up_to_block: how many of VGG16's 5 conv blocks to include (1-5). Full
        VGG16 (up_to_block=5) has 13 conv layers: [64,64] -> pool,
        [128,128] -> pool, [256,256,256] -> pool, [512,512,512] -> pool,
        [512,512,512] -> pool, each conv k=3 with padding="same" so spatial
        size only changes at the pooling steps, matching the real VGG16
        architecture.

    Returns a `NeuralNet` with no classifier head -- `.Forward(x,
    training=False)` gives conv feature maps directly. Pass its bound
    `.Forward` method as `vgg_loss(..., vgg_features=model.Forward)`, or use
    the model itself as `inception_score(..., classifier=...)`-style
    duck-typed feature/logit source (any object with
    `.Forward(batch) -> (N, ...)` works).
    """
    if not (1 <= up_to_block <= 5):
        raise ValueError(f"up_to_block must be between 1 and 5, got {up_to_block}")
    model = NeuralNet()
    is_first_layer = True
    for block in _VGG16_BLOCKS[:up_to_block]:
        for out_ch in block:
            model.add_conv2d(
                input_ch if is_first_layer else None, out_ch, k=3, activation="relu",
                init_method=init_method, padding="same",
                input_size=(224, 224) if is_first_layer else None,
            )
            is_first_layer = False
        model.add_maxpool2d(2)
    return model
