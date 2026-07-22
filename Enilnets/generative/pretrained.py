"""Pretrained-architecture skeletons -- randomly initialized layer stacks
matching well-known network shapes, for loading your own converted
pretrained weights via `.set_weights()`. Enilnets never downloads or embeds
any pretrained weights itself.
"""
from typing import Any

from ..core.base import NeuralNet

_VGG16_BLOCKS = [
    [64, 64],
    [128, 128],
    [256, 256, 256],
    [512, 512, 512],
    [512, 512, 512],
]


def build_vgg16_feature_extractor(up_to_block: int = 5, input_ch: int = 3,
                                   init_method: str = "he_normal") -> Any:
    """Randomly-initialized VGG16 conv/pool skeleton through `up_to_block`
    (1-5), with no classifier head. Call `.set_weights()` with your own
    converted pretrained weights before using it for real features.

    Full VGG16 (up_to_block=5) is 13 conv layers -- [64,64], [128,128],
    [256,256,256], [512,512,512], [512,512,512], each followed by a pool,
    every conv k=3 padding="same" so spatial size changes only at the pools.

    `.Forward(x, training=False)` returns conv feature maps directly; pass
    the bound method anywhere a duck-typed feature source is wanted (e.g.
    `vgg_loss(..., vgg_features=model.Forward)`)."""
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
