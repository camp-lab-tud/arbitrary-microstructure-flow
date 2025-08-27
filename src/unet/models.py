from typing import Literal

import torch
import torch.nn as nn

from .blocks import DoubleBlock, Down, Up, SelfAttention
from .blocks import zero_module, get_padding


_activ_type = Literal['silu', 'relu', 'leakyrelu','softplus']


class UNet(nn.Module):
    """
    U-Net model architecture, with (optionally) attention mechanism.
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        features: list[int] = [32, 64, 128, 256],
        kernel_size = 3,
        padding_mode = 'reflect',
        activation: _activ_type = 'silu',
        final_activation: _activ_type = None,
        attention: str = ''
    ) -> None:
        super().__init__()

        self.in_channels: int = in_channels
        self.out_channels: int = out_channels
        self.features: list[int] = features
        self.kernel_size: int = kernel_size
        self.padding_mode: str = padding_mode
        
        self._activation = activation
        self.activation = activation_function(activation)

        self._final_activation = final_activation
        self.final_activation = activation_function(final_activation)

        self.attention = attention

        self._attention_heads = eval_expression(
            expr=self.attention,
            max_levels=len(self.features)
        )

        self._padding = get_padding(self.kernel_size)

        """Model components"""
        # encoder
        self.encoder = build_encoder(
            features = [self.in_channels] + self.features,
            attention_heads=self._attention_heads,
            kernel_size=self.kernel_size,
            padding_mode=self.padding_mode,
            activation=self.activation
        )

        # bottleneck
        self.bottleneck = DoubleBlock(
            in_channels=self.features[-1],
            mid_channels = 2 * self.features[-1],
            out_channels = 2 * self.features[-1],
            kernel_size=self.kernel_size,
            padding_mode=self.padding_mode,
            activation=self.activation
        )

        # decoder
        self.decoder = build_decoder(
            features=list(reversed(self.features)),
            attention_heads=list(reversed(self._attention_heads)),
            kernel_size=self.kernel_size,
            padding_mode=self.padding_mode,
            activation=self.activation
        )

        # output part
        self.final_conv = zero_module(
                nn.Conv2d(
                in_channels=features[0],
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                padding=self._padding,
                padding_mode=self.padding_mode
            )
        )


    def forward(self, x: torch.Tensor):
        """
        `x`: input with shape: (batch, channels, height, width).
        """

        
        """Encoder"""
        skip_connections: list[torch.Tensor] = []

        # loop over each level in encoder
        for level, module_list in enumerate(self.encoder):
            # list of: convolution, attention (or not), pooling.

            conv_block, attn_block, pool_layer = module_list

            x = conv_block(x) # convolution

            if attn_block is not None:
                x = attn_block(x) # attention
            skip_connections.append(x) # cache output before pooling

            x = pool_layer(x)


        """Bottleneck"""
        x = self.bottleneck(x)

        
        """Decoder"""
        skip_connections.reverse() # reverse list

        for level, module_list in enumerate(self.decoder):
            # list of: up-sampling, convolution, attention (or not)

            up_conv, conv_block, attn_block = module_list
            skip = skip_connections[level]

            x = up_conv(x) # up-sampling

            x = torch.cat((skip, x), dim=1)
            x = conv_block(x) # convolution

            if attn_block is not None:
                x = attn_block(x) # attention


        # output
        x = self.final_conv(x)
        out = self.final_activation(x)

        return out



def build_encoder(
    features: list[int],
    attention_heads: list[int | None],
    kernel_size: int,
    padding_mode: str,
    activation: nn.Module
) -> nn.ModuleList:
    """
    Build encoder part of U-Net.
    """

    assert len(features) == 1 + len(attention_heads)

    level_blocks = nn.ModuleList()

    in_channels = features[0]
    for k, next_channels in enumerate(features[1:]):
        
        # Convolution block
        conv_blk = DoubleBlock(
            in_channels=in_channels,
            mid_channels=next_channels,
            out_channels=next_channels,
            kernel_size=kernel_size,
            padding_mode=padding_mode,
            activation=activation
        )
        
        # Attention block
        attn_heads = attention_heads[k]
        if attn_heads is None:
            attn_blk = None
        else:
            attn_blk = SelfAttention(
                in_channels=next_channels,
                num_heads=attn_heads
            )

        # Pooling layer
        pool_lyr = Down(
            in_channels=next_channels,
            activation=activation
        )

        # sequence of operation for level
        level_blk = nn.ModuleList([conv_blk, attn_blk, pool_lyr])
        level_blocks.append(level_blk)

        in_channels = next_channels

    return level_blocks


def build_decoder(
    features: list[int],
    attention_heads: list[int | None],
    kernel_size: int,
    padding_mode: str,
    activation: nn.Module
) -> nn.ModuleList:
    """
    Build decoder part of U-Net.
    """

    assert len(features) == len(attention_heads)

    level_blocks = nn.ModuleList()

    for k, next_channels in enumerate(features):
        
        in_channels = 2 * next_channels

        # Up-sampling block
        up_blk = Up(
            in_channels=in_channels,
            out_channels=next_channels,
            activation=activation
        )

        # Convolution block
        conv_blk = DoubleBlock(
            in_channels=in_channels,
            mid_channels=next_channels,
            out_channels=next_channels,
            kernel_size=kernel_size,
            padding_mode=padding_mode,
            activation=activation
        )
        
        # Attention block
        attn_heads = attention_heads[k]
        if attn_heads is None:
            attn_blk = None
        else:
            attn_blk = SelfAttention(
                in_channels=next_channels,
                num_heads=attn_heads
            )

        # sequence of operation for level
        level_blk = nn.ModuleList([up_blk, conv_blk, attn_blk])
        level_blocks.append(level_blk)

    return level_blocks


def activation_function(name: str | None) -> nn.Module:
    # Get activation function

    if name is not None:
        name = name.strip().lower()

    if not name:
        return nn.Identity()
    elif name == 'silu':
        return nn.SiLU()
    elif name == 'relu':
        return nn.ReLU()
    elif name == 'leakyrelu':
        return nn.LeakyReLU()
    elif name == 'softplus':
        return nn.Softplus()
    else:
        raise NotImplementedError


def eval_expression(expr: str, max_levels: int) -> list[int | None]:
    """
    Evaluate expression for attention mechanism.

    `expr`: The expression has the format `start.end.heads`.\n

    \t An empty string means no attention.\n
    \t '1.1.1' means: attention only at the 1st level, with 1 head.\n
    \t '3.5.2' means: attention from the 3rd to 5th level, with 2 heads.\n
    \t '3..2' means: attention from the 3rd level to the maximum number of levels, with 2 heads.\n
    
    `max_levels`: maximum number of levels in U-Net.

    Returns a list indicating the number of attention heads at each level.
    """
    expr = expr.strip()

    out = [None for _ in range(max_levels)]

    if not expr:
        return out
    
    try:
        start_level, end_level, num_heads = expr.split('.')
        if not end_level.strip():
            # if no value
            end_level = str(max_levels)

        # convert to integers
        start_level, end_level, num_heads = [
            int(val) for val in [start_level, end_level, num_heads]
        ]

        # convert to zero-based indexing for levels
        start_level = -1 + start_level
        end_level = -1 + end_level

        # assign heads
        for i in range(start_level, 1 + end_level):
            out[i] = num_heads

    except:
        raise ValueError('Check validity of expression string.')

    return out
