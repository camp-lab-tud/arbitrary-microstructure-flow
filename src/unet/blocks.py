import torch
import torch.nn as nn



class Block(nn.Module):
    """
    Basic convolutional block.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding_mode: str,
        activation: nn.Module
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding_mode = padding_mode
        self.activation=activation

        self._padding = get_padding(self.kernel_size)

        # layers
        self.conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            padding_mode=self.padding_mode,
            padding=self._padding,
            bias=False
        )
        self.norm = nn.GroupNorm(
            num_groups=1,
            num_channels=self.out_channels
        )

    def forward(self, x: torch.Tensor):

        x = self.conv(x)
        x = self.norm(x)
        out = self.activation(x)
        return out


class DoubleBlock(nn.Module):
    """
    Double convolutional block.
    """
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        kernel_size: int,
        padding_mode: str,
        activation: nn.Module
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.padding_mode = padding_mode
        self.activation = activation

        # layers
        self.block1 = Block(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=self.kernel_size,
            padding_mode=self.padding_mode,
            activation=activation
        )
        self.block2 = Block(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=self.kernel_size,
            padding_mode=self.padding_mode,
            activation=activation
        )

    def forward(self, x: torch.Tensor):
        x = self.block1(x)
        out = self.block2(x)
        return out



class Up(nn.Module):
    """
    Up-Convolution block.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: nn.Module
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation

        # layers
        self.conv = nn.ConvTranspose2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=2,
            stride=2
        )
        self.norm = nn.GroupNorm(
            num_groups=1,
            num_channels=self.out_channels
        )

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = self.norm(x)
        out = self.activation(x)
        return out


class Down(nn.Module):
    """
    Down-pooling block.
    """
    def __init__(
        self,
        in_channels: int,
        activation: nn.Module
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.activation = activation

        # layers
        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )
        self.norm = nn.GroupNorm(
            num_groups=1,
            num_channels=self.in_channels
        )

    def forward(self, x: torch.Tensor):
        x = self.pool(x)
        x = self.norm(x)
        out = self.activation(x)
        return out


class SelfAttention(nn.Module):
    """
    Self-Attention Block.
    """
    def __init__(
        self,
        in_channels: int,
        num_heads: int = 1
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_heads = num_heads

        # layers
        self.norm = nn.GroupNorm(
            num_groups=1,
            num_channels=self.in_channels
        )
        self.mha =  nn.MultiheadAttention(
            embed_dim=self.in_channels,
            num_heads=self.num_heads,
            batch_first=True
        )
        self.proj_out = zero_module(
            nn.Conv1d(
                in_channels=self.in_channels,
                out_channels=self.in_channels,
                kernel_size=1
            )
        )


    def forward(self, x: torch.Tensor):

        batch_size, channels, height, width = x.shape

        # normalization
        x_norm = self.norm(x)

        # attention values
        x_norm = x_norm.view(batch_size, channels, height*width) # shape: (B, C, H*W)
        x_norm = x_norm.swapaxes(1, 2) # shape: (B, H*W ,C)

        attn_val, _ = self.mha(
            query=x_norm,
            key=x_norm,
            value=x_norm,
            need_weights=False
        )

        # 1D-convolution
        attn_val = attn_val.swapaxes(2, 1) # shape: (B, C, H*W)
        
        h = self.proj_out(attn_val)
        h = h.reshape(batch_size, channels, height, width) # shape: (B, C, H, W)
        
        out = x + h
        return out


def zero_module(module: nn.Module):
    """
    Zero out parameters of module.
    
    See:
    Zhang, H., Dauphin, Y. N., & Ma, T. (2019).
    Fixup initialization: Residual learning without normalization.
    arXiv preprint arXiv:1901.09321.
    """

    for param in module.parameters():
        param.detach().zero_()
    
    return module


def get_padding(kernel_size: int):
    """Get padding."""

    if (kernel_size % 2) == 0:
        padding = (kernel_size // 2) - 1
    else:
        padding = kernel_size // 2
    
    return padding