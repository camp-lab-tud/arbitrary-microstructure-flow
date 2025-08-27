from typing import Callable

import torch
from torch import linalg


__all__ = [
    'cost_function',
    'mae_loss',
    'normalized_mae_loss'
]


def cost_function(name: str) -> Callable[..., torch.Tensor]:
    func = eval(name)
    return func


def mae_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    reduce=True
) -> torch.Tensor:
    """
    Mean Absolute Error.

    `output`: output predicted by the ML model with same shape as `target`.\n
    `target`: target value with shape: (batch,channels,height,width).\n
    `reduce`: whether to take batch average.
    """

    dim = (-3,-2,-1) # everthing except batch dimension

    # error for each sample
    loss = torch.mean(
        torch.abs((output - target)),
        dim=dim
    )

    if reduce:
        # average
        loss = loss.mean()

    return loss


def normalized_mae_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    reduce=True
) -> torch.Tensor:
    """
    Normalized Mean Absolute Error.

    `output`: output predicted by the ML model with same shape as `target`.\n
    `target`: target value with shape: (batch,channels,height,width).\n
    `reduce`: whether to take batch average.
    """

    dim = (-3,-2,-1) # everthing except batch dimension

    mae = torch.mean(
        torch.abs((output - target)),
        dim=dim
    )
    weight = torch.mean(
        torch.abs(target),
        dim=dim
    )

    # error for each sample
    error = mae / weight # shape: (B,)

    if reduce:
        # average
        error = error.mean() # shape: (1)

    return error


def normalized_mse_loss(
    output: torch.Tensor,
    target: torch.Tensor
):
    """
    Normalize mean-square-error loss.

    `output`: output predicted by the ML model (shape: [B,C,W,H]).\n
    `target`: target value (shape: [B,C,W,H]).\n
    """

    n_samples = target.shape[0]

    # Sample-wise norms
    # shape: (batch, channels)
    smp_wise_diff_norm = linalg.matrix_norm(
        target - output,
        dim=(-2,-1)
    )**2

    smp_wise_target_norm = linalg.matrix_norm(
        target,
        dim=(-2,-1)
    )**2

    # loss for each channel
    out = (1/n_samples) * torch.sum(
        smp_wise_diff_norm/smp_wise_target_norm,
        dim=0
    )
    # out = (1/n_samples) * torch.sum(
    #     smp_wise_diff_norm,
    #     dim=0
    # )

    return out


def normalized_exp_loss(
    output: torch.Tensor,
    target: torch.Tensor
):
    """
    Normalized mean absolute error.

    `output`: output predicted by the ML model (shape: [B,C,W,H]).\n
    `target`: target value (shape: [B,C,W,H]).\n
    """

    mae_per_sample = torch.mean(
        torch.abs((output - target)),
        dim=(-3,-2,-1)
    )
    weight_per_sample = torch.mean(
        torch.abs(target),
        dim=(-3,-2,-1)
    )
    loss = (
        torch.exp(mae_per_sample) / torch.exp(weight_per_sample)
    ).mean()
    
    # mae = torch.mean(torch.abs((10**(1+output) - 10**(1+target))))
    # weight = torch.mean(torch.abs(10**(1+target)))

    # # mae = torch.mean(torch.abs((output - target)))
    # # weight = torch.mean(torch.abs(target))
    # loss = mae / weight

    return loss

def mass_conservation_loss(
    rve_mat: torch.Tensor,
    vel_output: torch.Tensor,
    vel_target: torch.Tensor,
    dir: str
):
    """
    Compute the mass conservation loss along each dimension.

    `rve_mat`: binary matrix with 0 in locations of fibers, and 1 everywhere else.\n
    `vel_output`: tensor with predicted velocity values.\n
    `vel_target`: tensor with target velocity values.\n
    `dim`: direction (x or y) along which to compute the flow rate.
    """

    for mat in (rve_mat, vel_output, vel_target):
        assert mat.dim() == 4, "Matrix should be 4D with shape (samples, channels, height, width)."
    rve_mat = rve_mat[:, 0, :, :]

    # velocity field difference    
    diff_vel = vel_target - vel_output

    cor_dict = {
        'x': 0,
        'y': 1
    }

    # set dimension variables
    dim = cor_dict[dir]
    other_dim = 1 if dim==0 else 0

    # velocity mismatch for this dimension
    vel_mis_for_dim = diff_vel[:, dim, :, :]

    # Velocity mismatch at cross-sections
    reduced_dim = 1 + other_dim # due to batch dimension before

    vel_mis_at_sections = torch.sum(
        vel_mis_for_dim,
        dim=reduced_dim
    )
    section_weights = torch.sum(
        rve_mat,
        dim=reduced_dim
    )
    # mean values
    vel_mis_at_sections = vel_mis_at_sections / section_weights


    # Mass conservation loss for each sample
    cons_loss_per_batch = torch.mean(
        vel_mis_at_sections**2,
        dim=-1
    )

    loss = torch.mean(cons_loss_per_batch)

    return loss


def mass_consv_loss(
    rve_mat: torch.Tensor,
    vel_output: torch.Tensor,
    vel_target: torch.Tensor,
    cross_section_area = 1.
):

    # velocity field difference 
    diff_vel = abs(vel_target - vel_output)

    flow_diff = _get_flow_rate(
        rve_mat,
        diff_vel,
        cross_section_area
    )
    down = _get_flow_rate(
        rve_mat,
        abs(vel_target),
        cross_section_area
    )

    diff_sample_wise = torch.mean(flow_diff / down) # torch.mean(flow_diff)


    loss = torch.mean(diff_sample_wise)

    return loss



def _get_flow_rate(
    rve_mat: torch.Tensor,
    vel_mat: torch.Tensor,
    cross_section_area: float
):
    """
    Get flow rate Q in a given microstructure. It is assumed that the fluid flows horizontally.

    `rve_mat`: binary matrix with 0 in locations of fibers, and 1 everywhere else (shape: [batch, 1, height, width]).\n
    `vel_mat`: matrix with velocity values (shape: [batch, 2, height, width]).\n
    `cross_section_area`: cross-sectional area.\n
    """

    assert set(torch.unique(rve_mat.long()).tolist()) == {0,1}, "Values in the `rve_mat` matrix should be 0 or 1."

    num_rows = rve_mat.shape[0]

    # only consider velocities in fluid area
    masked_vel = rve_mat * vel_mat[:, [0], :, :]

    # sum of velocities for each cross-section along the horizontal direction
    section_sum_vel = torch.sum(masked_vel, dim=-2, keepdim=True)

    # sum of pixels in which fluid flows
    section_weights = torch.sum(rve_mat, dim=-2, keepdim=True)

    # mean-velocities at each cross-section
    section_mean_vel = section_sum_vel / section_weights

    # area at each cross-section
    fluid_areas = (section_weights / num_rows) * cross_section_area

    # Flow rate
    flow_rate = section_mean_vel * fluid_areas
    flow_rate = flow_rate.squeeze()

    return flow_rate
