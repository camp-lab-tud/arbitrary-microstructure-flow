import numpy as np
import torch



def get_flow_rate(
    rve_tensor: torch.Tensor,
    vel_tensor: torch.Tensor,
    cross_section_area: torch.Tensor | float = 1.
):
    """
    Get flow rate Q in a given microstructure. It is assumed that the fluid flows horizontally.

    Args:
        rve_tensor: 4D tensor with 0 and 1 respectively in fiber and fluid regions. Shape: (samples, 1, height, width).
        vel_tensor: 4D tensor with velocity values. Shape: (samples, channels, height, width).
        cross_section_area: cross-sectional area. Shape: (samples,).
    """

    if (rve_tensor.dim() == 3): rve_tensor = rve_tensor.unsqueeze(0)
    if (vel_tensor.dim() == 3): vel_tensor = vel_tensor.unsqueeze(0)

    assert (rve_tensor.dim() == 4) and (vel_tensor.dim() == 4), "Tensors must be 4D."
    assert set(
        torch.unique(rve_tensor.long()).tolist()
    ) == {0,1}, "`rve_tensor` should only contain zeros and ones."

    if isinstance(cross_section_area, float):
        cross_section_area = torch.tensor(
            data=[cross_section_area],
            device=rve_tensor.device
        )
    assert cross_section_area.dim() == 1


    mask = rve_tensor.clone()
    num_rows = mask.shape[-2]

    # sum of pixels in which fluid flows
    section_weights = torch.sum(mask, dim=-2, keepdim=True) # shape: (samples, 1, 1, width)
    section_fluid_vf = section_weights / num_rows

    # only consider velocities in fluid area
    masked_vel = mask * vel_tensor
    
    # sum of x-velocity for each cross-section
    masked_vel_x = masked_vel[:, [0], :, :] 
    section_sum_vel = torch.sum(masked_vel_x, dim=-2, keepdim=True) # shape: (samples, 1, 1, width)


    
    section_weights.squeeze_(dim=(1,2))
    section_fluid_vf.squeeze_(dim=(1,2))
    section_sum_vel.squeeze_(dim=(1,2))
    for val in [section_weights, section_fluid_vf, section_sum_vel]:
        assert val.dim() == 2

    # Mean-velocity at each cross-section
    section_mean_vel = section_sum_vel / section_weights

    # Fluid area at each cross-section
    fluid_areas = section_fluid_vf * cross_section_area.unsqueeze(-1)

    # Flow rate
    flow_rate = section_mean_vel * fluid_areas # shape: (samples, width)

    return flow_rate


def get_average_pressure(
    rve_tensor: torch.Tensor | np.ndarray,
    pres_tensor: torch.Tensor | np.ndarray
):  
    """
    Get average pressure at each section along length of microstructure.
    
    Args:
        rve_tensor: binary image of microstructure (with 1 in fluid areas); shape: (batch, height, width).
        pres_tensor: pressure field; shape: (batch, height, width).
    """
    def check_array(array: torch.Tensor | np.ndarray):
        if not isinstance(array, torch.Tensor):
            array = torch.tensor(array)

        if array.dim() == 4: # (B, C, H, W)
            array = array.squeeze(1)

        assert array.dim() == 3, "Array must be 3D."

        return array
    
    
    rve_tensor = check_array(rve_tensor)
    pres_tensor = check_array(pres_tensor)

    # sum of pressure for each cross-section along the horizontal direction
    section_sum_p = torch.sum(pres_tensor, dim=1)

    # TODO: add use of mask to ensure correctness of pressure sum

    # sum of pixels in which fluid flows
    section_weights = torch.sum(rve_tensor, dim=1)

    # mean-pressure at each cross-section
    section_mean_p = section_sum_p / section_weights

    return section_mean_p


def compute_permeability(
    flow_rate: torch.Tensor,
    pressure_drop: torch.Tensor,
    domain_length: torch.Tensor,
    cross_section_area: torch.Tensor,
    mu: float = 0.5
):
    """
    Compute permeability using Darcy's law.

    Args:
        flow_rate: 1D tensor of flow rates; shape: (samples,).
        pressure_drop: 1D tensor of pressure drops; shape: (samples,).
        domain_length: 1D tensor of domain lengths; shape: (samples,).
        cross_section_area: 1D tensor of cross-sectional areas; shape: (samples,)
        mu: dynamic viscosity of the fluid.
    """
    for val in (flow_rate, pressure_drop, domain_length, cross_section_area):
        assert val.dim() == 1, 'Inputs should be 1D arrays'

    # permeability (Darcy's law)
    out = flow_rate * mu * domain_length / (cross_section_area * pressure_drop)
    
    return out
