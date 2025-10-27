from typing import Literal, Callable
import time

import numpy as np
import torch

from .predictor import VelocityPredictor, PressurePredictor
from .physics import get_flow_rate, get_average_pressure


class SlidingWindow:
    """
    Class implementing the sliding window procedure to extend
    the prediction of velocity & pressure fields from square to rectangular microstructures.
    """

    def __init__(
        self,
        velocity_model: VelocityPredictor = None,
        pressure_model: PressurePredictor = None,
        window_size: tuple[int,int] = (256, 256),
        step_size: int = 2,
        batch_size: int = 40
    ):
        self.velocity_model = velocity_model
        self.pressure_model = pressure_model
        self.window_size = window_size
        self.step_size = step_size
        self.batch_size = batch_size

    @torch.no_grad()
    def predict_velocity(
        self,
        img: torch.Tensor,
        *,
        window_size: tuple[int,int] = None,
        step_size: int = None,
        batch_size: int = None
    ) -> dict[str, torch.Tensor]:
        """
        Predict velocity field for larger (rectangular) microstructure image.
        
        Args:
            img: microstructure image, 2D tensor with 0 and 1 respectively in
                fiber and fluid regions. Shape: (height, width).
        """
        
        if self.velocity_model is None:
            raise ValueError('Velocity model was not defined.')

        img = img.squeeze()
        assert img.dim() == 2, 'Tensor should be 2D with shape (nrows, ncols).'

        _window_size = window_size if window_size else self.window_size
        _step_size = step_size if step_size else self.step_size
        _batch_size = batch_size if batch_size else self.batch_size


        start_time = time.time()
        """1. Fragment image into squares"""
        sub_img_list, frame_positions = self.fragment_image(
            img=img,
            window_size=_window_size,
            step_size=_step_size
        )

        # reshape images into (1, 1, nrows, ncols)
        sub_img_list = [val.unsqueeze(0).unsqueeze(0) for val in sub_img_list]

        # batch images; each batch has the shape (batch_size, 1, nrows, ncols)
        batch_list = batch_images(
            img_list=sub_img_list,
            batch_size=_batch_size
        )

        """2. Prediction on square sub-images"""
        preds = self._predict_velocity(batch_list)


        """3. Overlay velocity fields"""
        result = self.overlay_fields(
            flow_fields=preds,
            frame_positions=frame_positions,
            domain_size=img.shape
        )

        """4. Velocity correction"""
        prediction_corrected = correct_velocity_field(
            img.unsqueeze(0).unsqueeze(0),
            result['average'].unsqueeze(0)
        ).squeeze(0)

        dtime = time.time() - start_time
        print(f'Prediction in {dtime:.2f}s.')

        out = {
            'microstructure': img.unsqueeze(0), # shape: (1, height, width)

            'prediction': prediction_corrected, # shape: (channels, height, width)
            'prediction_naive': result['average'], # shape: (channels, height, width)
            'window_predictions': result['values'], # shape: (channels, height, width, num of windows)
            'window_positions': frame_positions, # length: num of windows
        }
        return out
    
    @torch.no_grad()
    def predict_pressure(
        self,
        img: torch.Tensor,
        x_length: torch.Tensor,
        *,
        window_size: tuple[int,int] = None,
        step_size: int = None,
        batch_size: int = None
    ) -> dict[str, torch.Tensor]:
        """
        Predict pressure field for rectangular microstructure.

        Args:
            img: microstructure image, 2D tensor with 0 and 1 respectively in fiber and fluid regions. Shape: (height, width).
            x_length: physical length of microstructure in flow direction.
        """
        
        if self.pressure_model is None:
            raise ValueError('Pressure model was not defined.')

        img = img.squeeze()
        assert img.dim() == 2, 'Tensor should be 2D with shape (nrows, ncols).'

        _window_size = window_size if window_size else self.window_size
        _step_size = step_size if step_size else self.step_size
        _batch_size = batch_size if batch_size else self.batch_size

        start_time = time.time()
        """1. Fragment image into squares"""
        sub_img_list, frame_positions = self.fragment_image(
            img=img,
            window_size=_window_size,
            step_size=_step_size
        )
        dx_sub = x_length / (img.shape[-1] / _window_size[-1])

        # reshape images into (1, 1, nrows, ncols)
        sub_img_list = [val.unsqueeze(0).unsqueeze(0) for val in sub_img_list]

        # batch images; each batch has the shape (batch_size, 1, nrows, ncols)
        batch_list = batch_images(
            img_list=sub_img_list,
            batch_size=_batch_size
        )

        """2. Prediction on square sub-images"""
        preds = self._predict_pressure(batch_list, dx_sub)


        """3. Pressure correction"""
        xs = torch.cat(batch_list, dim=0)

        preds = shift_pressure_fields(
            imgs = xs,
            p_fields = preds,
            frame_positions = frame_positions
        )

        """4. Overlay pressure fields"""

        result = self.overlay_fields(
            flow_fields=preds,
            frame_positions=frame_positions,
            domain_size=img.shape
        )

        dtime = time.time() - start_time
        print(f'Prediction in {dtime:.2f}s.')

        out = {
            'microstructure': img.unsqueeze(0), # shape: (1, height, width)
            
            'prediction': result['average'], # shape: (channels, height, width)
            'window_predictions': result['values'], # shape: (channels, height, width, num of windows)
            'window_positions': frame_positions # length: num of windows
        }
        return out

    @staticmethod
    @torch.no_grad()
    def compute_flow_rate(
        img: torch.Tensor,
        velocity_dict: dict[str, torch.Tensor],
        cross_section_area: torch.Tensor | float = 1.0
    ) -> dict[str, torch.Tensor]:
        """
        Compute flow rate based on velocity field.

        Args:
            img: microstructure image, 2D tensor with 0 and 1 respectively in fiber and fluid regions. Shape: (height, width).
            velocity_dict: dictionary with velocity fields. Each field is a 3D tensor with shape (channels, height, width).
            cross_section_area: cross-sectional area (perpendicular to flow direction) of microstructure.
        """

        _keys = ['target', 'prediction', 'prediction_naive']

        img = img.squeeze()
        assert img.dim() == 2, 'Tensor should be 2D with shape (nrows, ncols).'
        img = img.unsqueeze(0).unsqueeze(0) # shape: (1, 1, height, width)

        out = {}
        for ky in _keys:
            if ky in velocity_dict.keys():

                vel_field = velocity_dict[ky] # shape: (channels, height, width)
                assert vel_field.dim() == 3, 'Tensor should be 3D with shape (channels, height, width).'
                vel_field = vel_field.unsqueeze(0) # shape: (1, channels, height, width)

                flow_rate = get_flow_rate(
                    rve_tensor=img,
                    vel_tensor=vel_field,
                    cross_section_area=cross_section_area
                ) # shape: (1, width)
                flow_rate = flow_rate.squeeze(0) # shape: (width,)
                out[ky] = flow_rate

        return out


    @staticmethod
    def compute_loss(
        func: Callable,
        target: torch.Tensor,
        prediction_dict: dict[str, torch.Tensor]
    ):
        """
        Compute loss between target and prediction.
        
        Args:
            func: loss function.
            target: target tensor.
            prediction_dict: dictionary with predicted tensors.
        """
        _keys = ['prediction', 'prediction_naive']

        out = {}
        for ky in _keys:
            if ky in prediction_dict.keys():
                prediction = prediction_dict[ky]
                loss = func(target, prediction)
                out[ky] = loss
        return out


    @staticmethod
    def fragment_image(
        img: torch.Tensor,
        window_size: tuple[int,int],
        step_size: int
    ) -> tuple[list[torch.Tensor], list[tuple[int,int]]]:
        """
        Fragment image into sub-images. Some of the sub-images might overlap in space.

        `img`: image to be fragmented.\n
        `window_size`: pixel size for each sub-image.\n
        `step_size`: (pixel) step size (in the horizontal direction) after which a new sub-image is cropped.
        """

        dom_rows, dom_cols = img.shape
        win_rows, win_cols = window_size

        # number of splits in the row direction
        n_splits_row = dom_rows // win_rows

        start_col_list = define_window_positions(img.shape, window_size, step_size)

        frame_positions = []
        sub_img_list = []
        for i in range(n_splits_row):

            row_edges = torch.linspace(0, dom_rows, 1 + n_splits_row).int().tolist()

            start_row, end_row = row_edges[i], row_edges[i+1]

            for j, start_col in enumerate(start_col_list):

                end_col = start_col + win_cols
                if end_col <= dom_cols:
                    pass
                else:
                    # if 'end_col' exceeds the maximum # of pixel in this direction,
                    # then re-define 'start_col'
                    end_col = dom_cols
                    start_col = dom_cols - win_cols

                sub_img = img[start_row:end_row, start_col:end_col]
        
                frame_positions.append((start_row, start_col))
                sub_img_list.append(sub_img)

        return sub_img_list, frame_positions

    @staticmethod
    def overlay_fields(
        flow_fields: torch.Tensor,
        frame_positions: list[tuple],
        domain_size: tuple[int,int]
    ):
        """
        Overlay velocity/pressure field (square) images onto larger (rectangular) domain.
        
        Args:
            flow_fields: flow fields with shape (num_windows, ...)
            frame_positions: position of flow field images in larger domain.
        """

        # Sticth predictions using sequence in large domain
        n_channels = flow_fields.shape[1]
        out_shape = (n_channels, *domain_size)

        out = average_frames(flow_fields, frame_positions, out_shape)

        return out

    def _predict_velocity(self, batch_list: list[torch.Tensor]):
        """
        Predict velocity field for list of batched images.
        """
        pred_list = []

        for x in batch_list:
            y_pred = self.velocity_model.predict(x)
            pred_list.append(y_pred)
        out = torch.cat(pred_list, dim=0)

        return out

    def _predict_pressure(
        self,
        batch_list: list[torch.Tensor],
        x_length: torch.Tensor
    ):
        """
        Predict pressure field for list of batched images.
        """
        pred_list = []

        for x in batch_list:
            y_pred = self.pressure_model.predict(x, x_length)
            y_pred = y_pred[:, [0]]
            pred_list.append(y_pred)

        out = torch.cat(pred_list, dim=0)

        return out
    


# HELPER FUNCTIONS

def correct_velocity_field(
    img_tensor: torch.Tensor,
    vel_tensor: torch.Tensor
):
    """
    Correct velocity field based on inlet flow rate.

    Args:
        img_tensor: 4D tensor with 0 and 1 respectively in fiber and fluid regions. Shape: (samples, 1, height, width).
        vel_tensor: 4D tensor with velocity values. Shape: (samples, channels, height, width).
    """

    flow_rate = get_flow_rate(img_tensor, vel_tensor) # shape: (samples, width)

    # correction factor based on flow at inlet
    correction_factor = flow_rate / flow_rate[:, 0] 
    correction_factor = correction_factor.unsqueeze(1).unsqueeze(1) # shape: (samples, 1, 1, width)

    # adjust field
    vel_tensor_new = vel_tensor / correction_factor # (samples, channels, height, width)

    return vel_tensor_new


def define_window_positions(
    domain_size: tuple[int, int],
    window_size: tuple[int, int],
    window_step: int
):
    """
    Define positions for sliding window inside larger domain.
    """

    dom_rows, dom_cols = domain_size
    win_rows, win_cols = window_size

    # number of splits in the row direction
    n_splits_row = dom_rows // win_rows

    # splits for columns
    start_col_list = np.arange(0, dom_cols, window_step).tolist()
    # if start_col_list[-1] == ncols:
    #     start_col_list.pop()

    # get rid of starting indices that would result in images exceeding the domain
    while True:
        if start_col_list[-1] > (dom_cols - win_cols):
            start_col_list.pop()
        else:
            break
    
    if (start_col_list[-1] + win_cols) < dom_cols:
        # add window that makes up for un-covered space
        tmp = dom_cols - win_cols
        start_col_list.append(tmp)

    return start_col_list


def batch_images(
    img_list: list[torch.Tensor],
    batch_size: int
):
    """
    Create batch of images.

    Args:
        img_list: list of images, each with shape (1, 1, nrows, ncols).
        batch_size: batch size.
    """

    batch_list = []
    
    if len(img_list) <= batch_size:
        # If the number of images is inferior to the batch size
        x = torch.cat(img_list, dim=0)
        batch_list.append(x)
    
    else:
        start_id_list = torch.arange(0, len(img_list), batch_size).long().tolist()

        for start_id in start_id_list:
            end_id = start_id + batch_size
            
            if end_id < len(img_list):
                x = torch.cat(
                    img_list[start_id:end_id],
                    dim=0
                )
            else:
                # if the ending index exceeds the total number of images
                x = torch.cat(
                    img_list[start_id:],
                    dim=0
                )
            batch_list.append(x)

    return batch_list


def average_frames(
    tensor: torch.Tensor,
    frame_positions: list[tuple],
    out_shape: tuple[int,int,int]
):
    """
    Average values from overlapping frames based on their positions to get a single image.

    Args:
        tensor: tensor with shape (N, x, nrows, ncols).
        frame_positions: N-long list of indices defining order of each image in `tensor`.
        out_shape: shape of domain into which images are being averaged.
    """

    _num = len(frame_positions)
    assert tensor.dim() == 4, 'Tensor should be 4D with shape (N, x, nrows, ncols).'
    assert len(tensor) == _num, "'tensor' and 'index_start_list' should have the same length."

    in_shape = tensor.shape
    sub_rows, sub_cols = in_shape[-2:]


    accum_array = torch.zeros(*out_shape, _num, device=tensor.device)
    weight_array = torch.zeros(*out_shape, device=tensor.device)

    # Arrange arrays according to position given by index_start_list
    for (k, array) in enumerate(tensor):
        # each array has a shape of (x, nrows, ncols)

        start_row, start_col = frame_positions[k]
        end_row = start_row + sub_rows
        end_col = start_col + sub_cols

        # store sub-images in larger array based on position
        accum_array[:, start_row:end_row, start_col:end_col, k] = array

        # keep track of how many times values were added
        weight_array[:, start_row:end_row, start_col:end_col] += 1

    sum_array = torch.sum(accum_array, dim=-1)
    
    # average based on weights
    avg_array = sum_array / weight_array

    out = {
        'average': avg_array,
        'values': accum_array,
        'weights': weight_array
    }
    return out


def shift_pressure_fields(
    imgs: torch.Tensor,
    p_fields: torch.Tensor,
    frame_positions: list[tuple[int,int]]
):
    """
    Shift pressure fields based on assessed pressure at outlet.
    This function assumes that the pressure fields are all on the same row of the larger domain.

    Args:
        imgs: microstructure image, 4D tensor with shape (N, 1, nrows, ncols).
        p_fields: pressure field, 4D tensor with shape (N, 1, nrows, ncols).
        frame_positions: N-long list of indices defining order of each image in `imgs`.
    """

    imgs = imgs.clone()
    p_fields = p_fields.clone()
    num_frames = len(frame_positions)

    assert imgs.dim() == 4, "Tensor should be 4D with shape (batch, 1, nrows, ncols)."
    assert p_fields.dim() == 4, "Tensor should be 4D with shape (batch, 1, nrows, ncols)."
    assert len(p_fields) == num_frames, "'p_fields' and 'frame_positions' should have the same length."
    assert len(set([val[0] for val in frame_positions])) == 1, "All frames should be on the same row."

    h, w = p_fields.shape[-2:]
    frame_columns = [val[1] for val in frame_positions]


    """1. Get average pressure values"""

    # average pressure along length of domain; shape: (batch, ncols)
    avg_pressure = get_average_pressure(imgs.squeeze(1), p_fields.squeeze(1))



    """2. Identify overlapping frames"""

    overlapping_frames = get_overlapping_frames(
        frame_columns=frame_columns,
        window_size=(h, w)
    )


    """
    3. Shift pressure fields
    
    We move from right (outlet) to left (inlet), correcting pressure values based on average pressure
    values of overlapping frames.
    """

    # Loop through frames (from right to left)
    for k in reversed( range(num_frames) ):

        micro = imgs[k]
        array = p_fields[k]

        # column position being considered
        target_col = frame_columns[k] + w # right neighbor (!!! no +1 due to Python indexing)

        # indices of overlapping frames
        indices = overlapping_frames[k]

        if len(indices) == 0:
            # no overlapping frames, so no need to correct
            continue

        else:
            
            p_val = get_reference_pressure(
                target_col = target_col,
                indices = indices,
                frame_columns = frame_columns,
                avg_pressure = avg_pressure,
                window_size = (h, w)
            )

            # shift pressure values
            array = array + p_val * micro

            # change data in original array
            p_fields[k] = array
            avg_pressure[k] += p_val

    return p_fields


def get_overlapping_frames(
    frame_columns: list[int],
    window_size: tuple[int,int]
):
    """
    For each frame, get indices of other frames that overlap with it (on its right side).

    Args:
        frame_columns: list of indices defining starting column position of each frame.
        window_size: size (height and width) of each frame.
    """

    _, win_cols = window_size

    def overlap_for_single_frame(
        target_col: int,
        frame_columns: int,
        win_cols: int
    ) -> list[int]:
        # for a single frame (at `target_col`), get indices of other frames that overlap with it (on its right side)
        contributing_indices = []
        for k, start_col in enumerate(frame_columns):

            end_col = start_col + win_cols
            end_target_col = target_col + win_cols

            if (start_col <= end_target_col < end_col):
                contributing_indices.append(k)

        return contributing_indices

    out = [
        overlap_for_single_frame(target_col, frame_columns, win_cols)
        for target_col in frame_columns
    ]
    return out



def get_reference_pressure(
    target_col: int,
    indices: list[int],
    frame_columns: torch.Tensor | list[int],
    avg_pressure: torch.Tensor,
    window_size: tuple[int,int],
    aggregation: Literal['mean', 'first', 'last'] = 'mean'
):
    """
    For a given position (target_row, target_col), get reference pressure value
    based on average pressure values of overlapping frames.

    Args:
        target_col: target column position for which interpolation is done.
        indices: list of indices of overlapping frames.
        frame_columns: list of starting column positions of all frames.
        avg_pressure: tensor with shape (N, ncols), where N is the number of frames.
        window_size: size (height and width) of each frame.
        aggregation: method to aggregate pressure values from overlapping frames.
    """
    if isinstance(frame_columns, list):
        frame_columns = torch.tensor(frame_columns, device=avg_pressure.device)

    assert avg_pressure.dim() == 2, 'Tensor should be 2D with shape (N, ncols).'
    assert frame_columns.dim() == 1, 'Tensor should be 1D with shape (N,).'

    _, win_cols = window_size

    # pick average pressure values corresponding to overlapping frames
    avg_pressure_sub = avg_pressure[indices]

    # pick starting column positions corresponding to overlapping frames
    frame_cols_sub = frame_columns[indices]

    # get target column position
    interp_ids = target_col - frame_cols_sub # all values should be positive
    assert all(interp_ids >= 0) and all(interp_ids < win_cols), 'Some indices are out of range.'

    # gather pressure values
    pressure_vals = avg_pressure_sub.gather(1, interp_ids.long().view(-1,1))


    if aggregation == 'mean':
        # average pressure values
        out = torch.mean(pressure_vals)

    elif aggregation == 'first':
        # pick last pressure value
        out = pressure_vals[0].item()

    elif aggregation == 'last':
        # pick last pressure value
        out = pressure_vals[-1].item()

    return out