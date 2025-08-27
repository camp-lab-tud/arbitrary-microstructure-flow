import bisect

import torch
import numpy as np

from src.predictor import VelocityPredictor, PressurePredictor
from utils.data import get_average_pressure, correct_velocity_field



@torch.no_grad()
def large_domain_velocity_prediction(
    model: VelocityPredictor,
    img: torch.Tensor,
    window_size: tuple[int,int] = (256, 256),
    window_step: int = None,
    batch_size = 40
):
    """
    Predict flow field in large 2D domain.

    `model`: model to be used.\n
    `img`: image of microstructure with shape (nrows, ncols).\n
    `window_size`: pixel size for each sub-image.\n
    `window_step`: (pixel) step size (in the horizontal direction) after which a new sub-image is cropped.
    If `window_step` is equal to `window_size`[1], then there will be no overlap between the sub-images.\n
    `batch_size`: batch size when doing inference on sub-images.\n
    """
    assert img.dim() == 2, 'Tensor should be 2D with shape (nrows, ncols).'
    
    if window_step is None:
        window_step = window_size[1]

    # fragment image into sub-images
    sub_img_list, index_list = fragment_image(img, window_size, window_step)

    # reshape images into (1, 1, nrows, ncols)
    sub_img_list = [val.unsqueeze(0).unsqueeze(0) for val in sub_img_list]

    # batch images; each batch has the shape (batch_size, 1, nrows, ncols)
    batch_list = batch_images(sub_img_list, batch_size)

    # Predictions with model
    pred_list = []
    for x in batch_list:
        y_pred = model.predict(x)
        pred_list.append(y_pred)

    preds = torch.cat(pred_list, dim=0)

    # Sticth predictions using sequence in large domain
    n_channels = preds.shape[1]
    out_shape = (n_channels, *img.shape)

    result = combine_velocity_images(preds, index_list, out_shape)


    """Velocity field correction"""

    prediction_corrected = correct_velocity_field(
        img.unsqueeze(0).unsqueeze(0),
        result['average'].unsqueeze(0)
    ).squeeze(0)

    out = {
        'prediction': prediction_corrected, # shape: (channels, height, width)
        'prediction_naive': result['average'], # shape: (channels, height, width)
        'window_predictions': result['values'], # shape: (channels, height, width, num of windows)
        'window_positions': index_list, # length: num of windows
    }
    return out


@torch.no_grad()
def large_domain_pressure_prediction(
    model: PressurePredictor,
    img: torch.Tensor,
    dx: torch.Tensor,
    window_size: tuple[int,int] = (256, 256),
    window_step: int = None,
    batch_size = 40
):
    """
    Predict flow field in large 2D domain.

    `model`: model to be used.\n
    `img`: image of microstructure with shape (nrows, ncols).\n
    `dx`: length of domain.\n
    `window_size`: pixel size for each sub-image.\n
    `window_step`: (pixel) step size (in the horizontal direction) after which a new sub-image is cropped.
    If `window_step` is equal to `window_size`[1], then there will be no overlap between the sub-images.\n
    `batch_size`: batch size when doing inference on sub-images.\n
    """
    assert img.dim() == 2, 'Tensor should be 2D with shape (nrows, ncols).'

    if window_step is None:
        window_step = window_size[1]

    # fragment image into sub-images
    sub_img_list, index_list = fragment_image(img, window_size, window_step)
    dx_sub = dx / (img.shape[-1] / window_size[-1])

    # reshape images into (1, 1, nrows, ncols)
    sub_img_list = [val.unsqueeze(0).unsqueeze(0) for val in sub_img_list]

    # batch images; each batch has the shape (batch_size, 1, nrows, ncols)
    batch_list = batch_images(sub_img_list, batch_size)

    # Predictions with model
    pred_list = []
    for x in batch_list:
        y_pred = model.predict(x, dx_sub)
        y_pred = y_pred[:, [0]]
        pred_list.append(y_pred)

    xs = torch.cat(batch_list, dim=0)
    preds = torch.cat(pred_list, dim=0)

    # Sticth predictions using sequence in large domain
    n_channels = preds.shape[1]
    out_shape = (n_channels, *img.shape)

    result = combine_pressure_images(
        (xs, preds),
        index_list,
        out_shape
    )

    out = {
        'prediction': result['average'], # shape: (channels, height, width)
        'window_predictions': result['values'], # shape: (channels, height, width, num of windows)
        'window_positions': index_list # length: num of windows
    }
    return out


def fragment_image(
    img: torch.Tensor,
    window_size: tuple[int,int],
    window_step: int
):
    """
    Fragment image into sub-images. Some of the sub-images might overlap in space.

    `img`: image to be fragmented.\n
    `window_size`: pixel size for each sub-image.\n
    `window_step`: (pixel) step size (in the horizontal direction) after which a new sub-image is cropped.
    """

    dom_rows, dom_cols = img.shape
    win_rows, win_cols = window_size

    # number of splits in the row direction
    n_splits_row = dom_rows // win_rows

    start_col_list = define_window_positions(img.shape, window_size, window_step)

    index_start_list: list[tuple] = []
    sub_img_list: list[torch.Tensor] = []
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
    
            index_start_list.append((start_row, start_col))
            sub_img_list.append(sub_img)

    return sub_img_list, index_start_list




def combine_velocity_images(
    tensor: torch.Tensor,
    index_start_list: list[tuple],
    out_shape: tuple[int,int,int]
):
    """
    Combine velocity images.

    `tensor`: tensor with shape (N, x, nrows, ncols).\n
    `index_list`: N-long list of indices defining order of each image in `tensor`.\n
    `out_shape`: shape of domain into which images are being averaged.
    """

    _num = len(index_start_list)
    assert tensor.dim() == 4, 'Tensor should be 4D with shape (N, x, nrows, ncols).'
    assert len(tensor) == _num, "'tensor' and 'index_start_list' should have the same length."

    in_shape = tensor.shape
    sub_rows, sub_cols = in_shape[-2:]


    accum_array = torch.zeros(*out_shape, _num, device=tensor.device)
    weight_array = torch.zeros(*out_shape, device=tensor.device)

    # Arrange arrays according to position given by index_start_list
    for (k, array) in enumerate(tensor):
        # each array has a shape of (x, nrows, ncols)

        start_row, start_col = index_start_list[k]
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


def combine_pressure_images(
    topo_field_pair: tuple[torch.Tensor, torch.Tensor],
    index_start_list: list[tuple],
    out_shape: tuple[int,int,int]
):
    """
    Combine pressure images.

    `topo_field_pair`: 2-tuple in which the 1st element is a
    microstructure topology (captured by a sliding window)
    and the 2nd element is the corresponding pressure field.
    Each have a shape of (N, 1, nrows, ncols).\n
    `index_list`: N-long list of indices defining order of each image in `tensor`.\n
    `out_shape`: shape of domain into which images are being averaged.
    """

    num_frames = len(index_start_list)

    xs, tensor = topo_field_pair

    assert xs.dim() == 4, 'Tensor should be 4D with shape (batch, 1, nrows, ncols).'
    assert tensor.dim() == 4, 'Tensor should be 4D with shape (batch, 1, nrows, ncols).'
    assert len(tensor) == num_frames, "'tensor' and 'index_start_list' should have the same length."

    # ending rows & columns for each frame
    sub_rows, sub_cols = tensor.shape[-2:]
    index_end_list = [
        (row_id+sub_rows, col_id+sub_cols) for (row_id, col_id) in index_start_list
    ]

    accum_array = torch.zeros(*out_shape, num_frames, device=tensor.device)
    weight_array = torch.zeros(*out_shape, device=tensor.device)


    """Flip so that we move from outlet to inlet"""
    xs = torch.flipud(xs)
    tensor = torch.flipud(tensor)

    index_start_list = index_start_list[::-1]
    index_end_list = index_end_list[::-1]


    """Get average pressure values"""

    # average pressure along length of domain; shape: (batch, ncols)
    avg_pressure = get_average_pressure(xs.squeeze(1), tensor.squeeze(1))


    # Loop through frames
    for k in range(num_frames):
        # each array has a shape of (x, nrows, ncols)

        micro = xs[k]
        array = tensor[k]

        start_row, start_col = index_start_list[k]
        end_row, end_col = index_end_list[k]

        right_nbr = (start_row, end_col) # right neighbor (!!! no +1 due to Python indexing)

        right_nbr_start_col = right_nbr[1]
        if right_nbr_start_col >= index_end_list[0][1]:
            # hypothetical neighbor is beyond outlet,
            # so no need to correct
            pass

        else:
            target_frame_idx = bisect.bisect_left(
                index_start_list,
                -1 * right_nbr_start_col, # multiply by -1 because of descending order
                key = lambda x: -1 * x[1] # multiply by -1 because of descending order
            )

            frame_avg_pressure = avg_pressure[target_frame_idx]

            # pick average pressure corresponding to location
            frame_start_col = index_start_list[target_frame_idx][1]
            interp_idx = right_nbr_start_col - frame_start_col

            pressure_incr = frame_avg_pressure[interp_idx]

            # shift pressure values
            array = array + pressure_incr * micro

            # Also, shift data in original array
            tensor[k] = array
            avg_pressure[k] += pressure_incr


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


def batch_images(
    img_list: list[torch.Tensor],
    batch_size: int
):
    """
    Create batch of images.

    `img_list`: list of images, each with shape (1, 1, nrows, ncols).\n
    `batch_size`: batch size.
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