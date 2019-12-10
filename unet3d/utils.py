import logging
import os
import shutil
import sys
import scipy.sparse as sparse
import cv2
import nibabel as nib
import numpy as np
import torch
from torch import Tensor


def simplex(t: Tensor, axis=1) -> bool:
    _sum = t.sum(axis).type(torch.float32)
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)


# def sset(a: Tensor, sub: Iterable) -> bool:
#     return uniq(a).issubset(sub)
#
#
# def one_hot(t: Tensor, axis=1) -> bool:
#     return simplex(t, axis) and sset(t, [0, 1])


def save_checkpoint(state, is_best, checkpoint_dir, logger=None):
    """Saves model and training parameters at '{checkpoint_dir}/last_checkpoint.pytorch'.
    If is_best==True saves '{checkpoint_dir}/best_checkpoint.pytorch' as well.

    Args:
        state (dict): contains model's state_dict, optimizer's state_dict, epoch
            and best evaluation metric value so far
        is_best (bool): if True state contains the best model seen so far
        checkpoint_dir (string): directory where the checkpoint are to be saved
    """

    def log_info(message):
        if logger is not None:
            logger.info(message)

    if not os.path.exists(checkpoint_dir):
        log_info(
            f"Checkpoint directory does not exists. Creating {checkpoint_dir}")
        os.mkdir(checkpoint_dir)

    last_file_path = os.path.join(checkpoint_dir, f'epoch{state["epoch"]}_checkpoint.pytorch')
    log_info(f"Saving last checkpoint to '{last_file_path}'")
    torch.save(state, last_file_path)
    if is_best:
        best_file_path = os.path.join(checkpoint_dir, 'best_checkpoint.pytorch')
        log_info(f"Saving best checkpoint to '{best_file_path}'")
        shutil.copyfile(last_file_path, best_file_path)


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """Loads model and training parameters from a given checkpoint_path
    If optimizer is provided, loads optimizer's state_dict of as well.

    Args:
        checkpoint_path (string): path to the checkpoint to be loaded
        model (torch.nn.Module): model into which the parameters are to be copied
        optimizer (torch.optim.Optimizer) optional: optimizer instance into
            which the parameters are to be copied

    Returns:
        state
    """
    if not os.path.exists(checkpoint_path):
        raise IOError(f"Checkpoint '{checkpoint_path}' does not exist")

    state = torch.load(checkpoint_path)
    try:
        model.load_state_dict(state['model_state_dict'])
    except BaseException as e:
        print('Failed to do something: ' + str(e))

    if optimizer is not None:
        try:
            optimizer.load_state_dict(state['optimizer_state_dict'])
        except Exception as e:
            print(e)

    return state


def get_logger(name, file_name='./', level=logging.INFO):
    # logging.basicConfig(filename=file_name+'model.log', level=level)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Logging to console
    stream_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        '%(asctime)s [%(threadName)s] %(levelname)s %(name)s - %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if not os.path.exists(file_name):
        os.makedirs(file_name)

    file_handler = logging.FileHandler(file_name+'model.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def get_number_of_learnable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


class RunningAverage:
    """Computes and stores the average
    """

    def __init__(self):
        self.count = 0
        self.sum = 0
        self.avg = 0

    def update(self, value, n=1):
        self.count += n
        self.sum += value * n
        self.avg = self.sum / self.count


class RunningAverageDuality:

    def __init__(self):
        self.count1 = 0
        self.sum1 = 0
        self.sum2 = 0
        self.dice_WT = 0
        self.sens_WT = 0

    def update(self, value, n=1):
        self.count1 += n
        self.sum1 += value[0] * n
        self.sum2 += value[1] * n
        self.dice_WT = self.sum1 / self.count1
        self.sens_WT = self.sum2 / self.count1


class RunningAverageMulti:

    def __init__(self):
        self.count1 = 0
        self.count2 = 0
        self.count3 = 0
        self.count4 = 0
        self.count5 = 0
        self.count6 = 0
        self.sum1 = 0
        self.sum2 = 0
        self.sum3 = 0
        self.sum4 = 0
        self.sum5 = 0
        self.sum6 = 0
        self.dice_WT = 0
        self.dice_TC = 0
        self.dice_ET = 0
        self.sens_WT = 0
        self.sens_TC = 0
        self.sens_ET = 0

    def update(self, value, n=1):
        self.count1 += n
        self.count2 += n
        self.count3 += n
        self.count4 += n
        self.count5 += n
        self.count6 += n

        self.sum1 += value[0] * n
        self.sum2 += value[1] * n
        self.sum3 += value[2] * n
        self.sum4 += value[3] * n
        self.sum5 += value[4] * n
        self.sum6 += value[5] * n

        self.dice_WT = self.sum1 / self.count1
        self.dice_TC = self.sum2 / self.count2
        self.dice_ET = self.sum3 / self.count3
        self.sens_WT = self.sum4 / self.count4
        self.sens_TC = self.sum5 / self.count5
        self.sens_ET = self.sum6 / self.count6


def find_maximum_patch_size(model, device):
    """Tries to find the biggest patch size that can be send to GPU for inference
    without throwing CUDA out of memory"""
    logger = get_logger('PatchFinder')
    in_channels = model.in_channels

    patch_shapes = [(64, 128, 128), (96, 128, 128),
                    (64, 160, 160), (96, 160, 160),
                    (64, 192, 192), (96, 192, 192)]

    for shape in patch_shapes:
        # generate random patch of a given size
        patch = np.random.randn(*shape).astype('float32')

        patch = torch \
            .from_numpy(patch) \
            .view((1, in_channels) + patch.shape) \
            .to(device)

        logger.info(f"Current patch size: {shape}")
        model(patch)


def unpad(probs, index, shape, pad_width=8):
    def _new_slices(slicing, max_size):
        if slicing.start == 0:
            p_start = 0
            i_start = 0
        else:
            p_start = pad_width
            i_start = slicing.start + pad_width

        if slicing.stop == max_size:
            p_stop = None
            i_stop = max_size
        else:
            p_stop = -pad_width
            i_stop = slicing.stop - pad_width

        return slice(p_start, p_stop), slice(i_start, i_stop)

    D, H, W = shape

    i_c, i_z, i_y, i_x = index
    p_c = slice(0, probs.shape[0])

    p_z, i_z = _new_slices(i_z, D)
    p_y, i_y = _new_slices(i_y, H)
    p_x, i_x = _new_slices(i_x, W)

    probs_index = (p_c, p_z, p_y, p_x)
    index = (i_c, i_z, i_y, i_x)
    return probs[probs_index], index


def create_feature_maps(init_channel_number, number_of_fmaps):
    return [init_channel_number * 2 ** k for k in range(number_of_fmaps)]


# Code taken from https://github.com/cremi/cremi_python
def adapted_rand(seg, gt, all_stats=False):
    """Compute Adapted Rand error as defined by the SNEMI3D contest [1]
    Formula is given as 1 - the maximal F-score of the Rand index
    (excluding the zero component of the original labels). Adapted
    from the SNEMI3D MATLAB script, hence the strange style.
    Parameters
    ----------
    seg : np.ndarray
        the segmentation to score, where each value is the label at that point
    gt : np.ndarray, same shape as seg
        the groundtruth to score against, where each value is a label
    all_stats : boolean, optional
        whether to also return precision and recall as a 3-tuple with rand_error
    Returns
    -------
    are : float
        The adapted Rand error; equal to $1 - \frac{2pr}{p + r}$,
        where $p$ and $r$ are the precision and recall described below.
    prec : float, optional
        The adapted Rand precision. (Only returned when `all_stats` is ``True``.)
    rec : float, optional
        The adapted Rand recall.  (Only returned when `all_stats` is ``True``.)
    References
    ----------
    [1]: http://brainiac2.mit.edu/SNEMI3D/evaluation
    """
    # just to prevent division by 0
    epsilon = 1e-6

    # segA is truth, segB is query
    segA = np.ravel(gt)
    segB = np.ravel(seg)
    n = segA.size

    n_labels_A = np.amax(segA) + 1
    n_labels_B = np.amax(segB) + 1

    ones_data = np.ones(n)

    p_ij = sparse.csr_matrix((ones_data, (segA[:], segB[:])), shape=(n_labels_A, n_labels_B))

    a = p_ij[1:n_labels_A, :]
    b = p_ij[1:n_labels_A, 1:n_labels_B]
    c = p_ij[1:n_labels_A, 0].todense()
    d = b.multiply(b)

    a_i = np.array(a.sum(1))
    b_i = np.array(b.sum(0))

    sumA = np.sum(a_i * a_i)
    sumB = np.sum(b_i * b_i) + (np.sum(c) / n)
    sumAB = np.sum(d) + (np.sum(c) / n)

    precision = sumAB / max(sumB, epsilon)
    recall = sumAB / max(sumA, epsilon)

    fScore = 2.0 * precision * recall / max(precision + recall, epsilon)
    are = 1.0 - fScore

    if all_stats:
        return are, precision, recall
    else:
        return are


def rotate_image(img, angle, interp=cv2.INTER_LINEAR):
    rows, cols = img.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    out = cv2.warpAffine(img, rotation_matrix, (cols, rows), flags=interp, borderMode=cv2.BORDER_REPLICATE)
    return np.reshape(out, img.shape)


def resize_image(im, size, interp=cv2.INTER_LINEAR):
    im_resized = cv2.resize(im, (size[1], size[0]), interpolation=interp)  # swap sizes to account for weird OCV API
    # add last dimension again if it was removed by resize
    if im.ndim > im_resized.ndim:
        im_resized = np.expand_dims(im_resized, im.ndim)
    return im_resized


def dense_image_warp(im, dx, dy, interp=cv2.INTER_LINEAR):

    map_x, map_y = deformation_to_transformation(dx, dy)

    do_optimization = (interp == cv2.INTER_LINEAR)
    # The following command converts the maps to compact fixed point representation
    # this leads to a ~20% increase in speed but could lead to accuracy losses
    # Can be uncommented
    if do_optimization:
        map_x, map_y = cv2.convertMaps(map_x, map_y, dstmap1type=cv2.CV_16SC2)

    remapped = cv2.remap(im, map_x, map_y, interpolation=interp, borderMode=cv2.BORDER_REFLECT) #borderValue=float(np.min(im)))
    if im.ndim > remapped.ndim:
        remapped = np.expand_dims(remapped, im.ndim)
    return remapped


def deformation_to_transformation(dx, dy):

    nx, ny = dx.shape

    grid_y, grid_x = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")

    map_x = (grid_x + dx).astype(np.float32)
    map_y = (grid_y + dy).astype(np.float32)

    return map_x, map_y


def save_nii(img_path, data, affine, header):
    '''
    Shortcut to save a nifty file
    '''

    nimg = nib.Nifti1Image(data, affine=affine, header=header)
    nimg.to_filename(img_path)