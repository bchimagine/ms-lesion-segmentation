# 3D Patch Prediction Fusion + Soft Voting

import scipy.signal
from tqdm import tqdm
import gc
import numpy as np

# `gc.collect()` clears RAM between operations.
# It should run faster if they are removed, if enough memory is available.

def _spline_window(window_size, power=2):
    """
    Squared spline (power=2) window function
    """
    intersection = int(window_size/4)
    wind_outer = (abs(2*(scipy.signal.triang(window_size))) ** power)/2
    wind_outer[intersection:-intersection] = 0

    wind_inner = 1 - (abs(2*(scipy.signal.triang(window_size) - 1)) ** power)/2
    wind_inner[:intersection] = 0
    wind_inner[-intersection:] = 0

    wind = wind_inner + wind_outer
    wind = wind / np.average(wind)
    return wind

cached_2d_windows = dict()
def _window_2D(window_size, power=2):
    """
    Make a 1D window function, then infer and return a 2D window function.
    Done with an augmentation, and self multiplication with its transpose.
    Could be generalized to more dimensions.
    """
    # Memorization
    global cached_2d_windows
    key = "{}_{}".format(window_size, power)
    if key in cached_2d_windows:
        wind = cached_2d_windows[key]
    else:
        wind = _spline_window(window_size, power)
        print(np.shape(wind))
        wind = np.expand_dims(np.expand_dims(wind, 3), 3)
        print(np.shape(wind))
        wind = wind * wind.transpose(1, 0, 2)
        cached_2d_windows[key] = wind
    return wind

cached_3d_windows = dict()
def _window_3D(window_size, power=2):
    """
    Make a 2D window function, then infer and return a 3D window function.
    Done with an augmentation, and self multiplication with its transpose.
    Could be generalized to more dimensions.
    """
    # Memorization
    global cached_3d_windows
    key = "{}_{}".format(window_size, power)
    if key in cached_3d_windows:
        wind = cached_3d_windows[key]
    else:
        wind = _window_2D(window_size, power)
        print(np.shape(wind))
        wind = np.expand_dims(wind, 3)
        print(np.shape(wind))
        wind = wind * np.expand_dims(np.expand_dims(np.expand_dims(_spline_window(window_size, power), 3), 3), 3) \
                    .transpose(2, 1, 0, 3)

        cached_3d_windows[key] = wind
    print(np.shape(wind))
    return wind

def _pad_img(img, window_size, subdivisions):
    """
    Add borders to img for a "valid" border pattern according to "window_size" and
    "subdivisions".
    Image is an np array of shape (x, y, z, nb_channels).
    """
    aug = int(round(window_size * (1 - 1.0/subdivisions)))
    more_borders = ((aug, aug), (aug, aug), (aug, aug), (0, 0))
    ret = np.pad(img, pad_width=more_borders, mode='reflect')
    gc.collect()

    return ret

def _unpad_img(padded_img, window_size, subdivisions):
    """
    Undo what's done in the `_pad_img` function.
    Image is an np array of shape (x, y, z, nb_channels).
    """
    aug = int(round(window_size * (1 - 1.0/subdivisions)))
    ret = padded_img[
        aug:-aug,
        aug:-aug,
        aug:-aug,
        :
    ]
    gc.collect()
    return ret

def _rotate_mirror_do(im):
    """
    Duplicate an np array (image) of shape (x, y, z, nb_channels) 4 times, in order
    to have all the possible rotations of that image that fits the
    possible 180 degrees rotations.
    """
    mirrs = []
    mirrs.append(np.array(im))
    mirrs.append(np.rot90(np.array(im), axes=(0, 1), k=2))
    mirrs.append(np.rot90(np.array(im), axes=(0, 2), k=2))
    mirrs.append(np.rot90(np.array(im), axes=(1, 2), k=2))
    return mirrs

def _rotate_mirror_undo(im_mirrs):
    """
    merges a list of 8 np arrays (images) of shape (x, y, z, nb_channels) generated
    from the `_rotate_mirror_do` function. Each images might have changed and
    merging them implies to rotated them back in order and average things out.
    """
    origs = []
    origs.append(np.array(im_mirrs[0]))
    origs.append(np.rot90(np.array(im_mirrs[1]), axes=(0, 1), k=2))
    origs.append(np.rot90(np.array(im_mirrs[2]), axes=(0, 2), k=2))
    origs.append(np.rot90(np.array(im_mirrs[3]), axes=(1, 2), k=2))
    return np.mean(origs, axis=0)

def _windowed_subdivs(padded_img, window_size, subdivisions, nb_classes, pred_func):
    """
    Create tiled overlapping patches.
    Returns:
        7D numpy array of shape = (
            nb_patches_along_X,
            nb_patches_along_Y,
            nb_patches_along_Z,
            patches_resolution_along_X,
            patches_resolution_along_Y,
            patches_resolution_along_Z,
            nb_output_channels
        )
    Note:
        patches_resolution_along_X == patches_resolution_along_Y == patches_resolution_along_Z == window_size
    """
    WINDOW_SPLINE_3D = _window_3D(window_size=window_size, power=2)

    step = int(window_size/subdivisions)
    padx_len = padded_img.shape[0]
    pady_len = padded_img.shape[1]
    padz_len = padded_img.shape[2]
    subdivs = []
    
    print(padx_len, pady_len, padz_len)

    for i in range(0, padx_len-window_size+1, step):
        subdivs.append([])
        for j in range(0, pady_len-window_size+1, step):
            subdivs[-1].append([])
            for k in range(0, padz_len-window_size+1, step):
                patch = padded_img[i:i+window_size, j:j+window_size, k:k+window_size, :]
                subdivs[-1][-1].append(patch)
                
    print(np.shape(subdivs))
    
    subdivs2 = np.zeros((np.shape(subdivs)[0], np.shape(subdivs)[1], np.shape(subdivs)[2], 
                         window_size, window_size, window_size, nb_classes))
    
    print(np.shape(subdivs2))

    gc.collect()
    subdivs = np.array(subdivs)
    gc.collect()
    a, b, c, d, e, f, g = subdivs.shape
    gc.collect()

    for i in range(0, np.shape(subdivs)[0]):
        for j in range(0, np.shape(subdivs)[1]):
            for k in range(0, np.shape(subdivs)[2]):
                subdivs2[i,j,k] = np.reshape(pred_func(subdivs[i,j,k:k+1]), 
                                                (window_size, window_size, window_size, nb_classes))
    
    gc.collect()
    subdivs2 = np.array([patch * WINDOW_SPLINE_3D for patch in subdivs2])
    gc.collect()

    subdivs2 = subdivs2.reshape(a, b, c, d, e, f, nb_classes)
    gc.collect()

    return subdivs2

def _recreate_from_subdivs(subdivs, window_size, subdivisions, padded_out_shape):
    """
    Merge tiled overlapping patches smoothly.
    """
    step = int(window_size/subdivisions)
    padx_len = padded_out_shape[0]
    pady_len = padded_out_shape[1]
    padz_len = padded_out_shape[2]

    y = np.zeros(padded_out_shape)

    a = 0
    for i in range(0, padx_len-window_size+1, step):
        b = 0
        for j in range(0, pady_len-window_size+1, step):
            c = 0
            for k in range(0, padz_len-window_size+1, step):
                windowed_patch = subdivs[a, b, c]
                y[i:i+window_size, j:j+window_size, k:k+window_size] = y[i:i+window_size, j:j+window_size, k:k+window_size] + windowed_patch
                c += 1
            b += 1
        a += 1
    return y / (subdivisions ** 2)

def predict_img_with_smooth_windowing(input_img, window_size, subdivisions, nb_classes, pred_func):
    """
    Apply the `pred_func` function to square patches of the image, and overlap
    the predictions to merge them smoothly.
    See 6th, 7th and 8th idea here:
    http://blog.kaggle.com/2017/05/09/dstl-satellite-imagery-competition-3rd-place-winners-interview-vladimir-sergey/
    """
    pad = _pad_img(input_img, window_size, subdivisions)
    pads = _rotate_mirror_do(pad)

    res = []
    for pad in tqdm(pads):
        # For every rotation:
        sd = _windowed_subdivs(pad, window_size, subdivisions, nb_classes, pred_func)
        one_padded_result = _recreate_from_subdivs(
            sd, window_size, subdivisions,
            padded_out_shape=list(pad.shape[:-1])+[nb_classes])

        res.append(one_padded_result)

    # Merge after rotations:
    padded_results = _rotate_mirror_undo(res)

    prd = _unpad_img(padded_results, window_size, subdivisions)

    prd = prd[:input_img.shape[0], :input_img.shape[1], :]

    return prd
