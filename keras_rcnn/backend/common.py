import keras.backend
import numpy

import keras_rcnn.backend


def bbox_transform(ex_rois, gt_rois):
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = keras.backend.log(gt_widths / ex_widths)
    targets_dh = keras.backend.log(gt_heights / ex_heights)

    targets = keras.backend.stack(
        (targets_dx, targets_dy, targets_dw, targets_dh))

    targets = keras.backend.transpose(targets)

    return keras.backend.cast(targets, 'float32')


def clip(boxes, shape):
    proposals = [
        keras.backend.maximum(
            keras.backend.minimum(boxes[:, 0::4], shape[1] - 1), 0),
        keras.backend.maximum(
            keras.backend.minimum(boxes[:, 1::4], shape[0] - 1), 0),
        keras.backend.maximum(
            keras.backend.minimum(boxes[:, 2::4], shape[1] - 1), 0),
        keras.backend.maximum(
            keras.backend.minimum(boxes[:, 3::4], shape[0] - 1), 0)
    ]

    return keras.backend.concatenate(proposals, axis=1)


def shift(shape, anchors, stride):
    shift_x = keras.backend.arange(0, shape[0]) * stride
    shift_y = keras.backend.arange(0, shape[1]) * stride

    shift_x, shift_y = keras_rcnn.backend.meshgrid(shift_x, shift_y)
    shift_x = keras.backend.reshape(shift_x, [-1])
    shift_y = keras.backend.reshape(shift_y, [-1])

    shifts = keras.backend.stack([
        shift_x,
        shift_y,
        shift_x,
        shift_y,
    ], axis=0)

    shifts = keras.backend.transpose(shifts)

    number_of_anchors = keras.backend.shape(anchors)[0]

    k = keras.backend.shape(shifts)[0]  # number of base points = feat_h * feat_w

    shifted_anchors = keras.backend.reshape(anchors, [1, number_of_anchors, 4]) + keras.backend.cast(keras.backend.reshape(shifts, [k, 1, 4]), keras.backend.floatx())

    shifted_anchors = keras.backend.reshape(shifted_anchors, [k * number_of_anchors, 4])

    return shifted_anchors


def overlap(a, b):
    """
    Parameters
    ----------
    a: (N, 4) ndarray of float
    b: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    area = (b[:, 2] - b[:, 0] + 1) * (b[:, 3] - b[:, 1] + 1)

    iw = keras.backend.minimum(keras.backend.expand_dims(a[:, 2], 1), b[:, 2]) - keras.backend.maximum(keras.backend.expand_dims(a[:, 0], 1), b[:, 0]) + 1
    ih = keras.backend.minimum(keras.backend.expand_dims(a[:, 3], 1), b[:, 3]) - keras.backend.maximum(keras.backend.expand_dims(a[:, 1], 1), b[:, 1]) + 1

    iw = keras.backend.maximum(iw, 0)
    ih = keras.backend.maximum(ih, 0)

    ua = keras.backend.expand_dims((a[:, 2] - a[:, 0] + 1) * (a[:, 3] - a[:, 1] + 1), 1) + area - iw * ih

    ua = keras.backend.maximum(ua, 0.0001)

    return iw * ih / ua


def filter_boxes(proposals, minimum):
    ws = proposals[:, 2] - proposals[:, 0] + 1
    hs = proposals[:, 3] - proposals[:, 1] + 1

    indices = keras_rcnn.backend.where((ws >= minimum) & (hs >= minimum))

    indices = keras.backend.flatten(indices)

    return keras.backend.cast(indices, "int32")


def inside_image(y_pred, img_info):
    """
    Calc indices of boxes which are located completely inside of the image
    whose size is specified by img_info ((height, width, scale)-shaped array).

    :param boxes: bounding boxes
    :param img_info:
    :return:
    """
    indices = keras_rcnn.backend.where(
        (boxes[:, 0] >= 0) &
        (boxes[:, 1] >= 0) &
        (boxes[:, 2] < img_info[1]) &  # width
        (boxes[:, 3] < img_info[0])  # height
    )

    indices = keras.backend.cast(indices, "int32")

    gathered = keras.backend.gather(boxes, indices)

    return indices[:, 0], keras.backend.reshape(gathered, [-1, 4])
