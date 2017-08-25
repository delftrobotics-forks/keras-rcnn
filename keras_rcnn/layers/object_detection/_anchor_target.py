import keras.backend
import keras.engine
import tensorflow

import keras_rcnn.backend
import keras_rcnn.layers

# TODO: Move the parameters to a config file.
RPN_FG_FRACTION         = 0.5
RPN_BATCHSIZE           = 256
RPN_BBOX_INSIDE_WEIGHTS = [1.0, 1.0, 1.0, 1.0]
RPN_POSITIVE_WEIGHT     = -1.0

class AnchorTarget(keras.layers.Layer):
    """
    Calculates anchor classification labels (1: positive, 0: negative, -1: ignore) and bounding-box regression targets.

    Args:
        allowed_border:    Allows boxes to be outside the image by a maximum number of pixels.
        clobber_positives: if an anchor statisfied by positive and negative conditions given to negative label
        negative_overlap:  IoU threshold below which labels should be given negative label.
        positive_overlap:  IoU threshold above which labels should be given positive label.

    Input shape:
        (# of batches, width of feature map, height of feature map, 2 * # of anchors), (# of samples, 4), (3)

    Output shape:
        (# of samples, ), (# of samples, 4)
    """

    def __init__(self, allowed_border=0, clobber_positives=False,
                 negative_overlap=0.3, positive_overlap=0.7, stride=16, **kwargs):

        self.allowed_border    = allowed_border
        self.clobber_positives = clobber_positives
        self.negative_overlap  = negative_overlap
        self.positive_overlap  = positive_overlap
        self.stride            = stride

        super(AnchorTarget, self).__init__(**kwargs)

    def build(self, input_shape):
        super(AnchorTarget, self).build(input_shape)

    def call(self, inputs, **kwargs):
        scores, gt_boxes, metadata = inputs

        metadata = metadata[0,:] # keras.backend.int_shape(image)[1:]
        gt_boxes = gt_boxes[0]

        # TODO: Fix usage of batch index.
        rows, cols, total_anchors = keras.backend.int_shape(scores)[1:]
        total_anchors             = rows * cols * total_anchors // 2

        # Generate proposals from bounding box deltas and shifted anchors.
        anchors = keras_rcnn.backend.shift((rows, cols), self.stride)

        # Only keep the anchors that are inside the image.
        inds_inside, anchors = inside_image(anchors, metadata, self.allowed_border)

        # Compute indices of the most likely (highest IoU overlap) ground-truth boxes and labels.
        argmax_overlap_inds, labels = compute_labels(gt_boxes, anchors, inds_inside,
                                                     self.negative_overlap,
                                                     self.positive_overlap,
                                                     self.clobber_positives)

        # Select the ground-truth boxes for the generated anchors.
        gt_boxes = keras.backend.gather(gt_boxes, argmax_overlap_inds)

        # Convert fixed anchors in (x, y, w, h) to (dx, dy, dw, dh).
        bbox_targets = keras_rcnn.backend.bbox_transform(anchors, gt_boxes)
        bbox_targets = keras.backend.reshape(bbox_targets, (-1, 4))

        # Compute inside and outside bounding-box weights.
        bbox_inside_weights, bbox_outside_weights = compute_bbox_weights(anchors, labels,
                                                                         RPN_POSITIVE_WEIGHT,
                                                                         RPN_BBOX_INSIDE_WEIGHTS)

        # Map back to the original set of anchors.
        labels               = unmap(labels,               total_anchors, inds_inside, fill=-1)
        bbox_targets         = unmap(bbox_targets,         total_anchors, inds_inside, fill=0)
        bbox_inside_weights  = unmap(bbox_inside_weights,  total_anchors, inds_inside, fill=0)
        bbox_outside_weights = unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)

        labels           = keras.backend.expand_dims(labels, axis=0)
        bbox_targets = keras.backend.expand_dims(bbox_targets, axis=0)

        # Reshape tensors as in the original code.
        labels = keras.backend.reshape(labels, (1, rows, cols, total_anchors // 2))
        labels = keras_rcnn.backend.transpose(labels, [0, 3, 1, 2])
        labels = keras.backend.reshape(labels, (1, 1, total_anchors // 2 * rows, cols))

        # There are A = (total_anchors // 2) anchors at each location in the feature map.
        bbox_targets = keras.backend.reshape(bbox_targets, (1, rows, cols, total_anchors * 2))
        bbox_targets = keras_rcnn.backend.transpose(bbox_targets, [0, 3, 1, 2])

        bbox_inside_weights = keras.backend.reshape(bbox_inside_weights, (1, rows, cols, total_anchors * 2))
        bbox_inside_weights = keras_rcnn.backend.transpose(bbox_inside_weights, [0, 3, 1, 2])

        bbox_outside_weights = keras.backend.reshape(bbox_outside_weights, (1, rows, cols, total_anchors * 2))
        bbox_outside_weights = keras_rcnn.backend.transpose(bbox_outside_weights, [0, 3, 1, 2])

        return [labels, bbox_targets, bbox_inside_weights, bbox_outside_weights]

    def compute_output_shape(self, input_shape):
        return [(1, None), (1, None, 4)]

    def compute_mask(self, inputs, mask=None):
        # Unfortunately, this is required by Keras.
        return 2 * [None]

def subsample_labels(labels):
    """
    Subsample labels by setting some to -1.

    Args:
        labels: Array of labels (1: positive, 0: negative, -1: ignore).

    Returns:
        labels: Subsampled labels.
    """

    # Subsample positive labels if we have too many.
    labels = subsample_positive_labels(labels)

    # Subsample negative labels if we have too many.
    labels = subsample_negative_labels(labels)

    return labels


def compute_labels(gt_boxes, anchors, inds_inside, RPN_NEGATIVE_OVERLAP=0.3,
                   RPN_POSITIVE_OVERLAP=0.7, clobber_positives=False):
    """
    Creates anchor classification labels (1: positive, 0: negative, -1: ignore).

    Args:
        gt_boxes:    Ground-truth bounding boxes.
        inds_inside: Indices of the anchors that are inside image.
        anchors:     Generated anchors.

    Returns:
        anchor_overlap_inds: Index of the most likely ground-truth box, for each generated anchor.
        labels:              Subsampled labels.
    """

    ones   = keras.backend.ones_like(inds_inside, dtype=keras.backend.floatx())
    labels = ones * -1
    zeros  = keras.backend.zeros_like(inds_inside, dtype=keras.backend.floatx())

    # Compute overlaps between anchors and ground-truth bounding boxes.
    anchor_overlap_inds, max_overlaps, gt_overlap_inds = compute_overlaps(anchors, gt_boxes, inds_inside)

    # Assign background labels first so that positive labels can clobber them.
    if not clobber_positives:
        labels = keras_rcnn.backend.where(keras.backend.less(max_overlaps, RPN_NEGATIVE_OVERLAP), zeros, labels)

    # TODO: generalize unique beyond 1D
    unique_gto_inds, _ = keras_rcnn.backend.unique(gt_overlap_inds, return_index=True)
    inverse_labels     = keras.backend.gather(-1 * labels, unique_gto_inds)
    unique_gto_inds    = keras.backend.expand_dims(unique_gto_inds, 1)

    # Assign foreground labels to each anchor for which a corresponding ground-truth box was found.
    updates = keras.backend.ones_like(keras.backend.reshape(unique_gto_inds, (-1,)), dtype=keras.backend.floatx())
    labels  = keras_rcnn.backend.scatter_add_tensor(labels, unique_gto_inds, inverse_labels + updates)

    # Assign foreground labels based on IoU overlaps that are higher than RPN_POSITIVE_OVERLAP.
    labels = keras_rcnn.backend.where(keras.backend.greater_equal(max_overlaps, RPN_POSITIVE_OVERLAP), ones, labels)

    # Assign background labels last so that negative labels can clobber positives.
    if clobber_positives:
        labels = keras_rcnn.backend.where(keras.backend.less(max_overlaps, RPN_NEGATIVE_OVERLAP), zeros, labels)

    return anchor_overlap_inds, subsample_labels(labels)


def compute_bbox_weights(anchors, labels, positive_weight, inside_weights):
    """
    Creates the inside and outside bounding-box weights.

    Args:
        anchors: Generated anchors.
        labels:  Labels obtained after subsampling.

    Returns:
        bbox_inside_weights:  Inside bounding-box weights.
        bbox_outside_weights: Outside bounding-box weights.
    """

    bbox_inside_weights  = keras.backend.zeros_like(anchors)
    bbox_outside_weights = keras.backend.zeros_like(anchors)

    num_anchors             = keras.backend.int_shape(anchors)[0]
    rpn_bbox_inside_weights = keras.backend.constant([inside_weights])
    rpn_bbox_inside_weights = keras.backend.tile(rpn_bbox_inside_weights, (num_anchors, 1))

    condition            = keras.backend.equal(labels, 1)
    bbox_inside_weights  = keras_rcnn.backend.where(condition, rpn_bbox_inside_weights, bbox_inside_weights)

    if positive_weight < 0:
        # Assign equal weights to both positive and negative labels.
        condition        = keras.backend.greater_equal(labels, 0)
        num_examples     = keras.backend.sum(keras.backend.cast(condition, keras.backend.floatx()))
        positive_weights = keras.backend.ones_like(anchors) / num_examples
        negative_weights = keras.backend.ones_like(anchors) / num_examples
    else:
        assert (positive_weight > 0) & (positive_weight < 1)
        # Assign weights that favor either the positive or the negative labels.
        condition        = keras.backend.equal(labels, 1)
        num_examples     = keras.backend.sum(keras.backend.cast(condition, keras.backend.floatx()))
        positive_weights = keras.backend.ones_like(anchors) * positive_weight / num_examples

        condition        = keras.backend.equal(labels, 0)
        num_examples     = keras.backend.sum(keras.backend.cast(condition, keras.backend.floatx()))
        negative_weights = keras.backend.ones_like(anchors) * (1 - positive_weight) / num_examples

    condition            = keras.backend.equal(labels, 1)
    bbox_outside_weights = keras_rcnn.backend.where(condition, positive_weights, bbox_outside_weights)
    condition            = keras.backend.equal(labels, 0)
    bbox_outside_weights = keras_rcnn.backend.where(condition, negative_weights, bbox_outside_weights)

    return bbox_inside_weights, bbox_outside_weights

def compute_overlaps(anchors, gt_boxes, inds_inside):
    """
    Computes overlaps between the generated anchors and the ground-truth boxes.

    Args:
        anchors:     Remaining anchors after removing the ones outside of the image.
        gt_boxes:    Ground-truth bounding boxes.
        inds_inside: Indices used to select the anchors that are inside the image.

    Returns:
        anchor_overlap_inds: Index of the most likely ground-truth box, for each generated anchor.
        max_overlaps:        IoU overlap of each anchor with its most likely ground-truth box.
        gt_overlap_inds:     Index of the most likely anchor, for each ground-truth box.
    """

    assert keras.backend.ndim(anchors)  == 2
    assert keras.backend.ndim(gt_boxes) == 2

    # Compute a matrix with the IoU overlaps between each anchor and each ground-truth box.
    reference = keras_rcnn.backend.overlap(anchors, gt_boxes)

    # Compute the index of the most likely anchor, for each ground-truth box.
    gt_overlap_inds = keras.backend.argmax(reference, axis=0)

    # Compute the index of the most likely ground-truth box, for each anchor.
    anchor_overlap_inds = keras.backend.argmax(reference, axis=1)

    # Create a matrix with the IoU overlap betwween each anchor and its most likely ground-truth box.
    arranged     = keras.backend.arange(0, keras.backend.shape(inds_inside)[0])
    indices      = keras.backend.stack([arranged, keras.backend.cast(anchor_overlap_inds, "int32")], axis=0)
    indices      = keras.backend.transpose(indices)
    max_overlaps = keras_rcnn.backend.gather_nd(reference, indices)

    return anchor_overlap_inds, max_overlaps, gt_overlap_inds

def subsample_negative_labels(labels):
    """
    Subsample negative labels if there are too many.

    Args:
        labels: Array of labels (1: positive, 0: negative, -1: ignore).

    Returns:
        labels: Subsampled negative labels.
    """

    # Determine how many extra negative labels we have in our array.
    num_bg      = RPN_BATCHSIZE - keras.backend.shape(keras_rcnn.backend.where(keras.backend.equal(labels, 1)))[0]
    bg_inds     = keras_rcnn.backend.where(keras.backend.equal(labels, 0))
    num_bg_inds = keras.backend.shape(bg_inds)[0]
    size        = num_bg_inds - num_bg

    # Lambda function used to select exactly `size` labels.
    def select_negative_labels():
        indices = keras_rcnn.backend.shuffle(keras.backend.reshape(bg_inds, (-1,)))[:size]
        updates = tensorflow.ones((size,)) * -1
        indices = keras.backend.reshape(indices, (-1, 1))

        return keras_rcnn.backend.scatter_add_tensor(labels, indices, updates)

    # Return `labels` if there are not enough labels to sample or the subsampled array otherwise.
    return keras.backend.switch(keras.backend.less_equal(size, 0), labels, lambda: select_negative_labels())


def subsample_positive_labels(labels):
    """
    Subsample positive labels if there are too many.

    Args:
        labels: Array of labels (1: positive, 0: negative, -1: ignore).

    Returns:
        labels: Subsampled positive labels.
    """

    # Determine how many extra positive labels we have in our array.
    num_fg      = int(RPN_FG_FRACTION * RPN_BATCHSIZE)
    fg_inds     = keras_rcnn.backend.where(keras.backend.equal(labels, 1))
    num_fg_inds = keras.backend.shape(fg_inds)[0]
    size        = num_fg_inds - num_fg

    # Lambda function used to select exactly `size` labels.
    def select_positive_labels():
        indices = keras_rcnn.backend.shuffle(keras.backend.reshape(fg_inds, (-1,)))[:size]
        updates = tensorflow.ones((size,)) * -2
        indices = keras.backend.reshape(indices, (-1, 1))

        return keras_rcnn.backend.scatter_add_tensor(labels, indices, updates)

    # Return `labels` if there are not enough labels to sample or the subsampled array otherwise.
    return keras.backend.switch(keras.backend.less_equal(size, 0), labels, lambda: select_positive_labels())


def unmap(data, count, inds_inside, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of size count). """

    if keras.backend.ndim(data) == 1:
        result  = keras.backend.ones((count,), dtype=keras.backend.floatx()) * fill
        inds_nd = keras.backend.expand_dims(inds_inside)
    else:
        result = keras.backend.ones((count,) + keras.backend.int_shape(data)[1:], dtype=keras.backend.floatx()) * fill
        data   = keras.backend.transpose(data)
        data   = keras.backend.reshape(data, (-1,))

        inds_ii = keras.backend.tile(inds_inside, [4])
        inds_ii = keras.backend.expand_dims(inds_ii)

        ones        = keras.backend.expand_dims(keras.backend.ones_like(inds_inside), 1)
        inds_coords = keras.backend.concatenate([ones * 0, ones, ones * 2, ones * 3], 0)
        inds_nd     = keras.backend.concatenate([inds_ii, inds_coords], 1)

    inverse_result = keras_rcnn.backend.squeeze(keras_rcnn.backend.gather_nd(-1 * result, inds_nd))

    return keras_rcnn.backend.scatter_add_tensor(result, inds_nd, inverse_result + data)


def inside_image(boxes, image_shape, allowed_border=0):
    """
    Calculates indices of boxes that are located inside an image shape, with some allowed
    tolerance.

    Args:
        boxes:          Tensor (None, 4) containing boxes in original image (x1, y1, x2, y2).
        image_shape:    Height, width and scale of the image the boxes are checked against.
        allowed_border: Number of pixels by which boxes are allowed to be outside the image.

    Returns:
        Indices (None, 4) of boxes inside the image.
        Tensor (None, 4) of boxes inside image.
    """

    indices = keras_rcnn.backend.where((boxes[:, 0] >= -allowed_border) &
                                       (boxes[:, 1] >= -allowed_border) &
                                       (boxes[:, 2] <   allowed_border + image_shape[1]) & # Width
                                       (boxes[:, 3] <   allowed_border + image_shape[0]))  # Height

    indices  = keras.backend.cast(indices, "int32")
    gathered = keras.backend.gather(boxes, indices)

    return indices[:, 0], keras.backend.reshape(gathered, [-1, 4])
