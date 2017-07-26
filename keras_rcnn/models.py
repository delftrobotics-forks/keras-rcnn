# -*- coding: utf-8 -*-

import keras
import keras_resnet.models

import keras_rcnn.layers
import keras_rcnn.classifiers
import keras_rcnn.losses

from keras_rcnn.layers.object_detection.generate_anchors import generate_anchors

import numpy as np


class RCNN(keras.models.Model):
    """
    Faster R-CNN model by S Ren et, al. (2015).

    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)
    :param heads: R-CNN classifiers for object detection and/or segmentation on the proposed regions
    :param rois: integer, number of regions of interest per image
    :param anchor_ratios: list of anchor ratios to generate
    :param anchor_scales: list of anchor scales to generate

    :return model: a functional model API for R-CNN.
    """

    def __init__(self, inputs, features, heads, rois, anchor_ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32]):
        data, im_info, gt_boxes = inputs

        num_anchors   = len(anchor_ratios) * len(anchor_scales)
        anchor_ratios = np.array(anchor_ratios, dtype=keras.backend.floatx())
        anchor_scales = np.array(anchor_scales, dtype=keras.backend.floatx())

        #TODO Figure out a way around using a lambda layer here
        anchors       = keras.layers.Lambda(lambda x: keras.backend.constant(generate_anchors(ratios=anchor_ratios, scales=anchor_scales), dtype=keras.backend.floatx()))(data)

        # Append RPN
        rpn_conv      = keras.layers.Conv2D(512            , kernel_size=(3, 3), padding="same", activation="relu", name="rpn_conv")(features)
        rpn_cls_prob  = keras.layers.Conv2D(num_anchors * 2, kernel_size=(1, 1), activation="sigmoid", name="rpn_cls_prob")(rpn_conv)
        rpn_bbox_pred = keras.layers.Conv2D(num_anchors * 4, kernel_size=(1, 1), name="rpn_bbox_pred")(rpn_conv)

        # Generate object proposals
        proposals, scores = keras_rcnn.layers.object_detection.ObjectProposal(
            maximum_proposals=rois,
            name="rpn_proposals",
        )([rpn_bbox_pred, rpn_cls_prob, anchors, im_info])

        # Add RPN losses
        #rpn_cls_loss  = keras_rcnn.losses.RpnLossCls()([rpn_labels, rpn_cls_prob])
        #rpn_bbox_loss = keras_rcnn.losses.RpnLossBbox()([proposals, targets, inside_weights, outside_weights])

        # Apply the classifiers on the proposed regions
        slices = keras_rcnn.layers.ROI((7, 7))([data, proposals])

        [score, boxes] = heads(slices)

        super(RCNN, self).__init__(inputs, [score, boxes])


class ResNet50RCNN(RCNN):
    """
    Faster R-CNN model with ResNet50.

    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)
    :param classes: integer, number of classes
    :param rois: integer, number of regions of interest per image
    :param blocks: list of blocks to use in ResNet50
    :param anchor_ratios: list of anchor ratios to generate
    :param anchor_scales: list of anchor scales to generate

    :return model: a functional model API for R-CNN.
    """

    def __init__(self, inputs, classes, rois=300, blocks=[3, 4, 6], anchor_ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32]):
        # ResNet50 as encoder
        data, _, _ = inputs
        features = keras_resnet.models.ResNet50(data, blocks=blocks, include_top=False).output

        # ResHead with score and boxes
        heads = keras_rcnn.classifiers.residual(classes)

        super(ResNet50RCNN, self).__init__(inputs, features, heads, rois, anchor_ratios=anchor_ratios, anchor_scales=anchor_scales)
