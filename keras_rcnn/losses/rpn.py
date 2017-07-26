import keras.backend as K
import keras.losses
import tensorflow as tf

import keras_rcnn.backend


class RpnLossCls(Layer):
    def call(self, inputs):
        rpn_labels, rpn_cls_score = inputs
        self.add_loss(K.mean(K.categorical_crossentropy(rpn_labels_reshape, rpn_labels_reshape)), inputs=inputs)
        return rpn_labels

class RpnLossBbox(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super().__init__(**kwargs)

    def smooth_l1(self, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=3.0):
        """
            ResultLoss = outside_weights * SmoothL1(inside_weights * (bbox_pred - bbox_targets))
            SmoothL1(x) = 0.5 * (sigma * x)^2,    if |x| < 1 / sigma^2
                          |x| - 0.5 / sigma^2,    otherwise
        """
        sigma2 = sigma**2

        inside_mul = tf.multiply(bbox_inside_weights, tf.subtract(bbox_pred, bbox_targets))

        smooth_l1_sign    = tf.cast(tf.less(tf.abs(inside_mul), 1.0 / sigma2), tf.float32)
        smooth_l1_option1 = tf.multiply(tf.multiply(inside_mul, inside_mul), 0.5 * sigma2)
        smooth_l1_option2 = tf.subtract(tf.abs(inside_mul), 0.5 / sigma2)
        smooth_l1_result  = tf.add(tf.multiply(smooth_l1_option1, smooth_l1_sign),
            tf.multiply(smooth_l1_option2, tf.abs(tf.subtract(smooth_l1_sign, 1.0))))

        outside_mul = tf.multiply(bbox_outside_weights, smooth_l1_result)

        return outside_mul

    def call(self, inputs):
        rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = inputs

        rpn_smooth_l1 = self.smooth_l1(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights)
        loss          = tf.reduce_mean(tf.reduce_sum(rpn_smooth_l1, axis=[1, 2, 3]))

        self.add_loss(loss, inputs=inputs)
        return rpn_bbox_pred
