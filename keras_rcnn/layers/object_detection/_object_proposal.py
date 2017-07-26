import keras.backend
import keras.engine
import tensorflow

import keras_rcnn.backend

class ObjectProposal(keras.engine.topology.Layer):
    def __init__(
        self,
        maximum_proposals=300,
        **kwargs
    ):
        self.maximum_proposals = maximum_proposals

        super(ObjectProposal, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ObjectProposal, self).build(input_shape)

    def call(self, inputs, **kwargs):
        boxes, scores, anchors, im_info = inputs
        return self.propose(boxes, scores, anchors, im_info)

    def compute_output_shape(self, input_shape):
        return [(input_shape[0], self.maximum_proposals, 4), (input_shape[0], self.maximum_proposals)]

    def propose(self, boxes, scores, anchors, im_info):
        # 1. Generate proposals from bbox deltas and shifted anchors
        shape = keras.backend.shape(boxes)[1:3]

        # shift the anchors to original image shape
        shifted = keras_rcnn.backend.shift(shape, anchors, 16)
        shifted = keras.backend.reshape(shifted, (-1, 1, 4))

        # apply shifts to anchors
        anchors = keras.backend.reshape(anchors, (1, -1, 4))
        anchors = keras.backend.reshape(anchors + shifted, (-1, 4))

        # reshape predicted bbox to get them into the same order as the anchor
        boxes  = keras.backend.reshape(boxes, (-1, 4))
        scores = keras.backend.reshape(scores, (-1, 1))

        # convert anchors into proposals via bbox transformations
        proposals = keras_rcnn.backend.bbox_transform_inv(anchors, boxes)

        # 2. Clip predicted boxes to image
        proposals = keras_rcnn.backend.clip(proposals, im_info[:2])

        # 3. Remove predicted boxes with either height or width < threshold
        indices   = keras_rcnn.backend.filter_boxes(proposals, 16.0 * im_info[2])
        proposals = keras.backend.gather(proposals, indices)
        scores    = keras.backend.gather(scores, indices)

        # 4. sort all (proposal, score) pairs by score from highest to lowest
        # 5. take top pre_nms_topN (e.g. 6000)
        # TODO ?

        # 6. apply nms (e.g. threshold = 0.7)
        # 7. take after_nms_topN (e.g. 300) (#TODO)
        # 8. return the top proposals (-> RoIs top) (#TODO)
        indices   = keras_rcnn.backend.non_maximum_suppression(proposals, scores, self.maximum_proposals, 0.7)
        proposals = keras.backend.gather(proposals, indices)
        scores    = keras.backend.gather(scores, indices)

        # These would have to be filled with the proposal labels and target proposals
        #labels          = keras.backend.placeholder((1,))
        #targets         = keras.backend.placeholder((1,))
        #inside_weights  = keras.backend.placeholder((1,))
        #outside_weights = keras.backend.placeholder((1,))

        return [keras.backend.expand_dims(proposals, 0), scores]

