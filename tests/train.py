import keras.backend
import keras.engine
import keras.layers
import numpy
import skimage.io
import tensorflow

import keras_resnet.blocks
import keras_resnet.models

import keras_rcnn.backend
import keras_rcnn.datasets.malaria
import keras_rcnn.losses
import keras_rcnn.layers.object_detection
import keras_rcnn.preprocessing

import sklearn.preprocessing
import skimage.transform

def create_feature_maps(image, options):
    y = keras.layers.Conv2D(64, **options)(image)
    y = keras.layers.Conv2D(64, **options)(y)

    y = keras.layers.MaxPooling2D(strides=(2, 2))(y)

    y = keras.layers.Conv2D(128, **options)(y)
    y = keras.layers.Conv2D(128, **options)(y)

    y = keras.layers.MaxPooling2D(strides=(2, 2))(y)

    y = keras.layers.Conv2D(256, **options)(y)
    y = keras.layers.Conv2D(256, **options)(y)
    y = keras.layers.Conv2D(256, **options)(y)
    y = keras.layers.Conv2D(256, **options)(y)

    y = keras.layers.MaxPooling2D(strides=(2, 2))(y)

    y = keras.layers.Conv2D(512, **options)(y)
    y = keras.layers.Conv2D(512, **options)(y)
    y = keras.layers.Conv2D(512, **options)(y)
    y = keras.layers.Conv2D(512, **options)(y)

    y = keras.layers.MaxPooling2D(strides=(2, 2))(y)

    y = keras.layers.Conv2D(512, **options)(y)
    y = keras.layers.Conv2D(512, **options)(y)
    y = keras.layers.Conv2D(512, **options)(y)

    y = keras.layers.Conv2D(512, **options)(y)

    return y

def load_dataset():
    train_data, test_data = keras_rcnn.datasets.malaria.load_data()

    image_shape   = None
    classes       = {}
    classes_count = 0

    for sample in train_data:
        sample_shape   = sample["shape"]
        sample_classes = [box["class"] for box in sample["boxes"]]

        # Make sure that all images have the same shape.
        if image_shape == None:
            image_shape = sample_shape
        elif sample_shape != image_shape:
            raise Exception("Cannot have different image shapes as input.")

        # Store all the available classes in a dictionary.
        for c in sample_classes:
            if not c in classes:
                classes_count += 1
                classes[c] = classes_count

    # Swap the width and the height, as required by the data generator.
    image_shape = (image_shape[1], image_shape[0], image_shape[2])

    return train_data, test_data, classes, image_shape

def create_model(options):
    # Store the image data.
    image = keras.layers.Input(image_batch[0].shape)

    # Store the ground-truth bounding boxes.
    gt_boxes = keras.layers.Input((None, 5))

    # Store a tuple containing the rescaled image height, width, and scale.
    metadata = keras.layers.Input((3,))

    # Creates the feature maps.
    y = create_feature_maps(image, options)

    # Store the anchors at each pixel location in the feature map.
    deltas = keras.layers.Conv2D(9 * 4, (1, 1))(y)

    # Store the anchor scores at each pixel location in the feature map.
    scores = keras.layers.Conv2D(9 * 2, (1, 1), activation="sigmoid")(y)

    # Compute the labels and regressions at each pixel location in the feature map.
    anchor_target_layer      = keras_rcnn.layers.object_detection._anchor_target.AnchorTarget()
    labels, bbox_reg_targets = anchor_target_layer([scores, gt_boxes, metadata])

    # Create the classification and regression losses.
    classification = keras_rcnn.layers.losses._rpn.RPNClassificationLoss(9)([scores, labels])
    regression     = keras_rcnn.layers.losses._rpn.RPNRegressionLoss(9)([deltas, bbox_reg_targets, labels])

    return keras.models.Model([image, gt_boxes, metadata], [classification, regression])

if __name__ == "__main__":
    configuration = tensorflow.ConfigProto()
    configuration.gpu_options.allow_growth  = True
    configuration.gpu_options.visible_device_list = "0"

    session = tensorflow.Session(config=configuration)
    keras.backend.set_session(session)

    train_data, test_data, classes, image_shape = load_dataset()

    generator = keras_rcnn.preprocessing.ObjectDetectionGenerator()
    generator = generator.flow(train_data, classes, image_shape)
    [image_batch, gt_boxes_batch, metadata], _ = generator.next()

    print("Using batch size: ", image_batch.shape[0])

    #skimage.io.imshow(image_batch[0])
    #skimage.io.show()

    options = {
        "activation": "relu",
        "kernel_size": (3, 3),
        "padding": "same"
    }

    model = create_model(options)
    model.compile("adam", [None, None])
    model.fit([image_batch, gt_boxes_batch, metadata], epochs=100)
