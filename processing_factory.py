# -*- coding: utf-8 -*-
"""Data Augmentation.
"""

import tensorflow as tf

def tf_random_stretch():
    # TODO
    return

def tf_random_translate():
    #TODO
    return

def distort_color(image,color_ordering=0,scope=None):
    """Distort the color of a Tensor image.
    Each color distortion is non-commutative and thus ordering of the color ops
    matters. Ideally we would randomly permute the ordering of the color ops.
    Rather then adding that level of complication, we select a distinct ordering
    of color ops for each preprocessing thread.

    Args:
        image: 3-D Tensor containing single image in [0,1]
        color_ordering: Python int, a type of distortion (valid values: 0-3)
    Returns:
        3-D Tensor color-distorted image on range [0,1]
    Raises:
        ValueError: if color_ordering not in [0,3]
    """
    with tf.name_scope(scope,"distort_color",[image]):
        if color_ordering == 0:
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        elif color_ordering == 1:
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)
        elif color_ordering == 2:
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        elif color_ordering == 3:
            image = tf.image.random_hue(image, max_delta=0.2)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
        else:
            raise ValueError('color_ordering must be in [0, 3]')
    return tf.clip_by_value(image,0.0,1.0)

def distort_bounding_box_crop(image,bbox,
            min_object_covered=0.3,
            aspect_ratio_range=(0.9,1.1),
            area_range=200,
            clip_bbox=True,
            scope=None):
    """Generates cropped_image using a one of the bboxes randomly distorted.
    """
    with tf.name_scope(scope,"distorted_bounding_box_crop",[image,bbox]):
        # Each bounding box has shape [1,num_boxes,box coords]
        # the coordinates are ordered [ymin,xmin,ymax,xmax]
        # TODO
        pass
    return

def preprocess_for_train(image,bbox,scope="preprocessing_train"):
    """Preprocess the given image for training.
    Args:
        image: A `Tensor` representing an image of 321 x 321 size.
        bbox: A list of `Tensor` representing bounding box in the image, 
            specifying [xmin,xmax,ymin,ymax].
    
    Returns:
        A processed image and its object bounding box.
    """
    # TODO
    pass





